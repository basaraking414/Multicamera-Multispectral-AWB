"""推理脚本 —— 使用训练好的模型对单张/多张图像进行 AWB 校正。

用法：
    python infer.py                          # 使用 config.yaml 默认配置
    python infer.py --config my_config.yaml  # 指定配置文件
    python infer.py --npz data/image_processed/IMG*.npz --mcs data/Mcsnpy/IMG*.npy  # 指定输入文件
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from geometry_utils import align_mcs_to_fov
from gt_utils import _auto_expose, _ensure_float01
from model import AWBTransformer


SENSOR_NAMES = ["tele", "main", "wide"]


def load_model(cfg, device: torch.device) -> AWBTransformer:
    """从 checkpoint 加载模型权重。"""
    ckpt_path = cfg.inference.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt.get("config", {})

    model = AWBTransformer(
        dim=model_cfg.get("dim", cfg.model.dim),
        num_heads=model_cfg.get("num_heads", cfg.model.num_heads),
        grid_size=model_cfg.get("grid_size", cfg.model.grid_size),
        use_positional_encoding=model_cfg.get("use_positional_encoding", cfg.model.use_positional_encoding),
        predict_ccm=model_cfg.get("predict_ccm", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Inference] Loaded model from {ckpt_path} (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")
    return model


def load_single_sample(npz_path: str, npy_path: str, img_size, mcs_size,
                       sensor_id: int = 1, ref_focal: float = None):
    """加载单个 npz + npy 样本，返回原始图像 + 模型输入张量。

    流程：保留原始尺寸图像用于最终输出，同时生成缩放后的张量供模型推理。

    Args:
        sensor_id: 0=tele, 1=main, 2=wide
        ref_focal: 主摄焦距。None 时不做 MCS 空间对齐（置信度全 1）。
    """
    img_data = np.load(npz_path, allow_pickle=True)
    mcs_data = np.load(npy_path).astype(np.float32)

    # 原始尺寸图像（用于最终输出）
    if "image_full" in img_data.files:
        image_orig = img_data["image_full"].astype(np.float32)
    else:
        image_orig = img_data["image"].astype(np.float32)

    # 缩放后的图像（供模型推理）
    image_resized = cv2.resize(image_orig, img_size[::-1], interpolation=cv2.INTER_AREA)

    # MCS 缩放
    mcs_tensor = torch.from_numpy(mcs_data).permute(2, 0, 1).unsqueeze(0)
    mcs_tensor = F.interpolate(mcs_tensor, size=mcs_size, mode='area')
    mcs = mcs_tensor[0].permute(1, 2, 0).numpy().astype(np.float32)

    # MCS 空间对齐（与训练时 dataloader 行为一致）
    focal_length = float(
        img_data["focal_length_35mm"]
        if "focal_length_35mm" in img_data
        else img_data.get("focal_length", 50.0)
    )
    if ref_focal is not None and ref_focal > 0:
        align_ratio = ref_focal / max(focal_length, 1e-4)
        mcs_aligned, confidence = align_mcs_to_fov(mcs, align_ratio, mcs_size)
    else:
        mcs_aligned, confidence = mcs, np.ones(mcs.shape[:2], dtype=np.float32)
    mcs = np.concatenate([mcs_aligned, confidence[..., None]], axis=-1).astype(np.float32)

    ccm1 = img_data["xyz2camera_rgb1"].astype(np.float32)
    ccm2 = img_data["xyz2camera_rgb2"].astype(np.float32)

    # 增加 batch 维度
    image_t = torch.from_numpy(image_resized).unsqueeze(0)
    mcs_t = torch.from_numpy(mcs).unsqueeze(0)
    ccm1_t = torch.from_numpy(ccm1).unsqueeze(0)
    ccm2_t = torch.from_numpy(ccm2).unsqueeze(0)

    return image_orig, image_t, mcs_t, ccm1_t, ccm2_t, focal_length, img_data


def infer_one(
    model: AWBTransformer,
    image_t: torch.Tensor,
    mcs_t: torch.Tensor,
    ccm1_t: torch.Tensor,
    ccm2_t: torch.Tensor,
    device: torch.device,
    original_size: tuple,
    focal_length: float = 50.0,
) -> np.ndarray:
    """对单张图像进行推理，返回 gain map [H_orig, W_orig, 3]。

    流程：模型在小尺寸上推理出 gain map，再上采样回原始尺寸。
    """
    with torch.no_grad():
        fl_t = torch.tensor([focal_length], device=device)

        pred_gain, _ = model(
            image_t.to(device),
            mcs_t.to(device),
            ccm1_t.to(device),
            ccm2_t.to(device),
            focal_length=fl_t,
        )  # [1, H_small, W_small, 3]

    # 将 gain map 上采样到原始图像尺寸
    gain_tensor = pred_gain.permute(0, 3, 1, 2)
    gain_tensor = F.interpolate(gain_tensor, size=original_size, mode='bilinear', align_corners=False)
    gain_map = gain_tensor[0].permute(1, 2, 0).cpu().numpy()

    return gain_map


def save_results(
    output_dir: str,
    base_name: str,
    image: np.ndarray,
    gain_map: np.ndarray,
    cfg,
) -> None:
    """保存推理结果：gain map 可视化 + 校正后图像。"""
    os.makedirs(output_dir, exist_ok=True)

    # ===== Gain Map 可视化 =====
    if cfg.inference.save_gain_map:
        gain_vis = gain_map.copy()
        gain_min = gain_vis.min()
        gain_max = gain_vis.max()
        if gain_max - gain_min > 1e-6:
            gain_vis = (gain_vis - gain_min) / (gain_max - gain_min)
        gain_vis = (gain_vis * 255).astype(np.uint8)
        gain_vis_bgr = cv2.cvtColor(gain_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_gain_map.png"), gain_vis_bgr)

    # ===== 校正后图像 =====
    if cfg.inference.save_corrected_image:
        corrected = np.clip(image * gain_map, 0.0, 1.0)
        corrected_display = _auto_expose(corrected)
        corrected_bgr = (corrected_display[..., ::-1] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_corrected.png"), corrected_bgr)

    # ===== 输入图像（对比）=====
    input_display = _auto_expose(image)
    input_bgr = (input_display[..., ::-1] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_input.png"), input_bgr)

    # ===== 拼接对比图 =====
    panel = np.concatenate([input_bgr, corrected_bgr], axis=1)
    cv2.putText(panel, "input", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "AWB corrected", (panel.shape[1] // 2 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.png"), panel)

    print(f"  [Saved] {base_name} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="AWB Transformer Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--npz", type=str, nargs="+", default=None, help="输入 npz 文件路径（可多个）")
    parser.add_argument("--mcs", type=str, nargs="+", default=None, help="对应 mcs npy 文件路径")
    parser.add_argument("--ref-focal", type=float, default=None,
                        help="主摄焦距 (mm)，用于 MCS 空间对齐。不指定时不做对齐。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    output_dir = cfg.inference.output_dir
    img_size = cfg.inference.img_size
    mcs_size = cfg.inference.mcs_size

    if args.npz and args.mcs:
        assert len(args.npz) == len(args.mcs), "npz 和 mcs 文件数量必须一致"
        npz_files = args.npz
        npy_files = args.mcs
    else:
        img_dir = os.path.join(cfg.data.root_dir, cfg.data.image_processed_dir)
        mcs_dir = os.path.join(cfg.data.root_dir, cfg.data.mcs_npy_dir)
        npz_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".npz")])
        npy_files = sorted([os.path.join(mcs_dir, f) for f in os.listdir(mcs_dir) if f.endswith(".npy")])

    for npz_path, npy_path in zip(npz_files, npy_files):
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        image_orig, image_t, mcs_t, ccm1_t, ccm2_t, focal_length, img_data = \
            load_single_sample(npz_path, npy_path, img_size, mcs_size,
                               ref_focal=args.ref_focal)
        original_size = (image_orig.shape[0], image_orig.shape[1])
        gain_map = infer_one(model, image_t, mcs_t, ccm1_t, ccm2_t, device,
                             original_size, focal_length=focal_length)
        save_results(output_dir, base_name, image_orig, gain_map, cfg)

    print(f"\n[Done] All results saved to {output_dir}")


if __name__ == "__main__":
    main()
