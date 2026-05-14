"""评估脚本 —— 在测试集上评估模型性能。

计算与训练一致的 loss 指标（angular, reconstruction, consistency, 可选 srgb）。
无梯度更新，无 checkpoint 保存，仅评估和记录。

用法：
    python eval.py                              # 默认配置
    python eval.py --config config.yaml         # 指定配置文件
    python eval.py --checkpoint path/to/model.pth  # 指定模型
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from dataloader import AWBDataset
from geometry_utils import safe_inv_ccm, srgb_gamma, XYZ_TO_SRGB
from loss import total_loss, build_srgb_gt
from model import AWBTransformer
from visualization import save_debug_scene, save_mcs_alignment_debug


def evaluate(config_path: str = "config.yaml", checkpoint_path: str = None):
    cfg = load_config(config_path)

    test_dir = cfg.test.root_dir
    if not os.path.exists(test_dir):
        print(f"[Eval] Test directory not found: {test_dir}")
        print("[Eval] Set test.root_dir in config.yaml and ensure the data exists.")
        return

    dataset = AWBDataset(
        test_dir,
        img_size=cfg.test.img_size,
        mcs_size=cfg.test.mcs_size,
        img_dir_name=cfg.data.image_processed_dir,
        mcs_dir_name=cfg.data.mcs_npy_dir,
    )
    if len(dataset) == 0:
        print(f"[Eval] No scenes found in {test_dir}")
        return
    loader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False)
    num_scenes = len(dataset)
    print(f"[Eval] Loaded {num_scenes} scenes from {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 加载模型 =====
    ckpt_path = checkpoint_path or cfg.test.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[Eval] Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = ckpt.get("config", {})
    predict_ccm = model_cfg.get("predict_ccm", False)

    model = AWBTransformer(
        dim=model_cfg.get("dim", cfg.model.dim),
        num_heads=model_cfg.get("num_heads", cfg.model.num_heads),
        grid_size=model_cfg.get("grid_size", cfg.model.grid_size),
        use_positional_encoding=model_cfg.get("use_positional_encoding", cfg.model.use_positional_encoding),
        focal_embed_dim=model_cfg.get("focal_embed_dim", 16),
        predict_ccm=predict_ccm,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"[Eval] Loaded model from {ckpt_path} (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")

    # ===== Loss 配置（与训练一致）=====
    loss_crop_size = getattr(cfg.test, "loss_crop_size", None) or cfg.training.loss_crop_size
    loss_weights = cfg.training.loss_weights
    smoothness_weight = getattr(cfg.training, "smoothness_weight", 0.0)
    use_srgb_loss = predict_ccm and loss_weights.get("srgb", 0) > 0

    metric_names = ["total", "awb", "reconstruction", "consistency"]
    if use_srgb_loss:
        metric_names.append("srgb")
    if smoothness_weight > 0:
        metric_names.append("smoothness")

    total_metrics = {k: 0.0 for k in metric_names}
    per_scene_results = []

    output_dir = cfg.test.output_dir
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # ===== 评估循环 =====
    with torch.no_grad():
        for step, batch in enumerate(loader):
            for key, value in batch.items():
                batch[key] = value.to(device)

            batch_size, scene_size = batch["image"].shape[:2]

            flat_image = batch["image"].reshape(batch_size * scene_size, *batch["image"].shape[2:])
            flat_mcs = batch["mcs"].reshape(batch_size * scene_size, *batch["mcs"].shape[2:])
            flat_ccm1 = batch["ccm1"].reshape(batch_size * scene_size, 3, 3)
            flat_ccm2 = batch["ccm2"].reshape(batch_size * scene_size, 3, 3)
            flat_gt_gain = batch["awb_gt_gain"].reshape(batch_size * scene_size, 3)
            flat_gt_image = batch["gt_image"].reshape(batch_size * scene_size, *batch["gt_image"].shape[2:])
            flat_focal = batch["focal_length"].reshape(-1)

            # ===== 模型前向 =====
            pred_gain, ccm_delta = model(flat_image, flat_mcs, flat_ccm1, flat_ccm2, flat_focal)
            pred_image = torch.clamp(pred_gain * flat_image, 0.0, 1.0)

            scene_pred_image = pred_image.reshape(batch_size, scene_size, *pred_image.shape[1:])

            # ===== 可选：sRGB 路径 =====
            pred_srgb = None
            gt_srgb = None
            if use_srgb_loss and ccm_delta is not None:
                gt_srgb = build_srgb_gt(flat_image, flat_gt_gain, flat_ccm1, flat_ccm2)

                ccm1_inv = safe_inv_ccm(flat_ccm1)
                effective_ccm = ccm1_inv + ccm_delta

                B_flat, H, W, _ = flat_image.shape
                corrected_flat = pred_image.reshape(B_flat, -1, 3)
                xyz = torch.bmm(corrected_flat, effective_ccm.transpose(1, 2))
                xyz_to_srgb = XYZ_TO_SRGB.to(device)
                linear_srgb = torch.bmm(xyz, xyz_to_srgb.T.unsqueeze(0).expand(B_flat, -1, -1))
                linear_srgb = linear_srgb.reshape(B_flat, H, W, 3).clamp(0.0, 1.0)
                pred_srgb = srgb_gamma(linear_srgb)

            # ===== Loss =====
            losses = total_loss(
                pred_gain=pred_gain,
                gt_gain=flat_gt_gain,
                pred_image=pred_image,
                gt_image=flat_gt_image,
                scene_pred_image=scene_pred_image,
                weights=loss_weights,
                crop_ratios=batch["crop_ratio"],
                loss_crop_size=loss_crop_size,
                pred_srgb=pred_srgb,
                gt_srgb=gt_srgb,
                raw_image=flat_image,
                smoothness_weight=smoothness_weight,
            )

            scene_loss = {k: float(v.item()) for k, v in losses.items()}
            for name in total_metrics:
                total_metrics[name] += scene_loss[name]

            scene_id = int(batch["scene_id"][0, 0].item()) if batch["scene_id"].numel() > 0 else step
            per_scene_results.append((scene_id, scene_loss))

            # ===== Debug 可视化 =====
            save_debug_scene(
                debug_dir,
                identifier=scene_id,
                input_image=batch["image"][0],
                pred_image=scene_pred_image[0],
                gt_image=batch["gt_image"][0],
                prefix="scene",
            )
            save_mcs_alignment_debug(
                debug_dir,
                identifier=scene_id,
                batch={k: v.detach().cpu() for k, v in batch.items()},
                prefix="scene",
            )

    # ===== 汇总结果 =====
    avg_metrics = {k: v / max(num_scenes, 1) for k, v in total_metrics.items()}

    print("\n" + "=" * 60)
    print(f"[Eval] Results on {num_scenes} scenes from {test_dir}")
    print(f"       Checkpoint: {ckpt_path}")
    print("=" * 60)
    for name in metric_names:
        print(f"  {name:20s} = {avg_metrics[name]:.6f}")
    print("=" * 60)

    # ===== 保存结果到文件 =====
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Eval Results\n")
        f.write(f"  Dataset: {test_dir}\n")
        f.write(f"  Checkpoint: {ckpt_path}\n")
        f.write(f"  Scenes: {num_scenes}\n")
        f.write(f"  Model epoch: {ckpt['epoch']}\n\n")
        f.write(f"{'Metric':20s} {'Value':>12s}\n")
        f.write("-" * 32 + "\n")
        for name in metric_names:
            f.write(f"{name:20s} {avg_metrics[name]:12.6f}\n")

    per_scene_path = os.path.join(output_dir, "per_scene.txt")
    with open(per_scene_path, "w") as f:
        header = f"{'Scene':>8s}"
        for name in metric_names:
            header += f"  {name:>14s}"
        f.write(header + "\n")
        f.write("-" * (8 + 16 * len(metric_names)) + "\n")
        for scene_id, scene_loss in sorted(per_scene_results, key=lambda x: x[0]):
            line = f"{scene_id:>8d}"
            for name in metric_names:
                line += f"  {scene_loss[name]:14.6f}"
            f.write(line + "\n")

    print(f"[Eval] Summary saved to {summary_path}")
    print(f"[Eval] Per-scene results saved to {per_scene_path}")
    print(f"[Eval] Debug visualizations saved to {debug_dir}")


def main():
    parser = argparse.ArgumentParser(description="AWB Transformer Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型checkpoint路径（覆盖config中的设置）")
    args = parser.parse_args()

    evaluate(config_path=args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
