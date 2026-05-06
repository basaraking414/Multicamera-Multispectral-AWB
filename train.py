import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config, build_lr_scheduler
from dataloader import AWBDataset
from gt_utils import _auto_expose, _ensure_float01
from loss import total_loss, build_srgb_gt
from model import AWBTransformer

SENSOR_NAMES = ["tele", "main", "wide"]


def _save_debug_scene(
    save_dir: str,
    epoch: int,
    input_image: torch.Tensor,
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    def to_uint8(tensor: torch.Tensor, display_scale: float) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        array = _ensure_float01(array)
        array = np.clip(array / max(display_scale, 1e-4), 0.0, 1.0)
        array = _auto_expose(array, percentile=100.0)
        return (array[..., ::-1] * 255.0).astype(np.uint8)

    scene_panels = []
    scene_stack = torch.cat([input_image, pred_image, gt_image], dim=0).detach().cpu().numpy()
    display_scale = float(np.percentile(np.clip(scene_stack, 0.0, 1.0), 99.5))
    for idx in range(input_image.shape[0]):
        panel = np.concatenate(
            [
                to_uint8(input_image[idx], display_scale),
                to_uint8(pred_image[idx], display_scale),
                to_uint8(gt_image[idx], display_scale),
            ],
            axis=1,
        )
        sensor_name = SENSOR_NAMES[idx] if idx < len(SENSOR_NAMES) else f"sensor_{idx}"
        cv2.putText(
            panel,
            f"{sensor_name}: input | pred | gt",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        scene_panels.append(panel)

    debug_image = np.concatenate(scene_panels, axis=0)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch:03d}.png"), debug_image)


def _save_mcs_alignment_debug(
    save_dir: str,
    epoch: int,
    batch: dict,
) -> None:
    """可视化 MCS 与 RAW 的空间对齐情况，用于调试数据流是否正确。

    每张图包含 3 行（tele/main/wide），每行 3 列：
      - input:      RAW 图像（自动曝光）
      - mcs_rgb:    MCS 前 3 通道伪彩色
      - conf_overlay: 置信度热力图叠加到 input 上（红色=外推区域，蓝色=真实数据）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取 batch 中第一个 scene
    images = batch["image"][0].detach().cpu().numpy()        # [S, H, W, 3]
    mcs = batch["mcs"][0].detach().cpu().numpy()              # [S, H, W, 10]
    focal_lengths = batch["focal_length"][0].detach().cpu().numpy()  # [S]

    rows = []
    for idx in range(images.shape[0]):
        sensor_name = SENSOR_NAMES[idx] if idx < len(SENSOR_NAMES) else f"sensor_{idx}"
        focal = float(focal_lengths[idx])

        # ===== 第 1 列：RAW 输入 =====
        raw_img = _ensure_float01(images[idx])
        raw_img = _auto_expose(raw_img, percentile=100.0)
        raw_bgr = (raw_img[..., ::-1] * 255).astype(np.uint8)

        # ===== 第 2 列：MCS 前 3 通道伪彩色 =====
        mcs_rgb = mcs[idx, :, :, :3].copy()                   # [H, W, 3]
        # 逐通道归一化到 [0, 1] 以显示细节
        for c in range(3):
            ch = mcs_rgb[..., c]
            lo, hi = float(ch.min()), float(ch.max())
            if hi - lo > 1e-6:
                mcs_rgb[..., c] = (ch - lo) / (hi - lo)
            else:
                mcs_rgb[..., c] = 0.0
        mcs_bgr = (mcs_rgb[..., ::-1] * 255).astype(np.uint8)

        # ===== 第 3 列：置信度热力图叠加 =====
        confidence = mcs[idx, :, :, 9]                         # [H, W]
        # 将置信度映射为彩色热力图（蓝→青→黄→红）
        conf_colormap = cv2.applyColorMap(
            (confidence * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        # 半透明叠加到 input 上：alpha 随置信度变化（低置信度区域更明显）
        alpha = 0.6 * (1.0 - confidence)                       # 外推区域红色更明显
        overlay = (raw_bgr * (1 - alpha[..., None]) + conf_colormap * alpha[..., None]).astype(np.uint8)

        # ===== 拼接成一行 =====
        row = np.concatenate([raw_bgr, mcs_bgr, overlay], axis=1)
        cv2.putText(
            row,
            f"{sensor_name} (f={focal:.0f}mm):  input  |  mcs_rgb(ch0,1,2)  |  conf_overlay",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        rows.append(row)

    debug_image = np.concatenate(rows, axis=0)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch:03d}_mcs_alignment.png"), debug_image)


def train(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    dataset = AWBDataset(cfg.data.root_dir, img_size=cfg.data.img_size, mcs_size=cfg.data.mcs_size)
    loader = DataLoader(dataset, batch_size=cfg.training.scene_batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从配置读取焦距编码器参数
    focal_embed_dim = getattr(cfg.model, "focal_embed_dim", 16)
    predict_ccm = getattr(cfg.model, "predict_ccm", False)

    model = AWBTransformer(
        dim=cfg.model.dim,
        num_heads=cfg.model.num_heads,
        grid_size=cfg.model.grid_size,
        use_positional_encoding=cfg.model.use_positional_encoding,
        focal_embed_dim=focal_embed_dim,
        predict_ccm=predict_ccm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = build_lr_scheduler(optimizer, cfg.training)

    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0

    # ===== 断点续训 =====
    resume_path = cfg.checkpoint.resume_from
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"[Resume] Loaded from {resume_path} at epoch {start_epoch}")

    debug_dir = os.path.join(cfg.debug.output_dir, "train")

    # 从 config 读取后裁剪参数
    loss_crop_size = getattr(cfg.training, "loss_crop_size", 64)
    srgb_weight = cfg.training.loss_weights.get("srgb", 0)
    use_srgb_loss = predict_ccm and srgb_weight > 0

    metric_names = ["total", "awb", "reconstruction", "consistency"]
    if use_srgb_loss:
        metric_names.append("srgb")

    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_metrics = {k: 0.0 for k in metric_names}
        model.train()

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

            # ===== 模型前向（仅用 focal_length，无需 sensor_id）=====
            pred_gain, ccm_delta = model(flat_image, flat_mcs, flat_ccm1, flat_ccm2, flat_focal)
            pred_image = torch.clamp(pred_gain * flat_image, 0.0, 1.0)

            scene_pred_image = pred_image.reshape(batch_size, scene_size, *pred_image.shape[1:])

            # ===== 可选：sRGB 路径 =====
            pred_srgb = None
            gt_srgb = None
            if use_srgb_loss and ccm_delta is not None:
                # 预测 sRGB: 使用预测的 gain + ccm_delta
                scene_ccm1 = batch["ccm1"].reshape(batch_size * scene_size, 3, 3)
                scene_ccm2 = batch["ccm2"].reshape(batch_size * scene_size, 3, 3)
                gt_srgb = build_srgb_gt(flat_image, flat_gt_gain, scene_ccm1, scene_ccm2)

                # 模型预测的 CCM
                ccm1_inv = torch.linalg.inv(scene_ccm1.float())
                ccm2_inv = torch.linalg.inv(scene_ccm2.float())
                ccm_inv = (ccm1_inv + ccm2_inv) / 2.0
                effective_ccm = ccm_inv + ccm_delta  # [B*S, 3, 3]

                from loss import srgb_gamma
                B_flat, H, W, _ = flat_image.shape
                rgb_flat = flat_image.reshape(B_flat, -1, 3)
                corrected_flat = pred_image.reshape(B_flat, -1, 3)
                xyz = torch.bmm(corrected_flat, effective_ccm.transpose(1, 2))
                xyz_to_srgb = torch.tensor(
                    [[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]],
                    dtype=torch.float32, device=device,
                )
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
                weights=cfg.training.loss_weights,
                crop_ratios=batch["crop_ratio"],
                loss_crop_size=loss_crop_size,
                pred_srgb=pred_srgb,
                gt_srgb=gt_srgb,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for name in epoch_metrics:
                epoch_metrics[name] += float(losses[name].item())

            if step == 0 and (epoch == 0 or (epoch + 1) % cfg.debug.save_interval == 0):
                _save_debug_scene(
                    debug_dir,
                    epoch=epoch,
                    input_image=batch["image"][0],
                    pred_image=scene_pred_image[0],
                    gt_image=batch["gt_image"][0],
                )
                _save_mcs_alignment_debug(
                    debug_dir,
                    epoch=epoch,
                    batch={k: v.detach().cpu() for k, v in batch.items()},
                )

        # ===== LR Scheduler 更新 =====
        if scheduler is not None:
            if cfg.training.lr_scheduler.lower() == "plateau":
                scheduler.step(epoch_metrics["total"] / max(len(loader), 1))
            else:
                scheduler.step()

        num_steps = max(len(loader), 1)
        epoch_total_loss = epoch_metrics["total"] / num_steps
        current_lr = optimizer.param_groups[0]["lr"]

        # 日志
        parts = [f"Epoch {epoch:03d} | total={epoch_total_loss:.4f}"]
        for name in metric_names:
            if name == "total":
                continue
            parts.append(f"{name}={epoch_metrics[name] / num_steps:.4f}")
        parts.append(f"lr={current_lr:.2e}")
        print(" ".join(parts))

        # ===== Checkpoint 保存 =====
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_total_loss,
            "config": {
                "dim": cfg.model.dim,
                "grid_size": cfg.model.grid_size,
                "num_heads": cfg.model.num_heads,
                "use_positional_encoding": cfg.model.use_positional_encoding,
                "focal_embed_dim": focal_embed_dim,
                "predict_ccm": predict_ccm,
            },
        }

        latest_path = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.latest)
        torch.save(checkpoint_state, latest_path)

        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            best_path = os.path.join(cfg.checkpoint.dir, cfg.checkpoint.best_model)
            torch.save(checkpoint_state, best_path)
            print(f"  ==> New best model: total_loss={epoch_total_loss:.6f}")

        if (epoch + 1) % cfg.checkpoint.interval == 0:
            periodic_path = os.path.join(
                cfg.checkpoint.dir, f"model_epoch_{epoch+1:03d}_loss_{epoch_total_loss:.4f}.pth"
            )
            torch.save(checkpoint_state, periodic_path)
            print(f"  ==> Periodic checkpoint saved: {periodic_path}")


if __name__ == "__main__":
    train()
