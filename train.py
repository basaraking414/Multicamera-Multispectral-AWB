import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config, build_lr_scheduler
from dataloader import AWBDataset
from geometry_utils import safe_inv_ccm
from loss import total_loss, build_srgb_gt, srgb_gamma, XYZ_TO_SRGB
from model import AWBTransformer
from visualization import save_debug_scene, save_mcs_alignment_debug


def train(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    # 设置随机种子确保可复现
    random.seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    torch.manual_seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.project.seed)

    dataset = AWBDataset(
        cfg.data.root_dir,
        img_size=cfg.data.img_size,
        mcs_size=cfg.data.mcs_size,
        img_dir_name=cfg.data.image_processed_dir,
        mcs_dir_name=cfg.data.mcs_npy_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.scene_batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.training.num_workers,
    )

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
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"[Resume] Loaded from {resume_path} at epoch {start_epoch}")

    debug_dir = os.path.join(cfg.debug.output_dir, "train")

    # 从 config 读取后裁剪参数
    loss_crop_size = getattr(cfg.training, "loss_crop_size", 64)
    smoothness_weight = getattr(cfg.training, "smoothness_weight", 0.0)
    srgb_weight = cfg.training.loss_weights.get("srgb", 0)
    use_srgb_loss = predict_ccm and srgb_weight > 0

    metric_names = ["total", "awb", "reconstruction", "consistency"]
    if use_srgb_loss:
        metric_names.append("srgb")
    if smoothness_weight > 0:
        metric_names.append("smoothness")

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
                gt_srgb = build_srgb_gt(flat_image, flat_gt_gain, flat_ccm1, flat_ccm2)

                # 模型预测的 CCM（用 ccm1 做基准）
                ccm1_inv = safe_inv_ccm(flat_ccm1)
                effective_ccm = ccm1_inv + ccm_delta  # [B*S, 3, 3]

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
                weights=cfg.training.loss_weights,
                crop_ratios=batch["crop_ratio"],
                loss_crop_size=loss_crop_size,
                pred_srgb=pred_srgb,
                gt_srgb=gt_srgb,
                raw_image=flat_image,
                smoothness_weight=smoothness_weight,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for name in epoch_metrics:
                epoch_metrics[name] += float(losses[name].item())

            if step == 0 and (epoch == 0 or (epoch + 1) % cfg.debug.save_interval == 0):
                save_debug_scene(
                    debug_dir,
                    identifier=epoch,
                    input_image=batch["image"][0],
                    pred_image=scene_pred_image[0],
                    gt_image=batch["gt_image"][0],
                    prefix="epoch",
                )
                save_mcs_alignment_debug(
                    debug_dir,
                    identifier=epoch,
                    batch={k: v.detach().cpu() for k, v in batch.items()},
                    prefix="epoch",
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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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
