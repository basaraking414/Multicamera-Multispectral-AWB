from typing import Dict, Optional

import torch
import torch.nn.functional as F


# =============================================================================
# 重叠区域裁剪工具（后裁剪）
# =============================================================================
def crop_to_overlap(
    images: torch.Tensor,
    crop_ratios: torch.Tensor,
    target_size: int = 64,
    min_size: int = 4,
) -> torch.Tensor:
    """将每个 sensor 图像裁剪到三摄共同视场区域，并 resize 到统一尺寸。

    Args:
        images:     [B, S, H, W, C] 每场景 3 张预测/GT 图像
        crop_ratios:[B, S]          每张图的裁剪比例（焦距比）
        target_size:int             裁剪后 resize 的统一尺寸
    Returns:
        [B, S, target_size, target_size, C] 裁剪并 resize 后的图像
    """
    B, S, H, W, C = images.shape
    device = images.device
    cropped_list = []

    for b in range(B):
        scene_crops = []
        for s in range(S):
            ratio = float(crop_ratios[b, s].item())
            crop_h = max(min_size, min(H, int(round(H * ratio))))
            crop_w = max(min_size, min(W, int(round(W * ratio))))
            y0 = max(0, (H - crop_h) // 2)
            x0 = max(0, (W - crop_w) // 2)
            crop = images[b, s, y0:y0 + crop_h, x0:x0 + crop_w, :]  # [crop_h, crop_w, C]
            # Resize 到统一尺寸
            crop = crop.permute(2, 0, 1).unsqueeze(0)  # [1, C, crop_h, crop_w]
            crop = F.interpolate(crop, size=(target_size, target_size), mode='bilinear', align_corners=False)
            crop = crop.squeeze(0).permute(1, 2, 0)  # [target, target, C]
            scene_crops.append(crop)
        cropped_list.append(torch.stack(scene_crops, dim=0))  # [S, target, target, C]

    return torch.stack(cropped_list, dim=0)  # [B, S, target, target, C]


# =============================================================================
# sRGB Gamma 校正
# =============================================================================
def srgb_gamma(linear: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """线性光 → sRGB gamma 校正（标准 sRGB 分段曲线）。"""
    mask = linear <= 0.0031308
    srgb = torch.where(
        mask,
        12.92 * linear,
        1.055 * torch.clamp(linear, min=eps).pow(1.0 / 2.4) - 0.055,
    )
    return torch.clamp(srgb, 0.0, 1.0)


# =============================================================================
# 损失函数
# =============================================================================
def _spatial_mean_gain(pred_gain: torch.Tensor) -> torch.Tensor:
    if pred_gain.dim() == 4:
        return pred_gain.mean(dim=(1, 2))
    return pred_gain


def angular_loss(pred_gain: torch.Tensor, gt_gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_vec = F.normalize(_spatial_mean_gain(pred_gain), dim=-1, eps=eps)
    gt_vec = F.normalize(gt_gain, dim=-1, eps=eps)
    cosine = torch.clamp((pred_vec * gt_vec).sum(dim=-1), -1.0 + eps, 1.0 - eps)
    return torch.acos(cosine).mean()


def reconstruction_loss(pred_image: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_image, gt_image)


def scene_consistency_loss(pred_image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_image: [B, S, H, W, 3]
    约束同场景内三摄的白平衡结果色度一致。
    """
    scene_mean_rgb = pred_image.mean(dim=(2, 3))
    chroma = scene_mean_rgb / torch.clamp(scene_mean_rgb.sum(dim=-1, keepdim=True), min=eps)
    scene_center = chroma.mean(dim=1, keepdim=True)
    return torch.mean(torch.abs(chroma - scene_center))


def srgb_loss(
    pred_srgb: torch.Tensor,
    gt_srgb: torch.Tensor,
) -> torch.Tensor:
    """在 sRGB 空间中的图像重建损失。"""
    return F.l1_loss(pred_srgb, gt_srgb)


def build_srgb_gt(
    raw: torch.Tensor,
    gt_gain: torch.Tensor,
    ccm1: torch.Tensor,
    ccm2: torch.Tensor,
) -> torch.Tensor:
    """从已知 CCM 和 GT gain 合成 sRGB ground truth。

    流程: raw * gain → cameraRGB → XYZ → linear sRGB → gamma → sRGB
    """
    # cameraRGB → XYZ: inverse of xyz2camera_rgb
    ccm1_inv = torch.linalg.inv(ccm1.float())
    ccm2_inv = torch.linalg.inv(ccm2.float())
    # 使用两个 CCM 的平均（或按场景光照选择，这里简化为平均）
    ccm_inv = (ccm1_inv + ccm2_inv) / 2.0  # [B, 3, 3]

    # AWB 校正
    corrected = raw * gt_gain.unsqueeze(1).unsqueeze(2)  # [B, H, W, 3]

    # XYZ → linear sRGB 标准矩阵
    xyz_to_srgb = torch.tensor(
        [[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570]],
        dtype=torch.float32, device=raw.device,
    )

    # cameraRGB → linear sRGB
    B, H, W, _ = corrected.shape
    rgb_flat = corrected.reshape(B, -1, 3)  # [B, H*W, 3]
    # ccm_inv @ rgb_flat → XYZ
    xyz = torch.bmm(rgb_flat, ccm_inv.transpose(1, 2))  # [B, H*W, 3]
    # XYZ → linear sRGB
    linear_srgb = torch.bmm(xyz, xyz_to_srgb.T.unsqueeze(0).expand(B, -1, -1))  # [B, H*W, 3]
    linear_srgb = linear_srgb.reshape(B, H, W, 3)
    linear_srgb = torch.clamp(linear_srgb, 0.0, 1.0)

    # gamma 校正
    return srgb_gamma(linear_srgb)


def total_loss(
    pred_gain: torch.Tensor,
    gt_gain: torch.Tensor,
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    scene_pred_image: torch.Tensor,
    weights: Dict[str, float],
    crop_ratios: Optional[torch.Tensor] = None,
    loss_crop_size: int = 64,
    # sRGB 相关（可选）
    pred_srgb: Optional[torch.Tensor] = None,
    gt_srgb: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """聚合所有损失。

    Args:
        pred_gain:    [B*S, ...] 预测 gain（angular_loss 直接使用）
        gt_gain:      [B*S, 3]
        pred_image:   [B*S, H, W, 3] 预测校正图（reconstruction_loss 使用）
        gt_image:     [B*S, H, W, 3] GT 校正图
        scene_pred_image: [B, S, H, W, 3] 场景分组（consistency_loss 使用）
        weights:      损失权重 dict
        crop_ratios:  [B, S] 后裁剪比例，None 则不做裁剪
        loss_crop_size: 后裁剪的统一尺寸
        pred_srgb:    [B*S, H, W, 3] 预测 sRGB（可选）
        gt_srgb:      [B*S, H, W, 3] GT sRGB（可选）
    """
    awb = angular_loss(pred_gain, gt_gain)

    # 后裁剪：在重叠区域计算 reconstruction + consistency
    if crop_ratios is not None:
        B, S = scene_pred_image.shape[:2]
        H, W, C = scene_pred_image.shape[2:]

        # pred_image 和 gt_image 都是 [B*S, H, W, 3]，需要 reshape 到 [B, S, H, W, 3]
        pred_reshape = pred_image.reshape(B, S, H, W, C)
        gt_reshape = gt_image.reshape(B, S, H, W, C)

        pred_cropped = crop_to_overlap(pred_reshape, crop_ratios, loss_crop_size)  # [B, S, t, t, 3]
        gt_cropped = crop_to_overlap(gt_reshape, crop_ratios, loss_crop_size)      # [B, S, t, t, 3]

        rec = reconstruction_loss(pred_cropped, gt_cropped)
        consistency = scene_consistency_loss(pred_cropped)
    else:
        rec = reconstruction_loss(pred_image, gt_image)
        consistency = scene_consistency_loss(scene_pred_image)

    total = weights["awb"] * awb + weights["reconstruction"] * rec + weights["consistency"] * consistency

    result = {
        "awb": awb,
        "reconstruction": rec,
        "consistency": consistency,
        "total": total,
    }

    # sRGB 损失（可选）
    if pred_srgb is not None and gt_srgb is not None:
        srgb = srgb_loss(pred_srgb, gt_srgb)
        srgb_weight = weights.get("srgb", 1.0)
        result["srgb"] = srgb
        result["total"] = total + srgb_weight * srgb

    return result
