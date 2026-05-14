from typing import Dict, Optional

import torch
import torch.nn.functional as F
import math

from geometry_utils import safe_inv_ccm, srgb_gamma, XYZ_TO_SRGB


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
# 损失函数
# =============================================================================
def angular_loss(pred_gain: torch.Tensor, gt_gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐像素角度损失：使用 atan2 计算稳定的角度误差（弧度）。

    gt_gain [B*S, 3] 通过广播与 pred_gain [B*S, H, W, 3] 逐像素比较。
    """
    pred_vec = F.normalize(pred_gain, dim=-1, eps=eps)
    gt_vec = F.normalize(gt_gain, dim=-1, eps=eps)
    # gt_vec: [B*S, 3] → [B*S, 1, 1, 3] 广播到 pred_vec 形状
    if gt_vec.dim() == 2:
        gt_vec = gt_vec.unsqueeze(1).unsqueeze(1)
    # 扩展 gt_vec 到 pred_vec 形状以计算 cross product
    gt_expanded = gt_vec.expand_as(pred_vec)
    # atan2 计算角度：cross 的模 / dot
    cross = torch.cross(pred_vec, gt_expanded, dim=-1)
    cross_norm = cross.norm(dim=-1)
    dot = (pred_vec * gt_expanded).sum(dim=-1)
    angle = torch.atan2(cross_norm, dot)  # [0, π]
    angle_deg = angle * (180.0 / math.pi)  # 转换为度
    return angle_deg.mean()


def reconstruction_loss(pred_image: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_image, gt_image)


def scene_consistency_loss(pred_image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_image: [B, S, H, W, 3] 线性RGB空间
    在 Lab 空间的 a,b 通道计算色度一致性，约束同场景三摄色度一致。
    使用欧氏距离衡量色度差异。
    """
    from geometry_utils import linear_rgb_to_lab

    # 转换到 Lab 空间
    lab = linear_rgb_to_lab(pred_image, eps=eps)  # [B, S, H, W, 3]
    # 只取 a,b 通道（色度），忽略亮度 L
    ab = lab[..., 1:]  # [B, S, H, W, 2]
    # 计算每个 sensor 的空间均值
    sensor_mean_ab = ab.mean(dim=(2, 3))  # [B, S, 2]
    # 场景中心（三摄均值）
    scene_center = sensor_mean_ab.mean(dim=1, keepdim=True)  # [B, 1, 2]
    # 欧氏距离：sqrt((a_i - a_center)^2 + (b_i - b_center)^2)
    return torch.mean(torch.norm(sensor_mean_ab - scene_center, dim=-1))


def srgb_loss(
    pred_srgb: torch.Tensor,
    gt_srgb: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CIE76 Lab 色差损失。"""
    from geometry_utils import srgb_to_lab

    pred_lab = srgb_to_lab(pred_srgb, eps=eps)
    gt_lab = srgb_to_lab(gt_srgb, eps=eps)
    # CIE76: 欧氏距离
    delta = pred_lab - gt_lab
    delta_e = torch.sqrt(torch.sum(delta ** 2, dim=-1) + eps)
    return delta_e.mean()


def spatial_smoothness_loss(
    gain_map: torch.Tensor,
    raw_image: torch.Tensor,
    edge_weight: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """边缘感知的空间平滑损失，约束 gain map 的空间连续性。

    在 raw 图像有强边缘处降低平滑惩罚（保留光照跳变边界）。
    """
    # gain 梯度
    gain_dy = gain_map[:, 1:, :, :] - gain_map[:, :-1, :, :]
    gain_dx = gain_map[:, :, 1:, :] - gain_map[:, :, :-1, :]

    # raw 灰度图梯度作为边缘感知权重
    raw_gray = raw_image.mean(dim=-1, keepdim=True)
    raw_dy = raw_gray[:, 1:, :, :] - raw_gray[:, :-1, :, :]
    raw_dx = raw_gray[:, :, 1:, :] - raw_gray[:, :, :-1, :]

    w_y = torch.exp(-edge_weight * raw_dy.abs())
    w_x = torch.exp(-edge_weight * raw_dx.abs())

    return (w_y * gain_dy.abs()).mean() + (w_x * gain_dx.abs()).mean()


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
    # 用 ccm1（典型 D65）作为主光照 CCM，避免平均两个不同光照的矩阵
    ccm_inv = safe_inv_ccm(ccm1)  # [B, 3, 3]

    # AWB 校正
    corrected = raw * gt_gain.unsqueeze(1).unsqueeze(2)  # [B, H, W, 3]

    # XYZ → linear sRGB 标准矩阵
    xyz_to_srgb = XYZ_TO_SRGB.to(raw.device)

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
    # 空间平滑（可选）
    raw_image: Optional[torch.Tensor] = None,
    smoothness_weight: float = 0.0,
    # sRGB 相关（可选）
    pred_srgb: Optional[torch.Tensor] = None,
    gt_srgb: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """聚合所有损失。

    Args:
        pred_gain:    [B*S, H, W, 3] 预测 gain（逐像素角度损失）
        gt_gain:      [B*S, 3]
        pred_image:   [B*S, H, W, 3] 预测校正图
        gt_image:     [B*S, H, W, 3] GT 校正图
        scene_pred_image: [B, S, H, W, 3] 场景分组
        weights:      损失权重 dict
        crop_ratios:  [B, S] 后裁剪比例，None 则不做裁剪
        loss_crop_size: 后裁剪的统一尺寸
        raw_image:    [B*S, H, W, 3] 原始 RAW（空间平滑的 edge-aware 用）
        smoothness_weight: 空间平滑损失权重
        pred_srgb:    [B*S, H, W, 3] 预测 sRGB（可选）
        gt_srgb:      [B*S, H, W, 3] GT sRGB（可选）
    """
    # 后裁剪：所有 loss 在重叠区域上计算
    if crop_ratios is not None:
        B, S = scene_pred_image.shape[:2]
        H, W, C = scene_pred_image.shape[2:]

        # pred_image/gt_image: [B*S, H, W, 3] → [B, S, H, W, 3] → 裁剪
        pred_reshape = pred_image.reshape(B, S, H, W, C)
        gt_reshape = gt_image.reshape(B, S, H, W, C)
        pred_cropped = crop_to_overlap(pred_reshape, crop_ratios, loss_crop_size)
        gt_cropped = crop_to_overlap(gt_reshape, crop_ratios, loss_crop_size)

        # 也对 pred_gain 做裁剪，使所有 loss 作用在同一空间区域
        gain_reshape = pred_gain.reshape(B, S, *pred_gain.shape[1:])
        gain_cropped = crop_to_overlap(gain_reshape, crop_ratios, loss_crop_size)
        # 展平回 [B*S, t, t, 3] 用于 angular loss
        _, S, t, _, _ = gain_cropped.shape
        gain_flat = gain_cropped.reshape(B * S, t, t, 3)

        rec = reconstruction_loss(pred_cropped, gt_cropped)
        consistency = scene_consistency_loss(pred_cropped)
        awb = angular_loss(gain_flat, gt_gain)
    else:
        rec = reconstruction_loss(pred_image, gt_image)
        consistency = scene_consistency_loss(scene_pred_image)
        awb = angular_loss(pred_gain, gt_gain)

    total = weights["awb"] * awb + weights["reconstruction"] * rec + weights["consistency"] * consistency

    result = {
        "awb": awb,
        "reconstruction": rec,
        "consistency": consistency,
        "total": total,
    }

    # 空间平滑损失（可选，与裁剪保持一致）
    if smoothness_weight > 0 and raw_image is not None:
        if crop_ratios is not None:
            raw_reshape = raw_image.reshape(B, S, *raw_image.shape[1:])
            raw_cropped = crop_to_overlap(raw_reshape, crop_ratios, loss_crop_size)
            _, S, t, _, _ = raw_cropped.shape
            smooth = spatial_smoothness_loss(gain_flat, raw_cropped.reshape(B * S, t, t, 3))
        else:
            smooth = spatial_smoothness_loss(pred_gain, raw_image)
        result["smoothness"] = smooth
        result["total"] = result["total"] + smoothness_weight * smooth

    # sRGB 损失（可选）
    if pred_srgb is not None and gt_srgb is not None:
        srgb = srgb_loss(pred_srgb, gt_srgb)
        srgb_weight = weights.get("srgb", 1.0)
        result["srgb"] = srgb
        result["total"] = result["total"] + srgb_weight * srgb

    return result
