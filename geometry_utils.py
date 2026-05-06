from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def focal_to_crop_ratio(focal_length: float, reference_focal: float) -> float:
    focal_length = max(float(focal_length), 1e-4)
    reference_focal = max(float(reference_focal), 1e-4)
    return float(np.clip(focal_length / reference_focal, 0.05, 1.0))


def center_crop(image: np.ndarray, crop_ratio: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    crop_h = max(4, min(h, int(round(h * crop_ratio))))
    crop_w = max(4, min(w, int(round(w * crop_ratio))))

    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def resize_image(image: np.ndarray, target_hw: Tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    target_h, target_w = target_hw
    if image.ndim == 3 and image.shape[2] > 4:
        channels = [
            cv2.resize(image[..., idx], (target_w, target_h), interpolation=interpolation)
            for idx in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def crop_and_resize(
    image: np.ndarray,
    crop_ratio: float,
    target_hw: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cropped, crop_box = center_crop(image, crop_ratio)
    resized = resize_image(cropped, target_hw, interpolation=interpolation)
    return resized, crop_box


def compute_scene_crop_plan(focal_lengths: Iterable[float]) -> List[Dict[str, float]]:
    focal_lengths = [float(v) for v in focal_lengths]
    reference_focal = max(focal_lengths)

    plan = []
    for focal_length in focal_lengths:
        crop_ratio = focal_to_crop_ratio(focal_length, reference_focal)
        plan.append(
            {
                "focal_length": focal_length,
                "reference_focal": reference_focal,
                "crop_ratio": crop_ratio,
            }
        )
    return plan


def compute_crop_boxes_from_ratios(
    img_h: int, img_w: int, crop_ratios: List[float], min_size: int = 4
) -> List[Tuple[int, int, int, int]]:
    """根据 crop_ratios 列表计算每张图的中心裁剪框坐标 (x0, y0, x1, y1)。

    此函数只计算坐标不执行裁剪，供 loss 阶段对模型输出做后裁剪使用。
    """
    boxes = []
    for ratio in crop_ratios:
        crop_h = max(min_size, min(img_h, int(round(img_h * ratio))))
        crop_w = max(min_size, min(img_w, int(round(img_w * ratio))))
        y0 = max(0, (img_h - crop_h) // 2)
        x0 = max(0, (img_w - crop_w) // 2)
        boxes.append((x0, y0, x0 + crop_w, y0 + crop_h))
    return boxes


def align_mcs_to_fov(
    mcs: np.ndarray,
    align_ratio: float,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """将 MCS 数据从主摄视场空间对齐到各模组视场，同时生成置信度图。

    MCS 传感器固定 FOV ≈ Main 模组。Tele 和 Wide 的视场不同，
    需要对 MCS 做空间变换，使 raw 和 MCS 在每个像素对应。

    对齐方式：
      - main (ratio≈1.0) : 不变，置信度全 1.0
      - tele (ratio<1.0)  : 裁剪中心放大，数据全部真实 → 置信度全 1.0
      - wide (ratio>1.0)  : 缩小后 reflect 外推填充，置信度从中心 1.0 平滑衰减到边缘 0

    Args:
        mcs:         [H, W, C] 原始 MCS 数据（主摄视场）
        align_ratio: focal_main / focal_camera
        target_hw:   (H, W) 输出尺寸

    Returns:
        aligned_mcs:  [target_h, target_w, C] 对齐后的 MCS 数据
        confidence:   [target_h, target_w]     置信度图 (0~1)
    """
    th, tw = target_hw
    mcs_resized = resize_image(mcs, target_hw, interpolation=cv2.INTER_AREA)

    if abs(align_ratio - 1.0) < 1e-6:
        # Main：MCS FOV ≈ Main FOV，完全置信
        return mcs_resized, np.ones((th, tw), dtype=np.float32)

    if align_ratio < 1.0:
        # Tele：裁剪中心再放大，数据全部来自真实 MCS → 置信度全 1
        crop_h = max(4, int(round(th * align_ratio)))
        crop_w = max(4, int(round(tw * align_ratio)))
        y0 = (th - crop_h) // 2
        x0 = (tw - crop_w) // 2
        cropped = mcs_resized[y0:y0 + crop_h, x0:x0 + crop_w]
        aligned = resize_image(cropped, target_hw, interpolation=cv2.INTER_AREA)
        return aligned, np.ones((th, tw), dtype=np.float32)

    # Wide：缩小 + reflect 外推填充 + 置信度高斯衰减
    small_h = max(4, min(th, int(round(th / align_ratio))))
    small_w = max(4, min(tw, int(round(tw / align_ratio))))
    small = resize_image(mcs_resized, (small_h, small_w), interpolation=cv2.INTER_AREA)

    pad_top = (th - small_h) // 2
    pad_bottom = th - small_h - pad_top
    pad_left = (tw - small_w) // 2
    pad_right = tw - small_w - pad_left

    # reflect 镜像外推，光谱连续性比 edge 复制更好
    aligned = np.pad(small,
                     ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                     mode='reflect')

    # 置信度图：中心真实区域 1.0 → 边缘外推区域 0.0，高斯平滑过渡
    confidence = np.zeros((th, tw), dtype=np.float32)
    confidence[pad_top:pad_top + small_h, pad_left:pad_left + small_w] = 1.0
    sigma = max(pad_top, pad_bottom, pad_left, pad_right) * 0.4
    if sigma > 1.0:
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=sigma)
    confidence = np.clip(confidence, 0.0, 1.0)

    return aligned, confidence
