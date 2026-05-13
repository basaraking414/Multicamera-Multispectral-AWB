"""可视化工具模块 —— 训练和测试共用的 debug 可视化函数。

提供两个核心函数：
- save_debug_scene: 可视化模型预测结果（input | pred | gt）
- save_mcs_alignment_debug: 可视化 MCS 与 RAW 的空间对齐情况
"""

import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from geometry_utils import auto_expose, ensure_float01

SENSOR_NAMES = ["tele", "main", "wide"]


def save_debug_scene(
    save_dir: str,
    identifier: int,
    input_image: torch.Tensor,
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    prefix: str = "scene",
) -> None:
    """可视化模型预测结果：每个 sensor 一行，每行 input | pred | gt。

    Args:
        save_dir:    保存目录
        identifier:  场景ID或epoch编号
        input_image: [S, H, W, 3] RAW 输入
        pred_image:  [S, H, W, 3] 模型预测
        gt_image:    [S, H, W, 3] Ground Truth
        prefix:      文件名前缀（"scene" 或 "epoch"）
    """
    os.makedirs(save_dir, exist_ok=True)

    def to_uint8(tensor: torch.Tensor, display_scale: float) -> np.ndarray:
        array = tensor.contiguous().detach().cpu().numpy()
        array = ensure_float01(array)
        array = np.clip(array / max(display_scale, 1e-4), 0.0, 1.0)
        array = auto_expose(array, percentile=100.0)
        return (array[..., ::-1] * 255.0).astype(np.uint8)

    scene_panels = []
    scene_stack = torch.cat([input_image, pred_image, gt_image], dim=0).contiguous().detach().cpu().numpy()
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
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_{identifier:04d}.png"), debug_image)


def save_mcs_alignment_debug(
    save_dir: str,
    identifier: int,
    batch: Dict[str, torch.Tensor],
    prefix: str = "scene",
) -> None:
    """可视化 MCS 与 RAW 的空间对齐情况，用于调试数据流是否正确。

    每张图包含 3 行（tele/main/wide），每行 3 列：
      - input:      RAW 图像（自动曝光）
      - mcs_rgb:    MCS 前 3 通道伪彩色
      - conf_overlay: 置信度热力图叠加到 input 上（红色=外推区域，蓝色=真实数据）

    Args:
        save_dir:    保存目录
        identifier:  场景ID或epoch编号
        batch:       包含 image, mcs, focal_length 的 dict
        prefix:      文件名前缀
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
        raw_img = ensure_float01(images[idx])
        raw_img = auto_expose(raw_img, percentile=100.0)
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
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_{identifier:04d}_mcs_alignment.png"), debug_image)
