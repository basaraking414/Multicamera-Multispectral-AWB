"""
独立的 AWB Ground Truth 提取模块
=================================
与 DNG 解码完全解耦，可独立运行在任意 RAW NPZ 上。

提供两种 GT 获取方法:
  1. colorchecker — 自动/半自动检测 X-Rite ColorChecker 灰阶块，获得准确 AWB 增益
  2. white_patch  — 启发式白块检测，适用于无色彩场景

使用方式:
  # 批量处理（自动检测色卡）
  python gt_extractor.py --input_dir ../image_raw --output_dir ../image_processed --method colorchecker

  # 带 ROI 的手动色卡模式（ROI格式: x0,y0,x1,y1）
  python gt_extractor.py --input_dir ../image_raw --output_dir ../image_processed --method colorchecker --roi "100,200,400,500"

  # 启发式白块模式
  python gt_extractor.py --input_dir ../image_raw --output_dir ../image_processed --method white_patch
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ==============================================================================
# 色彩工具
# ==============================================================================

def _ensure_float01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / image.max()
    return np.clip(image, 0.0, 1.0)


def _auto_expose(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    image = _ensure_float01(image)
    scale = np.percentile(image, percentile)
    scale = max(float(scale), 1e-4)
    image = np.clip(image / scale, 0.0, 1.0)
    image = np.power(image, 1.0 / 2.2)
    return np.clip(image, 0.0, 1.0)


# ==============================================================================
# 方法 A：X-Rite ColorChecker 灰阶块检测（推荐）
# ==============================================================================
# ColorChecker Classic 布局：4行×6列
# 第4行（最下面一行）为6个灰阶块：白→灰→黑（亮度递减）
# row_indices: 0=行1, 1=行2, 2=行3, 3=行4(灰阶行)
#
# 物理尺寸 ≈ 216mm × 140mm → 宽高比 ≈ 1.543

COLORCHECKER_GRID = (4, 6)
COLORCHECKER_ASPECT = 216.0 / 140.0  # ≈ 1.543
COLORCHECKER_ASPECT_TOLERANCE = 0.3
GRAY_ROW_INDEX = 3  # 0-indexed，第4行


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """将四个角点排序为：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上（和最小）
    rect[2] = pts[np.argmax(s)]   # 右下（和最大）
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # 右上（差最小）
    rect[3] = pts[np.argmax(d)]   # 左下（差最大）
    return rect


def detect_colorchecker_auto(image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    自动检测图中的 X-Rite ColorChecker。

    返回:
        (gray_patch_rgbs, patch_centers)
        gray_patch_rgbs: [6, 3] 灰阶块 RGB
        patch_centers:   [6, 2] 灰阶块中心坐标
    如果未检测到，返回 None。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image.shape[:2]

    # 多尺度边缘检测
    edges = cv2.Canny((gray * 255).astype(np.uint8), 30, 100)

    # 寻找四边形轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < h * w * 0.01:  # 至少占图像 1%
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        rect = _order_corners(pts)

        # 计算宽高比
        dx = rect[1] - rect[0]
        dy = rect[3] - rect[0]
        aspect = np.linalg.norm(dx) / max(np.linalg.norm(dy), 1e-4)

        if not (COLORCHECKER_ASPECT * (1 - COLORCHECKER_ASPECT_TOLERANCE) <= aspect <=
                COLORCHECKER_ASPECT * (1 + COLORCHECKER_ASPECT_TOLERANCE)):
            continue

        candidates.append((area, rect, aspect))

    if not candidates:
        return None

    # 取面积最大的候选
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_rect, _ = candidates[0]

    # 透视变换，获得正面视图
    dst_w, dst_h = 600, int(600 / COLORCHECKER_ASPECT)
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(best_rect, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))

    # 4×6 网格分割
    cell_w = dst_w / COLORCHECKER_GRID[1]
    cell_h = dst_h / COLORCHECKER_GRID[0]

    gray_patches = []
    gray_centers_img = []

    for col in range(COLORCHECKER_GRID[1]):
        cx = int(col * cell_w + cell_w / 2)
        cy = int(GRAY_ROW_INDEX * cell_h + cell_h / 2)

        # 在 warped 图中的网格中心采样一个小块
        margin = 0.3
        x1 = max(0, int(cx - cell_w * margin / 2))
        x2 = min(dst_w, int(cx + cell_w * margin / 2))
        y1 = max(0, int(cy - cell_h * margin / 2))
        y2 = min(dst_h, int(cy + cell_h * margin / 2))
        patch = warped[y1:y2, x1:x2]
        patch_rgb = patch.reshape(-1, 3).mean(axis=0)

        gray_patches.append(patch_rgb)

        # 将中心坐标映射回原图
        src_pt = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), cv2.getPerspectiveTransform(dst_pts, best_rect)
        )
        gray_centers_img.append(src_pt[0, 0])

    gray_patches = np.array(gray_patches, dtype=np.float32)
    gray_centers_img = np.array(gray_centers_img, dtype=np.float32)

    return gray_patches, gray_centers_img


def detect_colorchecker_roi(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定 ROI（x0, y0, x1, y1），提取色卡的灰阶块。
    将 ROI 区域划分为 4×6 网格，取最后一行作为灰阶块。

    返回:
        gray_patch_rgbs: [6, 3]
        patch_centers:   [6, 2]
    """
    x0, y0, x1, y1 = roi
    roi_img = image[y0:y1, x0:x1]
    rh, rw = roi_img.shape[:2]

    cell_w = rw / COLORCHECKER_GRID[1]
    cell_h = rh / COLORCHECKER_GRID[0]
    margin = 0.3

    gray_patches = []
    gray_centers = []

    for col in range(COLORCHECKER_GRID[1]):
        cx = int(col * cell_w + cell_w / 2)
        cy = int(GRAY_ROW_INDEX * cell_h + cell_h / 2)

        px1 = max(0, int(cx - cell_w * margin / 2))
        px2 = min(rw, int(cx + cell_w * margin / 2))
        py1 = max(0, int(cy - cell_h * margin / 2))
        py2 = min(rh, int(cy + cell_h * margin / 2))

        patch = roi_img[py1:py2, px1:px2]
        patch_rgb = patch.reshape(-1, 3).mean(axis=0)

        gray_patches.append(patch_rgb)
        gray_centers.append([x0 + cx, y0 + cy])

    return np.array(gray_patches, dtype=np.float32), np.array(gray_centers, dtype=np.float32)


def compute_gain_from_gray_patches(gray_patches: np.ndarray) -> np.ndarray:
    """
    从灰阶块 RGB 值计算 AWB 增益。
    输入: gray_patches [6, 3] — 6个灰阶块的RGB均值（从白到黑）

    策略：使用最亮且未饱和的灰阶块（通常是第1-3个）来计算。
    因为最暗的灰块信噪比低，最亮的白块可能过曝。
    """
    gray_patches = np.clip(gray_patches, 1e-4, None)
    luminance = gray_patches.mean(axis=1)

    # 排除最亮（可能过曝）和最暗（低信噪比）的块
    used_indices = [1, 2, 3, 4] if len(gray_patches) >= 5 else range(len(gray_patches))
    used = gray_patches[used_indices]

    # 平均多个灰块提高鲁棒性
    mean_rgb = used.mean(axis=0)
    gain = np.array(
        [mean_rgb[1] / mean_rgb[0], 1.0, mean_rgb[1] / mean_rgb[2]],
        dtype=np.float32,
    )
    return np.clip(gain, 0.25, 4.0)


# ==============================================================================
# 方法 B：启发式白块检测（从 gt_utils 移植，独立版本）
# ==============================================================================

def _window_mean(map_2d: np.ndarray, window_hw: Tuple[int, int]) -> np.ndarray:
    kernel = np.ones(window_hw, dtype=np.float32)
    summed = cv2.filter2D(map_2d, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return summed / float(window_hw[0] * window_hw[1])


def detect_white_patch(
    image: np.ndarray,
    patch_fraction: float = 0.1,
) -> Tuple[Tuple[int, int, int, int], np.ndarray, Dict[str, float]]:
    """
    检测图像中最亮的低饱和度区域作为白块近似。
    适用于包含白色/中性色块的场景。

    返回:
        bbox: (x0, y0, x1, y1)
        white_patch_rgb: (3,) — 白块区域 RGB 均值
        diagnostics: 调试信息
    """
    image = _ensure_float01(image)
    h, w, _ = image.shape

    patch_h = max(12, int(h * patch_fraction))
    patch_w = max(12, int(w * patch_fraction))

    max_rgb = image.max(axis=2)
    min_rgb = image.min(axis=2)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    saturation = (max_rgb - min_rgb) / np.clip(max_rgb, 1e-4, None)

    bright_score = luminance / np.clip(np.percentile(luminance, 98), 1e-4, None)
    neutral_score = 1.0 - np.clip(saturation / max(np.percentile(saturation, 75), 1e-4), 0.0, 1.0)

    clipped_penalty = (max_rgb > 0.98).astype(np.float32) * 0.35
    score = bright_score * 0.7 + neutral_score * 0.3 - clipped_penalty
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=3.0)

    score_mean = _window_mean(score, (patch_h, patch_w))
    peak_y, peak_x = np.unravel_index(np.argmax(score_mean), score_mean.shape)

    y0 = int(np.clip(peak_y - patch_h // 2, 0, h - patch_h))
    x0 = int(np.clip(peak_x - patch_w // 2, 0, w - patch_w))
    y1 = y0 + patch_h
    x1 = x0 + patch_w

    patch = image[y0:y1, x0:x1]
    white_patch_rgb = patch.reshape(-1, 3).mean(axis=0)
    diagnostics = {
        "score_max": float(score_mean.max()),
        "score_min": float(score_mean.min()),
        "patch_fraction": float(patch_fraction),
    }
    return (x0, y0, x1, y1), white_patch_rgb.astype(np.float32), diagnostics


def compute_awb_gain(white_patch_rgb: np.ndarray) -> np.ndarray:
    """从白块 RGB 计算 AWB 增益（G通道归一化）"""
    white_patch_rgb = np.clip(white_patch_rgb.astype(np.float32), 1e-4, None)
    green = white_patch_rgb[1]
    gain = np.array([green / white_patch_rgb[0], 1.0, green / white_patch_rgb[2]], dtype=np.float32)
    return np.clip(gain, 0.25, 4.0)


# ==============================================================================
# 统一接口
# ==============================================================================

def extract_awb_gt(
    image: np.ndarray,
    method: str = "colorchecker",
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    统一 GT 提取接口。

    参数:
        image:    RGB图像 [H, W, 3], float32, range [0, 1]
        method:   "colorchecker" 或 "white_patch"
        roi:      可选 ROI (x0, y0, x1, y1)，仅 colorchecker 模式使用

    返回:
        {
            "awb_gt_gain":    (3,) AWB 增益 [R_gain, 1.0, B_gain]
            "white_patch_rgb": (3,) 白块/灰阶块 RGB 均值
            "white_patch_box": (4,) bbox (x0, y0, x1, y1)
            "white_patch_score": (3,) [score_max, score_min, patch_fraction]
            "gt_method":      str, 实际使用的方法
        }
    """
    image = _ensure_float01(image)

    if method == "colorchecker":
        if roi is not None:
            gray_patches, _ = detect_colorchecker_roi(image, roi)
            success = True
            bbox = np.array([roi[0], roi[1], roi[2], roi[3]], dtype=np.int32)
            white_patch_rgb = gray_patches.mean(axis=0)
            gt_method = "colorchecker_roi"
        else:
            result = detect_colorchecker_auto(image)
            if result is not None:
                gray_patches, centers = result
                white_patch_rgb = gray_patches.mean(axis=0)
                # bbox 取灰阶块的包围盒
                x0 = int(centers[:, 0].min()) - 10
                y0 = int(centers[:, 1].min()) - 10
                x1 = int(centers[:, 0].max()) + 10
                y1 = int(centers[:, 1].max()) + 10
                bbox = np.array([x0, y0, x1, y1], dtype=np.int32)
                gt_method = "colorchecker_auto"
                success = True
            else:
                # 自动检测失败，回退到 white_patch
                print("  [WARN] ColorChecker auto-detection failed, falling back to white_patch")
                gt_method = "white_patch_fallback"
                success = False

        if success and method == "colorchecker" and gt_method.startswith("colorchecker"):
            gain = compute_gain_from_gray_patches(gray_patches)
            return {
                "awb_gt_gain": gain,
                "white_patch_rgb": white_patch_rgb,
                "white_patch_box": bbox,
                "white_patch_score": np.array([1.0, 0.0, 0.1], dtype=np.float32),
                "gt_method": np.array(gt_method),
            }

    # white_patch 方法（或 colorchecker 的回退）
    bbox, white_patch_rgb, diagnostics = detect_white_patch(image)
    gain = compute_awb_gain(white_patch_rgb)
    return {
        "awb_gt_gain": gain,
        "white_patch_rgb": white_patch_rgb,
        "white_patch_box": np.array(bbox, dtype=np.int32),
        "white_patch_score": np.array(
            [diagnostics["score_max"], diagnostics["score_min"], diagnostics["patch_fraction"]],
            dtype=np.float32,
        ),
        "gt_method": np.array("white_patch"),
    }


# ==============================================================================
# 调试可视化
# ==============================================================================

def render_gt_debug(
    image: np.ndarray,
    bbox: np.ndarray,
    white_patch_rgb: np.ndarray,
    gain: np.ndarray,
    save_path: str,
    method: str = "unknown",
) -> Optional[str]:
    """将检测到的白块位置和 GT 增益渲染到图像上，保存调试图。"""
    if image is None:
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = _ensure_float01(image)

    # 应用 GT 增益得到白平衡预览
    gain_reshaped = gain.reshape(1, 1, 3).astype(np.float32)
    balanced = np.clip(image * gain_reshaped, 0.0, 1.0)

    # 自动曝光 + gamma 校正
    preview = _auto_expose(balanced)
    preview = (preview[..., ::-1] * 255.0).astype(np.uint8)

    # 绘制 bbox
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)

    text = "[{}] rgb={:.3f}/{:.3f}/{:.3f} gain={:.2f}/{:.2f}/{:.2f}".format(
        method,
        white_patch_rgb[0], white_patch_rgb[1], white_patch_rgb[2],
        gain[0], gain[1], gain[2],
    )
    cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, preview)
    return save_path


# ==============================================================================
# 批处理：读取 raw NPZ → 添加 GT → 输出到 image_processed
# ==============================================================================

def _parse_roi(roi_str: str) -> Tuple[int, int, int, int]:
    parts = roi_str.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"ROI must have 4 values (x0 y0 x1 y1), got: {roi_str}")
    return tuple(int(p) for p in parts)


def process_raw_npz(
    npz_path: str,
    output_dir: str,
    debug_dir: str,
    method: str = "colorchecker",
    roi: Optional[Tuple[int, int, int, int]] = None,
    size: int = 256,
) -> None:
    """
    读取解码后的 raw NPZ，添加 GT 后保存为完整 NPZ。
    输出格式兼容 dataloader。
    """
    raw_data = np.load(npz_path, allow_pickle=True)

    # 读取全分辨率 RAW 图像用于 GT 检测（精度更高）
    if "image_full" in raw_data:
        image_for_gt = raw_data["image_full"]
    else:
        image_for_gt = raw_data["image"]

    # 提取 GT
    gt_result = extract_awb_gt(image_for_gt, method=method, roi=roi)

    # 构建兼容 dataloader 的输出
    sample = {
        "image": raw_data["image"],  # 已缩放的图像
        "image_full": raw_data["image_full"],
        "focal_length": raw_data["focal_length"],
        "focal_length_35mm": raw_data["focal_length_35mm"],
        "xyz2camera_rgb1": raw_data["xyz2camera_rgb1"],
        "xyz2camera_rgb2": raw_data["xyz2camera_rgb2"],
        "file_name": raw_data["file_name"],
        "raw_resolution": raw_data["raw_resolution"],
        "processed_resolution": raw_data["processed_resolution"],
        "crop_strategy": np.array("intrinsics_fallback_center_crop"),
        # GT 字段
        "awb_gt_gain": gt_result["awb_gt_gain"],
        "white_patch_rgb": gt_result["white_patch_rgb"],
        "white_patch_box": gt_result["white_patch_box"],
        "white_patch_score": gt_result["white_patch_score"],
        "gt_method": gt_result["gt_method"],
    }

    # 保存
    safe_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(output_dir, f"{safe_name}.npz")
    np.savez(out_path, **sample)

    # 调试可视化
    debug_path = os.path.join(debug_dir, f"{safe_name}_gt_debug.png")
    render_gt_debug(
        image=image_for_gt,
        bbox=gt_result["white_patch_box"],
        white_patch_rgb=gt_result["white_patch_rgb"],
        gain=gt_result["awb_gt_gain"],
        save_path=debug_path,
        method=str(gt_result.get("gt_method", method)),
    )

    print(f"  OK: {safe_name} | gt_gain={gt_result['awb_gt_gain'].tolist()} | method={gt_result['gt_method']}")


def batch_process(
    input_dir: str,
    output_dir: str,
    method: str = "colorchecker",
    roi: Optional[Tuple[int, int, int, int]] = None,
    size: int = 256,
) -> None:
    """批量处理目录下的所有 raw NPZ 文件。"""
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_gt")
    os.makedirs(debug_dir, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])

    if not npz_files:
        print(f"Warning: No NPZ files found in {input_dir}")
        print(f"  Run dng_decoder.py first to decode DNG files.")
        return

    print(f"Found {len(npz_files)} NPZ files, extracting GT (method={method})...")

    for npz_file in npz_files:
        npz_path = os.path.join(input_dir, npz_file)
        try:
            process_raw_npz(npz_path, output_dir, debug_dir, method=method, roi=roi, size=size)
        except Exception as exc:
            print(f"  FAIL: {npz_file} -> {exc}")

    print(f"GT extraction complete. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWB Ground Truth 提取（独立于 DNG 解码）")
    parser.add_argument("--input_dir", default="image_raw", help="decoder 输出的 raw NPZ 目录")
    parser.add_argument("--output_dir", default="image_processed", help="输出完整 NPZ 目录（兼容 dataloader）")
    parser.add_argument("--method", default="colorchecker", choices=["colorchecker", "white_patch"],
                        help="GT 提取方法: colorchecker（推荐）或 white_patch")
    parser.add_argument("--roi", type=str, default=None,
                        help="色卡 ROI (x0,y0,x1,y1)，指定后跳过自动检测直接使用该区域")
    parser.add_argument("--size", type=int, default=256, help="输出图像尺寸")
    args = parser.parse_args()

    roi = _parse_roi(args.roi) if args.roi else None

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        roi=roi,
        size=args.size,
    )


if __name__ == "__main__":
    main()
