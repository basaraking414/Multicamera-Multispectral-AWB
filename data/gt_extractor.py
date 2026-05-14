"""
独立的 AWB Ground Truth 提取模块
=================================
手动框选色卡灰阶行（6 个灰阶块），计算 AWB 增益。

使用方式:
  # 默认：弹出窗口手动框选灰阶行
  python gt_extractor.py --input_dir ../image_raw --output_dir ../image_processed

  # 指定灰阶行 ROI（跳过手动框选，适合批量处理）
  python gt_extractor.py --input_dir ../image_raw --output_dir ../image_processed --roi "100,200,400,500"
"""

import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# 灰阶行采样参数
NUM_GRAY_PATCHES = 6       # PMCC 和 X-Rite 灰阶行都是 6 列
MARGIN_H = 0.30            # 水平采样边距（每列中心 30% 区域）
MARGIN_V = 0.15            # 垂直采样边距（灰阶行窄，用小值避免溢出）


# ==============================================================================
# 工具函数
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
# 灰阶块提取
# ==============================================================================

def detect_gray_row_patches(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从灰阶行 ROI 中提取 6 个灰阶块。
    将 ROI 水平等分为 6 列，每列中心采样。

    参数:
        image: RGB 图像 [H, W, 3]
        roi: (x0, y0, x1, y1) 灰阶行区域

    返回:
        gray_patches: [6, 3]  每个灰阶块的 RGB 均值
        patch_centers: [6, 2] 中心坐标（全图坐标）
        patch_bboxes: [6, 4] 每个灰阶块的 bbox (x0,y0,x1,y1)
    """
    x0, y0, x1, y1 = roi
    roi_img = image[y0:y1, x0:x1]
    rh, rw = roi_img.shape[:2]

    if rw < 60:
        print(f"  [WARN] ROI 宽度 {rw}px 偏窄，采样可能不准确")

    cell_w = rw / NUM_GRAY_PATCHES

    gray_patches = []
    gray_centers = []
    gray_bboxes = []

    for col in range(NUM_GRAY_PATCHES):
        cx = int(col * cell_w + cell_w / 2)
        cy = rh // 2

        px1 = max(0, int(cx - cell_w * MARGIN_H / 2))
        px2 = min(rw, int(cx + cell_w * MARGIN_H / 2))
        py1 = max(0, int(cy - rh * MARGIN_V / 2))
        py2 = min(rh, int(cy + rh * MARGIN_V / 2))

        patch = roi_img[py1:py2, px1:px2]
        gray_patches.append(patch.reshape(-1, 3).mean(axis=0))
        gray_centers.append([x0 + cx, y0 + cy])
        gray_bboxes.append([x0 + px1, y0 + py1, x0 + px2, y0 + py2])

    patches_arr = np.array(gray_patches, dtype=np.float32)

    # 验证：灰阶行亮度应单调递减（白→黑）
    lum = patches_arr.mean(axis=1)
    if int(np.sum(np.diff(lum) < 0)) < 4:
        print(f"  [WARN] 灰阶行亮度非单调递减，框选可能不准确")

    return (
        patches_arr,
        np.array(gray_centers, dtype=np.float32),
        np.array(gray_bboxes, dtype=np.int32),
    )


def compute_gain_from_gray_patches(gray_patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从灰阶块 RGB 值计算 AWB 增益。
    排除最亮（可能过曝）和最暗（低信噪比）的块。

    返回:
        gain: (3,) AWB 增益 [R_gain, 1.0, B_gain]
        used_indices: (M,) 实际使用的灰阶块索引
    """
    gray_patches = np.clip(gray_patches, 1e-4, None)
    used_indices = [1, 2, 3, 4] if len(gray_patches) >= 5 else list(range(len(gray_patches)))
    used = gray_patches[used_indices]
    mean_rgb = used.mean(axis=0)
    gain = np.array(
        [mean_rgb[1] / mean_rgb[0], 1.0, mean_rgb[1] / mean_rgb[2]],
        dtype=np.float32,
    )
    return np.clip(gain, 0.25, 4.0), np.array(used_indices, dtype=np.int32)


# ==============================================================================
# 手动框选
# ==============================================================================

def _manual_select_roi(
    image: np.ndarray,
    window_title: str = "Select Gray Row",
) -> Optional[Tuple[int, int, int, int]]:
    """
    弹出窗口让用户手动框选灰阶行，支持鼠标滚轮放大。
    操作：滚轮=缩放, 右键拖拽=平移, 左键拖拽=框选, 中键=重置缩放, SPACE=确认, ESC=取消
    """
    preview = _auto_expose(image)
    display_img = (preview[..., ::-1] * 255).astype(np.uint8)  # RGB -> BGR

    h, w = display_img.shape[:2]
    WIN_W, WIN_H = 1200, 800
    init_scale = min(WIN_W / w, WIN_H / h, 1.0)

    # 状态
    state = {
        'scale': init_scale, 'pan_x': 0.0, 'pan_y': 0.0,
        'roi_start': None, 'roi_end': None, 'drawing': False, 'pan_ref': None,
    }

    def to_orig(dx, dy):
        return state['pan_x'] + dx / state['scale'], state['pan_y'] + dy / state['scale']

    def to_display(ox, oy):
        return (ox - state['pan_x']) * state['scale'], (oy - state['pan_y']) * state['scale']

    def mouse_cb(event, x, y, flags, _):
        s = state
        if event == cv2.EVENT_MOUSEWHEEL:
            orig = to_orig(x, y)
            factor = 1.2 if flags > 0 else 1 / 1.2
            new_scale = max(init_scale, min(s['scale'] * factor, init_scale * 10))
            s['pan_x'] = orig[0] - x / new_scale
            s['pan_y'] = orig[1] - y / new_scale
            s['scale'] = new_scale
        elif event == cv2.EVENT_RBUTTONDOWN:
            s['pan_ref'] = (x, y, s['pan_x'], s['pan_y'])
        elif event == cv2.EVENT_MOUSEMOVE and s['pan_ref'] and (flags & cv2.EVENT_FLAG_RBUTTON):
            sx, sy, ox, oy = s['pan_ref']
            s['pan_x'] = ox - (x - sx) / s['scale']
            s['pan_y'] = oy - (y - sy) / s['scale']
        elif event == cv2.EVENT_RBUTTONUP:
            s['pan_ref'] = None
        elif event == cv2.EVENT_MBUTTONDOWN:
            s['scale'], s['pan_x'], s['pan_y'] = init_scale, 0.0, 0.0
        elif event == cv2.EVENT_LBUTTONDOWN:
            s['roi_start'] = to_orig(x, y)
            s['roi_end'] = s['roi_start']
            s['drawing'] = True
        elif event == cv2.EVENT_MOUSEMOVE and s['drawing'] and (flags & cv2.EVENT_FLAG_LBUTTON):
            s['roi_end'] = to_orig(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            s['roi_end'] = to_orig(x, y)
            s['drawing'] = False

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, int(w * init_scale), int(h * init_scale))
    cv2.setMouseCallback(window_title, mouse_cb)

    while True:
        s = state
        vw, vh = WIN_W / s['scale'], WIN_H / s['scale']
        s['pan_x'] = max(0.0, min(s['pan_x'], w - vw))
        s['pan_y'] = max(0.0, min(s['pan_y'], h - vh))

        x1, y1 = int(s['pan_x']), int(s['pan_y'])
        x2, y2 = min(w, x1 + int(vw) + 1), min(h, y1 + int(vh) + 1)
        disp = cv2.resize(display_img[y1:y2, x1:x2], (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

        # 绘制 ROI
        if s['roi_start'] and s['roi_end']:
            rx1, ry1 = to_display(min(s['roi_start'][0], s['roi_end'][0]),
                                   min(s['roi_start'][1], s['roi_end'][1]))
            rx2, ry2 = to_display(max(s['roi_start'][0], s['roi_end'][0]),
                                   max(s['roi_start'][1], s['roi_end'][1]))
            cv2.rectangle(disp, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 2)

        zoom_pct = int(s['scale'] / init_scale * 100)
        cv2.putText(disp, f"Zoom: {zoom_pct}%", (WIN_W - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(disp, "Scroll=Zoom  Right-drag=Pan  Mid=Reset  Left-drag=ROI  SPACE=OK  ESC=Cancel",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_title, disp)

        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_title)
            return None
        if key in (13, 32) and s['roi_start'] and s['roi_end']:
            rx1 = int(min(s['roi_start'][0], s['roi_end'][0]))
            ry1 = int(min(s['roi_start'][1], s['roi_end'][1]))
            rx2 = int(max(s['roi_start'][0], s['roi_end'][0]))
            ry2 = int(max(s['roi_start'][1], s['roi_end'][1]))
            cv2.destroyWindow(window_title)
            return (rx1, ry1, rx2, ry2) if rx2 - rx1 >= 10 and ry2 - ry1 >= 10 else None


# ==============================================================================
# 统一接口
# ==============================================================================

def extract_awb_gt(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    interactive: bool = True,
) -> Dict[str, np.ndarray]:
    """
    提取 AWB Ground Truth。

    参数:
        image: RGB 图像 [H, W, 3], float32, range [0, 1]
        roi: 灰阶行 ROI (x0, y0, x1, y1)，指定后跳过手动框选
        interactive: True=手动框选(默认), False=必须提供 roi

    返回:
        {
            "awb_gt_gain":     (3,) AWB 增益 [R_gain, 1.0, B_gain]
            "gray_patch_rgb":  (3,) 灰阶块 RGB 均值
            "gray_patch_box":  (4,) 灰阶行 bbox
            "gt_method":       str
            "gray_patch_boxes": (6,4) 各灰阶块 bbox
            "used_indices":    (M,) 用于 gain 计算的索引
        }
    """
    image = _ensure_float01(image)

    if roi is not None:
        gray_patches, _, patch_bboxes = detect_gray_row_patches(image, roi)
        bbox = np.array(roi, dtype=np.int32)
        gt_method = "gray_row_roi"
    elif interactive:
        print("  [INFO] 请框选灰阶行 (6 个灰阶块, SPACE/ENTER 确认, ESC 取消)")
        roi_manual = _manual_select_roi(image)
        if roi_manual is None:
            raise ValueError("手动框选取消")
        gray_patches, _, patch_bboxes = detect_gray_row_patches(image, roi_manual)
        bbox = np.array(roi_manual, dtype=np.int32)
        gt_method = "gray_row_manual"
    else:
        raise ValueError("非交互模式必须提供 --roi 参数")

    gain, used_indices = compute_gain_from_gray_patches(gray_patches)
    return {
        "awb_gt_gain": gain,
        "gray_patch_rgb": gray_patches.mean(axis=0),
        "gray_patch_box": bbox,
        "gt_method": np.array(gt_method),
        "gray_patch_boxes": patch_bboxes,
        "used_indices": used_indices,
    }


# ==============================================================================
# 调试可视化
# ==============================================================================

def render_gt_debug(
    image: np.ndarray,
    bbox: np.ndarray,
    gain: np.ndarray,
    save_path: str,
    method: str = "unknown",
    gray_patch_boxes: Optional[np.ndarray] = None,
    used_indices: Optional[np.ndarray] = None,
) -> Optional[str]:
    """将灰阶块位置和 GT 增益渲染到图像上，保存调试图。"""
    if image is None:
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = _ensure_float01(image)

    # 应用 GT 增益得到白平衡预览
    gain_reshaped = gain.reshape(1, 1, 3).astype(np.float32)
    balanced = np.clip(image * gain_reshaped, 0.0, 1.0)
    preview = _auto_expose(balanced)
    preview = (preview[..., ::-1] * 255.0).astype(np.uint8)

    # 绘制灰阶行 ROI
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # 绘制各灰阶块
    if gray_patch_boxes is not None and len(gray_patch_boxes) > 0:
        used_set = set(used_indices.tolist()) if used_indices is not None else set()
        for i, (px0, py0, px1, py1) in enumerate(gray_patch_boxes):
            px0, py0, px1, py1 = int(px0), int(py0), int(px1), int(py1)
            if i in used_set:
                overlay = preview.copy()
                cv2.rectangle(overlay, (px0, py0), (px1, py1), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.35, preview, 0.65, 0, preview)
                cv2.rectangle(preview, (px0, py0), (px1, py1), (0, 200, 0), 2)
            else:
                cv2.rectangle(preview, (px0, py0), (px1, py1), (255, 255, 0), 1)

    text = "[{}] gain={:.2f}/{:.2f}/{:.2f}".format(method, gain[0], gain[1], gain[2])
    cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, preview)
    return save_path


# ==============================================================================
# 批处理
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
    roi: Optional[Tuple[int, int, int, int]] = None,
    interactive: bool = True,
) -> None:
    """读取解码后的 raw NPZ，添加 GT 后保存。"""
    raw_data = np.load(npz_path, allow_pickle=True)

    if "image_full" in raw_data:
        image_for_gt = raw_data["image_full"]
    else:
        image_for_gt = raw_data["image"]

    gt_result = extract_awb_gt(image_for_gt, roi=roi, interactive=interactive)

    sample = {
        "image": raw_data["image"],
        "image_full": raw_data["image_full"],
        "focal_length": raw_data["focal_length"],
        "focal_length_35mm": raw_data["focal_length_35mm"],
        "xyz2camera_rgb1": raw_data["xyz2camera_rgb1"],
        "xyz2camera_rgb2": raw_data["xyz2camera_rgb2"],
        "file_name": raw_data["file_name"],
        "raw_resolution": raw_data["raw_resolution"],
        "processed_resolution": raw_data["processed_resolution"],
        "crop_strategy": np.array("intrinsics_fallback_center_crop"),
        "awb_gt_gain": gt_result["awb_gt_gain"],
        "gray_patch_rgb": gt_result["gray_patch_rgb"],
        "gray_patch_box": gt_result["gray_patch_box"],
        "gt_method": gt_result["gt_method"],
    }

    safe_name = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(output_dir, f"{safe_name}.npz")
    np.savez(out_path, **sample)

    debug_path = os.path.join(debug_dir, f"{safe_name}_gt_debug.png")
    render_gt_debug(
        image=image_for_gt,
        bbox=gt_result["gray_patch_box"],
        gain=gt_result["awb_gt_gain"],
        save_path=debug_path,
        method=str(gt_result["gt_method"]),
        gray_patch_boxes=gt_result.get("gray_patch_boxes"),
        used_indices=gt_result.get("used_indices"),
    )

    print(f"  OK: {safe_name} | gain={gt_result['awb_gt_gain'].tolist()} | method={gt_result['gt_method']}")


def batch_process(
    input_dir: str,
    output_dir: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    interactive: bool = True,
) -> None:
    """批量处理目录下的所有 raw NPZ 文件。"""
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_gt")
    os.makedirs(debug_dir, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])
    if not npz_files:
        print(f"Warning: No NPZ files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} NPZ files, extracting GT (interactive={interactive})...")

    for npz_file in npz_files:
        npz_path = os.path.join(input_dir, npz_file)
        try:
            process_raw_npz(npz_path, output_dir, debug_dir, roi=roi, interactive=interactive)
        except Exception as exc:
            print(f"  FAIL: {npz_file} -> {exc}")

    print(f"GT extraction complete. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AWB Ground Truth 提取 — 灰阶行框选模式")
    parser.add_argument("--input_dir", default="image_raw", help="decoder 输出的 raw NPZ 目录")
    parser.add_argument("--output_dir", default="image_processed", help="输出 NPZ 目录")
    parser.add_argument("--roi", type=str, default=None,
                        help="灰阶行 ROI (x0,y0,x1,y1)，指定后跳过手动框选")
    parser.add_argument("--no_interactive", dest="interactive", action="store_false",
                        help="非交互模式（必须配合 --roi 使用）")
    args = parser.parse_args()

    roi = _parse_roi(args.roi) if args.roi else None

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        roi=roi,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
