import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _ensure_float01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / image.max()
    return np.clip(image, 0.0, 1.0)


def _apply_awb(image: np.ndarray, white_patch_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image = _ensure_float01(image)
    gain = compute_awb_gain(white_patch_rgb).reshape(1, 1, 3)
    balanced = np.clip(image * gain, 0.0, 1.0)
    return balanced, gain.reshape(3)


def _auto_expose(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    image = _ensure_float01(image)
    scale = np.percentile(image, percentile)
    scale = max(float(scale), 1e-4)
    image = np.clip(image / scale, 0.0, 1.0)
    image = np.power(image, 1.0 / 2.2)
    return np.clip(image, 0.0, 1.0)


def _window_mean(map_2d: np.ndarray, window_hw: Tuple[int, int]) -> np.ndarray:
    kernel = np.ones(window_hw, dtype=np.float32)
    summed = cv2.filter2D(map_2d, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return summed / float(window_hw[0] * window_hw[1])


def detect_white_patch(
    image: np.ndarray,
    patch_fraction: float = 0.1,
) -> Tuple[Tuple[int, int, int, int], np.ndarray, Dict[str, float]]:
    """
    Detect the brightest low-saturation region as a proxy for the white patch.
    The color checker is assumed to be visible in every frame.
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
    white_patch_rgb = np.clip(white_patch_rgb.astype(np.float32), 1e-4, None)
    green = white_patch_rgb[1]
    gain = np.array([green / white_patch_rgb[0], 1.0, green / white_patch_rgb[2]], dtype=np.float32)
    return np.clip(gain, 0.25, 4.0)


def extract_awb_gt(image: np.ndarray) -> Dict[str, np.ndarray]:
    bbox, white_patch_rgb, diagnostics = detect_white_patch(image)
    awb_gt_gain = compute_awb_gain(white_patch_rgb)
    return {
        "awb_gt_gain": awb_gt_gain,
        "white_patch_rgb": white_patch_rgb,
        "white_patch_box": np.array(bbox, dtype=np.int32),
        "white_patch_score": np.array(
            [diagnostics["score_max"], diagnostics["score_min"], diagnostics["patch_fraction"]],
            dtype=np.float32,
        ),
    }


def render_white_patch_debug(
    image: np.ndarray,
    bbox: np.ndarray,
    white_patch_rgb: np.ndarray,
    save_path: str,
) -> Optional[str]:
    if image is None:
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = _ensure_float01(image)
    balanced, gain = _apply_awb(image, white_patch_rgb)
    preview = _auto_expose(balanced)
    preview = (preview[..., ::-1] * 255.0).astype(np.uint8)

    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
    text = "white={:.3f}/{:.3f}/{:.3f} gain={:.2f}/{:.2f}/{:.2f}".format(
        white_patch_rgb[0],
        white_patch_rgb[1],
        white_patch_rgb[2],
        gain[0],
        gain[1],
        gain[2],
    )
    cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, preview)
    return save_path
