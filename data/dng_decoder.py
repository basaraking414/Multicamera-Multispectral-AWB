"""
解耦的 DNG 解码器
==================
只做 RAW 解码和元数据提取，不包含任何白平衡/白块 GT 计算。
输出干净的 NPZ 文件，供 gt_extractor.py 后续独立添加 GT。

与 dng_process_rawpy.py 的区别：
  - 不调用 extract_awb_gt / render_white_patch_debug
  - 不依赖 gt_utils.py
  - 输出到独立的输出目录（image_raw/），不污染原有的 image_processed/
"""

import os
import sys
from typing import Any, Dict, Optional

import cv2
import exiftool
import numpy as np
import rawpy


DEFAULT_EXIFTOOL = "E:/ricky/research/oppo-project-2025/exiftool-13.53_64/exiftool-13.53_64/exiftool.exe"
RAW_EXTENSIONS = (".dng", ".nef", ".arw")


def _parse_float(value: Any, default: float = 0.0) -> np.float32:
    if value is None:
        return np.float32(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.float32(value)

    text = str(value).strip().replace(" mm", "")
    if "/" in text:
        num, den = text.split("/", 1)
        return np.float32(float(num) / max(float(den), 1e-6))
    return np.float32(float(text))


def _parse_matrix(value: Any) -> np.ndarray:
    if value is None:
        return np.eye(3, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(3, 3)

    matrix = np.array(str(value).split(), dtype=np.float32).reshape(3, 3)
    return matrix


def decode_dng(
    raw_file_path: str,
    size: int = 256,
) -> Dict[str, Any]:
    """
    解码单张 DNG 文件，返回 RAW 图像 + 元数据。
    不包含任何 GT 计算，纯解码。
    """
    raw_file = os.path.basename(raw_file_path)

    # 提取 EXIF 元数据
    exiftool_kwargs = {"executable": DEFAULT_EXIFTOOL} if os.path.exists(DEFAULT_EXIFTOOL) else {}
    with exiftool.ExifTool(**exiftool_kwargs) as et:
        metadata = et.execute_json(raw_file_path)[0]

    # RAW 解码
    with rawpy.imread(raw_file_path) as raw:
        img_linear = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
            output_bps=16,
            no_auto_bright=True,
            no_auto_scale=True,
            use_camera_wb=False,
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            user_black=raw.black_level_per_channel[0],
            user_sat=raw.white_level,
        )
        img_normalized = np.clip(img_linear.astype(np.float32) / float(raw.white_level), 0.0, 1.0)

    # 缩放到统一尺寸
    img_resized = cv2.resize(img_normalized, (size, size), interpolation=cv2.INTER_AREA)

    sample = {
        "image": img_resized.astype(np.float32),
        "image_full": img_normalized.astype(np.float32),
        "focal_length": _parse_float(metadata.get("EXIF:FocalLength")),
        "focal_length_35mm": _parse_float(metadata.get("EXIF:FocalLengthIn35mmFormat")),
        "xyz2camera_rgb1": _parse_matrix(metadata.get("EXIF:ColorMatrix1")),
        "xyz2camera_rgb2": _parse_matrix(metadata.get("EXIF:ColorMatrix2")),
        "file_name": np.array(str(metadata.get("File:FileName", ""))),
        "raw_resolution": np.array(
            [
                int(metadata.get("EXIF:ImageHeight", img_normalized.shape[0])),
                int(metadata.get("EXIF:ImageWidth", img_normalized.shape[1])),
            ],
            dtype=np.int32,
        ),
        "processed_resolution": np.array(img_resized.shape[:2], dtype=np.int32),
        "crop_strategy": np.array("none"),
    }
    return sample


def batch_decode(
    input_dir: str,
    output_dir: str,
    size: int = 256,
) -> None:
    """
    批量解码目录下的所有 DNG 文件。
    输出 NPZ 到 output_dir，不含 GT。
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_files = sorted([
        f for f in os.listdir(input_dir) if f.lower().endswith(RAW_EXTENSIONS)
    ])

    if not raw_files:
        print(f"Warning: No DNG files found in {input_dir}")
        return

    print(f"Found {len(raw_files)} DNG files, decoding...")

    for raw_file in raw_files:
        raw_path = os.path.join(input_dir, raw_file)
        try:
            sample = decode_dng(raw_path, size=size)
            safe_name = os.path.splitext(raw_file)[0]
            out_path = os.path.join(output_dir, f"{safe_name}.npz")
            np.savez(out_path, **sample)
            print(f"  OK: {raw_file} -> {safe_name}.npz")
        except Exception as exc:
            print(f"  FAIL: {raw_file} -> {exc}")

    print(f"Decoding complete. Output: {output_dir}")


def main() -> None:
    input_dir = "image_dng"
    output_dir = "image_raw"
    batch_decode(input_dir, output_dir, size=256)


if __name__ == "__main__":
    main()
