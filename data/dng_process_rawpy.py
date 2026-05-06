import os
import sys
from typing import Any, Dict, Optional

import cv2
import exiftool
import numpy as np
import rawpy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import load_config
from gt_utils import extract_awb_gt, render_white_patch_debug


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


def _build_sample(
    image: np.ndarray,
    metadata: Dict[str, Any],
    gt_payload: Dict[str, np.ndarray],
    crop_strategy: str,
) -> Dict[str, Any]:
    return {
        "image": image.astype(np.float32),
        "image_full": image.astype(np.float32),
        "focal_length": _parse_float(metadata.get("EXIF:FocalLength")),
        "focal_length_35mm": _parse_float(metadata.get("EXIF:FocalLengthIn35mmFormat")),
        "xyz2camera_rgb1": _parse_matrix(metadata.get("EXIF:ColorMatrix1")),
        "xyz2camera_rgb2": _parse_matrix(metadata.get("EXIF:ColorMatrix2")),
        "file_name": np.array(str(metadata.get("File:FileName", ""))),
        "raw_resolution": np.array(
            [
                int(metadata.get("EXIF:ImageHeight", image.shape[0])),
                int(metadata.get("EXIF:ImageWidth", image.shape[1])),
            ],
            dtype=np.int32,
        ),
        "processed_resolution": np.array(image.shape[:2], dtype=np.int32),
        "crop_strategy": np.array(crop_strategy),
        **gt_payload,
    }


def process_raw_file(
    raw_file_path: str,
    output_dir: str,
    debug_dir: str,
    exiftool_path: Optional[str] = None,
    size: int = 256,
) -> None:
    raw_file = os.path.basename(raw_file_path)
    print(f"Processing: {raw_file}")

    exiftool_kwargs = {"executable": exiftool_path} if exiftool_path and os.path.exists(exiftool_path) else {}
    with exiftool.ExifTool(**exiftool_kwargs) as et:
        metadata = et.execute_json(raw_file_path)[0]

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

    gt_payload = extract_awb_gt(img_normalized)
    img_resized = cv2.resize(img_normalized, (size, size), interpolation=cv2.INTER_AREA)

    sample = _build_sample(
        image=img_resized,
        metadata=metadata,
        gt_payload=gt_payload,
        crop_strategy="intrinsics_fallback_center_crop",
    )

    safe_name = os.path.splitext(raw_file)[0]
    np.savez(os.path.join(output_dir, f"{safe_name}.npz"), **sample)

    debug_path = os.path.join(debug_dir, f"{safe_name}_white_patch.png")
    render_white_patch_debug(
        image=img_normalized,
        bbox=gt_payload["white_patch_box"],
        white_patch_rgb=gt_payload["white_patch_rgb"],
        save_path=debug_path,
    )

    print(
        " Success: focal35mm={} raw_shape={} gt_gain={}".format(
            sample["focal_length_35mm"],
            tuple(sample["raw_resolution"].tolist()),
            sample["awb_gt_gain"].tolist(),
        )
    )


def main() -> None:
    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))

    dataset_path = os.path.join(cfg.data.root_dir, cfg.data.image_dng_dir)
    save_path = os.path.join(cfg.data.root_dir, cfg.data.image_processed_dir)
    debug_dir = os.path.join(cfg.data.root_dir, cfg.data.debug_white_patch_dir)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    raw_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(tuple(cfg.data.raw_extensions))]
    raw_files.sort()

    for raw_file in raw_files:
        raw_file_path = os.path.join(dataset_path, raw_file)
        try:
            process_raw_file(
                raw_file_path,
                save_path,
                debug_dir,
                exiftool_path=cfg.external_tools.exiftool_path,
            )
        except Exception as exc:
            print(f" Failed to process {raw_file}: {exc}")

    print("Processing completed.")


if __name__ == "__main__":
    main()

