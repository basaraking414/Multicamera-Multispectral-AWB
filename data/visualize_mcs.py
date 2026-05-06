import os
import sys
from typing import Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import load_config


def _normalize_for_display(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    low = float(np.percentile(channel, 1.0))
    high = float(np.percentile(channel, 99.5))
    if high <= low:
        high = low + 1e-6
    channel = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return (channel * 255.0).astype(np.uint8)


def _build_channel_panel(channel: np.ndarray, idx: int) -> np.ndarray:
    gray = _normalize_for_display(channel)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    mean = float(channel.mean())
    p99 = float(np.percentile(channel, 99.0))
    cv2.putText(color, f"ch{idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        color,
        f"mean={mean:.3f} p99={p99:.3f}",
        (8, color.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return color


def _make_montage(mcs: np.ndarray, tile_size: Tuple[int, int] = (220, 220)) -> np.ndarray:
    tiles = []
    for idx in range(mcs.shape[-1]):
        tile = _build_channel_panel(mcs[..., idx], idx)
        tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_NEAREST)
        tiles.append(tile)

    rows = []
    for row_idx in range(0, len(tiles), 3):
        row_tiles = tiles[row_idx : row_idx + 3]
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def _make_summary_map(mcs: np.ndarray, tile_size: Tuple[int, int] = (330, 220)) -> np.ndarray:
    summary = mcs.mean(axis=-1)
    summary_u8 = _normalize_for_display(summary)
    summary_color = cv2.applyColorMap(summary_u8, cv2.COLORMAP_TURBO)
    summary_color = cv2.resize(summary_color, tile_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(summary_color, "mean_over_channels", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return summary_color


def visualize_one_file(mcs_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    mcs = np.load(mcs_path).astype(np.float32)

    montage = _make_montage(mcs)
    summary = _make_summary_map(mcs, tile_size=(montage.shape[1], 220))
    debug_image = np.concatenate([summary, montage], axis=0)


    base_name = os.path.splitext(os.path.basename(mcs_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_mcs_debug.png")
    cv2.imwrite(output_path, debug_image)
    return output_path


def main() -> None:
    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))

    mcs_dir = os.path.join(cfg.data.root_dir, cfg.data.mcs_npy_dir)
    output_dir = os.path.join(cfg.data.root_dir, cfg.data.debug_mcs_dir)
    os.makedirs(output_dir, exist_ok=True)

    mcs_files = sorted([f for f in os.listdir(mcs_dir) if f.endswith(".npy")])
    for name in mcs_files:
        mcs_path = os.path.join(mcs_dir, name)
        output_path = visualize_one_file(mcs_path, output_dir)
        print(f"Saved MCS debug: {output_path}")


if __name__ == "__main__":
    main()
