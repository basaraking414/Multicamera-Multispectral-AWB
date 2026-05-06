import os
import struct
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# MCS 旋转校正参数
# =============================================================================
# 多光谱传感器与 RGB 摄像头的硬件安装方向可能不同。
# 通过 np.rot90 的 k 参数校正:
#   0 = 不旋转
#   1 = 逆时针 90°
#   2 = 180°
#   3 = 顺时针 90°（等价于逆时针 270°）
# 请根据实际视觉效果调整此值。
# 你可以运行 visualize_mcs.py 观察 MCS 的各通道图像，与对应 raw 图像对比，
# 找到正确的旋转参数后重新运行本脚本即可。
MCS_ROT90_K = 3

def robust_channel_normalize(mcs: np.ndarray, upper_percentile: float = 99.5) -> np.ndarray:
    """
    Normalize each spectral channel independently so that channels with
    different response ranges contribute on a similar scale.
    """
    normalized = np.zeros_like(mcs, dtype=np.float32)
    for channel_idx in range(mcs.shape[-1]):
        channel = np.clip(mcs[..., channel_idx].astype(np.float32), 0.0, None)
        scale = np.percentile(channel, upper_percentile)
        scale = max(float(scale), 1e-6)
        normalized[..., channel_idx] = np.clip(channel / scale, 0.0, 1.0)
    return normalized


if __name__ == "__main__":
    pathBin = r"McsBin"
    listBins = [x for x in os.listdir(pathBin) if x.endswith('.bin')]

    mcs_dir = "Mcsnpy"
    if not os.path.exists(mcs_dir):
        os.makedirs(mcs_dir)

    for name in listBins:
        bin_file = os.path.join(pathBin, name)
        with open(bin_file, "rb") as f:
            f.seek(32)
            black_level = struct.unpack("f", f.read(4))[0]
            height = struct.unpack("I", f.read(4))[0]
            width = struct.unpack("I", f.read(4))[0]
            _ = struct.unpack("f", f.read(4))[0]

            rawdata = np.zeros((height, width), dtype=np.float32)
            for i_data in range(width * height):
                rawdata_temp = int.from_bytes(f.read(2), byteorder="little", signed=False)
                i_row = i_data // width
                i_col = i_data - i_row * width
                rawdata[i_row, i_col] = rawdata_temp

        # 提取多通道光谱数据
        rows = (height - 3 * 2) // 3
        cols = (width - 4 * 2) // 3

        output_channels = np.zeros((rows, cols, 9), dtype=np.float32)
        for i_block_row in range(rows):
            for i_block_col in range(cols):
                start_row = 3 + i_block_row * 3
                start_col = 4 + i_block_col * 3

                raw_block = rawdata[start_row:start_row + 3, start_col:start_col + 3]
                output_channels[i_block_row, i_block_col, :] = raw_block.reshape(-1)

        output_channels = np.clip(output_channels - black_level, 0.0, None)
        output_channels = robust_channel_normalize(output_channels)

        # MCS 旋转校正：使 MCS 坐标系与 raw 图像坐标系一致
        if MCS_ROT90_K != 0:
            output_channels = np.rot90(output_channels, k=MCS_ROT90_K)
            print(f"  Rotated MCS by k={MCS_ROT90_K} (shape: {output_channels.shape})")

        base_name = os.path.splitext(name)[0]
        output_file = os.path.join(mcs_dir, f"{base_name}.npy")
        np.save(output_file, output_channels)


