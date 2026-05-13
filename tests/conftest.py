"""共享测试fixtures —— 提供常用的小模型实例和合成数据。"""

import sys
import os

import pytest
import torch

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AWBTransformer


@pytest.fixture
def device():
    """返回可用的设备（CUDA 或 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model(device):
    """创建一个小模型用于快速测试。"""
    model = AWBTransformer(
        dim=32,
        num_heads=4,
        grid_size=8,
        use_positional_encoding=True,
        focal_embed_dim=8,
        predict_ccm=True,
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def synthetic_batch(device, batch_size=2, scene_size=3, img_size=32, mcs_size=32):
    """创建合成的测试数据 batch。"""
    batch = {
        "image": torch.randn(batch_size, scene_size, img_size, img_size, 3, device=device),
        "mcs": torch.randn(batch_size, scene_size, mcs_size, mcs_size, 10, device=device),
        "ccm1": torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, scene_size, -1, -1),
        "ccm2": torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, scene_size, -1, -1),
        "focal_length": torch.tensor([[50.0, 26.0, 13.0]] * batch_size, device=device),
        "awb_gt_gain": torch.ones(batch_size, scene_size, 3, device=device),
        "gt_image": torch.randn(batch_size, scene_size, img_size, img_size, 3, device=device),
        "crop_ratio": torch.ones(batch_size, scene_size, device=device),
        "sensor_id": torch.tensor([[0, 1, 2]] * batch_size, device=device),
        "scene_id": torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, scene_size),
    }
    return batch


@pytest.fixture
def ccm_matrices(device, batch_size=4):
    """创建测试用的CCM矩阵。"""
    # 创建可逆的CCM矩阵
    ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1) + \
           torch.randn(batch_size, 3, 3, device=device) * 0.1
    ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1) + \
           torch.randn(batch_size, 3, 3, device=device) * 0.1
    return ccm1, ccm2


@pytest.fixture
def gain_maps(device, batch_size=4, img_size=32):
    """创建测试用的gain map。"""
    # 创建接近1的gain map（模拟白平衡校正）
    gain = torch.ones(batch_size, img_size, img_size, 3, device=device) + \
           torch.randn(batch_size, img_size, img_size, 3, device=device) * 0.1
    return torch.clamp(gain, 0.5, 2.0)
