"""geometry_utils.py 的单元测试 —— 覆盖几何变换工具函数。"""

import pytest
import torch
import numpy as np

from geometry_utils import (
    focal_to_crop_ratio,
    center_crop,
    resize_image,
    align_mcs_to_fov,
    safe_inv_ccm,
    ensure_float01,
)


class TestFocalToCropRatio:
    """焦距到裁剪比例转换测试。"""

    def test_equal_focals_returns_one(self):
        """相同焦距应返回1.0。"""
        ratio = focal_to_crop_ratio(50.0, 50.0)
        assert abs(ratio - 1.0) < 1e-6

    def test_tele_focal_returns_ratio(self):
        """长焦距（如100mm）对短参考焦距（如26mm）应返回比例。"""
        ratio = focal_to_crop_ratio(100.0, 26.0)
        # focal_to_crop_ratio 返回 focal_length / reference_focal，限制在[0.05, 1.0]
        # 100/26 ≈ 3.85，被clip到1.0
        assert ratio <= 1.0
        assert ratio >= 0.05

    def test_wide_focal_returns_ratio(self):
        """短焦距（如13mm）对长参考焦距（如26mm）应返回比例。"""
        ratio = focal_to_crop_ratio(13.0, 26.0)
        # 13/26 = 0.5，在[0.05, 1.0]范围内
        assert ratio > 0.05
        assert ratio <= 1.0

    def test_ratio_clipping(self):
        """比例应被限制在[0.05, 1.0]范围内。"""
        # 极端长焦
        ratio = focal_to_crop_ratio(1000.0, 26.0)
        assert ratio >= 0.05
        assert ratio <= 1.0

        # 极端广角
        ratio = focal_to_crop_ratio(1.0, 26.0)
        assert ratio >= 0.05
        assert ratio <= 1.0

    def test_zero_focal_handled(self):
        """焦距为0时应被clamp到1e-4。"""
        ratio = focal_to_crop_ratio(0.0, 26.0)
        assert ratio >= 0.05


class TestCenterCrop:
    """中心裁剪测试。"""

    def test_crop_ratio_one_no_crop(self):
        """crop_ratio=1.0时应返回原图。"""
        image = np.random.randn(64, 64, 3).astype(np.float32)
        cropped, (x0, y0, x1, y1) = center_crop(image, 1.0)
        assert cropped.shape == (64, 64, 3)

    def test_crop_ratio_half(self):
        """crop_ratio=0.5时应裁剪到一半。"""
        image = np.random.randn(64, 64, 3).astype(np.float32)
        cropped, (x0, y0, x1, y1) = center_crop(image, 0.5)
        assert cropped.shape[0] == 32
        assert cropped.shape[1] == 32

    def test_crop_box_coordinates(self):
        """裁剪框坐标应正确。"""
        image = np.random.randn(64, 64, 3).astype(np.float32)
        _, (x0, y0, x1, y1) = center_crop(image, 0.5)
        assert x0 >= 0
        assert y0 >= 0
        assert x1 <= 64
        assert y1 <= 64
        assert x1 - x0 == 32
        assert y1 - y0 == 32

    def test_minimum_size(self):
        """裁剪尺寸应至少为4。"""
        image = np.random.randn(10, 10, 3).astype(np.float32)
        cropped, _ = center_crop(image, 0.1)
        # crop_h = max(4, min(10, int(round(10 * 0.1)))) = max(4, 1) = 4
        assert cropped.shape[0] >= 4
        assert cropped.shape[1] >= 4


class TestResizeImage:
    """图像resize测试。"""

    def test_basic_resize(self):
        """基本resize应正常工作。"""
        image = np.random.randn(64, 64, 3).astype(np.float32)
        resized = resize_image(image, (32, 32))
        assert resized.shape == (32, 32, 3)

    def test_multispectral_resize(self):
        """多光谱图像（>4通道）应逐通道resize。"""
        image = np.random.randn(64, 64, 9).astype(np.float32)  # 9通道MCS
        resized = resize_image(image, (32, 32))
        assert resized.shape == (32, 32, 9)

    def test_single_channel(self):
        """单通道图像应正常resize。"""
        image = np.random.randn(64, 64, 1).astype(np.float32)
        resized = resize_image(image, (32, 32))
        # cv2.resize对单通道图像可能返回(32, 32)而非(32, 32, 1)
        assert resized.shape[:2] == (32, 32)


class TestAlignMcsToFov:
    """MCS空间对齐测试。"""

    def test_main_mode_ratio_one(self):
        """ratio=1.0时（main模式）应不变。"""
        mcs = np.random.randn(64, 64, 9).astype(np.float32)
        aligned, confidence = align_mcs_to_fov(mcs, 1.0, (32, 32))
        assert aligned.shape == (32, 32, 9)
        assert confidence.shape == (32, 32)
        assert np.allclose(confidence, 1.0)

    def test_tele_mode_ratio_less_than_one(self):
        """ratio<1.0时（tele模式）应裁剪中心放大。"""
        mcs = np.random.randn(64, 64, 9).astype(np.float32)
        aligned, confidence = align_mcs_to_fov(mcs, 0.5, (32, 32))
        assert aligned.shape == (32, 32, 9)
        assert confidence.shape == (32, 32)
        # tele模式置信度应全1
        assert np.allclose(confidence, 1.0)

    def test_wide_mode_ratio_greater_than_one(self):
        """ratio>1.0时（wide模式）应缩小+reflect填充。"""
        mcs = np.random.randn(64, 64, 9).astype(np.float32)
        aligned, confidence = align_mcs_to_fov(mcs, 2.0, (32, 32))
        assert aligned.shape == (32, 32, 9)
        assert confidence.shape == (32, 32)
        # wide模式置信度应从中心1.0衰减到边缘
        assert confidence.max() > 0.5
        assert confidence.min() < 1.0

    def test_output_types(self):
        """输出应为numpy数组。"""
        mcs = np.random.randn(64, 64, 9).astype(np.float32)
        aligned, confidence = align_mcs_to_fov(mcs, 1.0, (32, 32))
        assert isinstance(aligned, np.ndarray)
        assert isinstance(confidence, np.ndarray)


class TestSafeInvCcm:
    """CCM矩阵求逆测试。"""

    def test_identity_matrix(self, device):
        """单位矩阵的逆应为单位矩阵。"""
        ccm = torch.eye(3, device=device).unsqueeze(0)
        inv = safe_inv_ccm(ccm)
        assert torch.allclose(inv, ccm, atol=1e-6)

    def test_batch_processing(self, device):
        """批量处理应正常工作。"""
        ccm = torch.randn(4, 3, 3, device=device)
        # 确保矩阵可逆
        ccm = ccm + torch.eye(3, device=device) * 5
        inv = safe_inv_ccm(ccm)
        assert inv.shape == (4, 3, 3)

    def test_inverse_property(self, device):
        """CCM @ inv(CCM) 应接近单位矩阵。"""
        ccm = torch.randn(4, 3, 3, device=device)
        ccm = ccm + torch.eye(3, device=device) * 5
        inv = safe_inv_ccm(ccm)
        product = torch.bmm(ccm, inv)
        identity = torch.eye(3, device=device).unsqueeze(0).expand(4, -1, -1)
        assert torch.allclose(product, identity, atol=1e-5)

    def test_singular_matrix_handled(self, device):
        """奇异矩阵应通过eps正则化处理。"""
        # 创建奇异矩阵
        ccm = torch.zeros(1, 3, 3, device=device)
        ccm[0, 0, 0] = 1.0
        ccm[0, 1, 1] = 1.0
        # 第三行全0，奇异
        inv = safe_inv_ccm(ccm)
        # 不应产生NaN或Inf
        assert not torch.isnan(inv).any()
        assert not torch.isinf(inv).any()


class TestEnsureFloat01:
    """图像归一化测试。"""

    def test_uint8_image(self):
        """8-bit图像应归一化到[0, 1]。"""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ensure_float01(image)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_float_image_already_normalized(self):
        """已归一化的图像应保持不变。"""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = ensure_float01(image)
        assert np.allclose(result, image, atol=1e-6)

    def test_float_image_not_normalized(self):
        """未归一化的浮点图像应被归一化。"""
        image = np.random.rand(64, 64, 3).astype(np.float32) * 10
        result = ensure_float01(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
