"""loss.py 的单元测试 —— 覆盖核心损失函数。"""

import pytest
import torch

from loss import (
    angular_loss,
    reconstruction_loss,
    scene_consistency_loss,
    srgb_loss,
    spatial_smoothness_loss,
    srgb_gamma,
    build_srgb_gt,
    crop_to_overlap,
    total_loss,
)


class TestAngularLoss:
    """角度损失测试。"""

    def test_identical_vectors_returns_zero(self, device):
        """相同向量的角度损失应接近0。"""
        pred = torch.randn(4, 32, 32, 3, device=device)
        gt = pred.clone()
        loss = angular_loss(pred, gt)
        assert loss.item() < 1e-5  # 浮点精度容差

    def test_opposite_vectors_returns_two(self, device):
        """反向向量的角度损失应接近2。"""
        pred = torch.ones(4, 32, 32, 3, device=device)
        gt = -torch.ones(4, 32, 32, 3, device=device)
        loss = angular_loss(pred, gt)
        assert abs(loss.item() - 2.0) < 0.1

    def test_orthogonal_vectors_returns_one(self, device):
        """正交向量的角度损失应接近1。"""
        pred = torch.zeros(4, 32, 32, 3, device=device)
        pred[..., 0] = 1.0  # x方向
        gt = torch.zeros(4, 32, 32, 3, device=device)
        gt[..., 1] = 1.0  # y方向
        loss = angular_loss(pred, gt)
        assert abs(loss.item() - 1.0) < 0.1

    def test_broadcast_gt_gain(self, device):
        """gt_gain [B*S, 3] 应能与 pred_gain [B*S, H, W, 3] 广播。"""
        pred = torch.randn(4, 32, 32, 3, device=device)
        gt = torch.randn(4, 1, 1, 3, device=device)  # 广播形状
        loss = angular_loss(pred, gt)
        assert loss.dim() == 0  # 标量
        assert loss.item() >= 0


class TestReconstructionLoss:
    """重建损失测试。"""

    def test_identical_images_returns_zero(self, device):
        """相同图像的重建损失应为0。"""
        pred = torch.randn(4, 32, 32, 3, device=device)
        gt = pred.clone()
        loss = reconstruction_loss(pred, gt)
        assert loss.item() < 1e-6

    def test_symmetry(self, device):
        """重建损失应对称。"""
        a = torch.randn(4, 32, 32, 3, device=device)
        b = torch.randn(4, 32, 32, 3, device=device)
        loss_ab = reconstruction_loss(a, b)
        loss_ba = reconstruction_loss(b, a)
        assert abs(loss_ab.item() - loss_ba.item()) < 1e-6


class TestSceneConsistencyLoss:
    """场景一致性损失测试。"""

    def test_identical_sensors_returns_near_zero(self, device):
        """相同sensor预测的一致性损失应接近0。"""
        pred = torch.randn(2, 3, 32, 32, 3, device=device)
        pred[:, 1, :, :, :] = pred[:, 0, :, :, :]
        pred[:, 2, :, :, :] = pred[:, 0, :, :, :]
        loss = scene_consistency_loss(pred)
        assert loss.item() < 1e-4  # 浮点精度容差

    def test_output_shape(self, device):
        """输出应为标量。"""
        pred = torch.randn(2, 3, 32, 32, 3, device=device)
        loss = scene_consistency_loss(pred)
        assert loss.dim() == 0


class TestSrgbLoss:
    """sRGB损失测试。"""

    def test_identical_returns_zero(self, device):
        """相同输入的sRGB损失应为0。"""
        pred = torch.randn(4, 32, 32, 3, device=device)
        gt = pred.clone()
        loss = srgb_loss(pred, gt)
        assert loss.item() < 1e-6


class TestSpatialSmoothnessLoss:
    """空间平滑损失测试。"""

    def test_constant_gain_returns_zero(self, device):
        """常数gain map的平滑损失应为0。"""
        gain = torch.ones(4, 32, 32, 3, device=device) * 1.5
        raw = torch.randn(4, 32, 32, 3, device=device)
        loss = spatial_smoothness_loss(gain, raw)
        assert loss.item() < 1e-6

    def test_output_shape(self, device):
        """输出应为标量。"""
        gain = torch.randn(4, 32, 32, 3, device=device)
        raw = torch.randn(4, 32, 32, 3, device=device)
        loss = spatial_smoothness_loss(gain, raw)
        assert loss.dim() == 0


class TestSrgbGamma:
    """sRGB gamma编码测试。"""

    def test_zero_input_returns_zero(self, device):
        """零输入应返回零。"""
        x = torch.zeros(4, 32, 32, 3, device=device)
        result = srgb_gamma(x)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_one_input_returns_one(self, device):
        """输入1应接近1（gamma编码后）。"""
        x = torch.ones(4, 32, 32, 3, device=device)
        result = srgb_gamma(x)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-6)

    def test_output_range(self, device):
        """输出应在[0, 1]范围内。"""
        x = torch.randn(4, 32, 32, 3, device=device) * 2
        result = srgb_gamma(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestCropToOverlap:
    """重叠区域裁剪测试。"""

    def test_crop_ratio_one_no_crop(self, device):
        """crop_ratio=1.0时不应裁剪。"""
        images = torch.randn(2, 3, 32, 32, 3, device=device)
        crop_ratios = torch.ones(2, 3, device=device)
        cropped = crop_to_overlap(images, crop_ratios, target_size=32)
        assert cropped.shape == (2, 3, 32, 32, 3)

    def test_crop_ratio_half(self, device):
        """crop_ratio=0.5时应裁剪到一半。"""
        images = torch.randn(2, 3, 32, 32, 3, device=device)
        crop_ratios = torch.ones(2, 3, device=device) * 0.5
        cropped = crop_to_overlap(images, crop_ratios, target_size=16)
        assert cropped.shape == (2, 3, 16, 16, 3)

    def test_output_shape(self, device):
        """输出形状应正确。"""
        B, S, H, W, C = 2, 3, 64, 64, 3
        images = torch.randn(B, S, H, W, C, device=device)
        crop_ratios = torch.rand(B, S, device=device) * 0.5 + 0.5
        target_size = 32
        cropped = crop_to_overlap(images, crop_ratios, target_size)
        assert cropped.shape == (B, S, target_size, target_size, C)


class TestBuildSrgbGt:
    """sRGB GT构建测试。"""

    def test_output_shape(self, device):
        """输出形状应正确。"""
        B, H, W = 4, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        gt_gain = torch.ones(B, 3, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        result = build_srgb_gt(raw, gt_gain, ccm1, ccm2)
        assert result.shape == (B, H, W, 3)

    def test_output_range(self, device):
        """输出应在[0, 1]范围内。"""
        B, H, W = 4, 32, 32
        raw = torch.randn(B, H, W, 3, device=device).abs()
        gt_gain = torch.ones(B, 3, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        result = build_srgb_gt(raw, gt_gain, ccm1, ccm2)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestTotalLoss:
    """总损失函数测试。"""

    def test_basic_loss_computation(self, device):
        """基本损失计算应正常工作。"""
        B, S, H, W = 2, 3, 32, 32
        pred_gain = torch.randn(B * S, H, W, 3, device=device)
        gt_gain = torch.randn(B * S, 3, device=device)
        pred_image = torch.randn(B * S, H, W, 3, device=device)
        gt_image = torch.randn(B * S, H, W, 3, device=device)
        scene_pred_image = pred_image.reshape(B, S, H, W, 3)
        weights = {"awb": 1.0, "reconstruction": 1.0, "consistency": 1.0}

        losses = total_loss(
            pred_gain=pred_gain,
            gt_gain=gt_gain,
            pred_image=pred_image,
            gt_image=gt_image,
            scene_pred_image=scene_pred_image,
            weights=weights,
        )

        assert "total" in losses
        assert "awb" in losses
        assert "reconstruction" in losses
        assert "consistency" in losses
        assert losses["total"].dim() == 0

    def test_with_cropping(self, device):
        """带裁剪的损失计算应正常工作。"""
        B, S, H, W = 2, 3, 32, 32
        pred_gain = torch.randn(B * S, H, W, 3, device=device)
        gt_gain = torch.randn(B * S, 3, device=device)
        pred_image = torch.randn(B * S, H, W, 3, device=device)
        gt_image = torch.randn(B * S, H, W, 3, device=device)
        scene_pred_image = pred_image.reshape(B, S, H, W, 3)
        crop_ratios = torch.ones(B, S, device=device) * 0.8
        weights = {"awb": 1.0, "reconstruction": 1.0, "consistency": 1.0}

        losses = total_loss(
            pred_gain=pred_gain,
            gt_gain=gt_gain,
            pred_image=pred_image,
            gt_image=gt_image,
            scene_pred_image=scene_pred_image,
            weights=weights,
            crop_ratios=crop_ratios,
            loss_crop_size=16,
        )

        assert "total" in losses

    def test_with_smoothness(self, device):
        """带平滑损失的计算应正常工作。"""
        B, S, H, W = 2, 3, 32, 32
        pred_gain = torch.randn(B * S, H, W, 3, device=device)
        gt_gain = torch.randn(B * S, 3, device=device)
        pred_image = torch.randn(B * S, H, W, 3, device=device)
        gt_image = torch.randn(B * S, H, W, 3, device=device)
        scene_pred_image = pred_image.reshape(B, S, H, W, 3)
        raw_image = torch.randn(B * S, H, W, 3, device=device)
        weights = {"awb": 1.0, "reconstruction": 1.0, "consistency": 1.0}

        losses = total_loss(
            pred_gain=pred_gain,
            gt_gain=gt_gain,
            pred_image=pred_image,
            gt_image=gt_image,
            scene_pred_image=scene_pred_image,
            weights=weights,
            raw_image=raw_image,
            smoothness_weight=0.1,
        )

        assert "smoothness" in losses
        assert "total" in losses
