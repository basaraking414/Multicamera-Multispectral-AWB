"""model.py 的单元测试 —— 覆盖模型组件和前向传播。"""

import pytest
import torch

from model import (
    PositionalEncoding2D,
    CCMEncoder,
    FocalLengthEncoder,
    SensorEncoder,
    CCMHead,
    AWBTransformer,
)


class TestPositionalEncoding2D:
    """2D位置编码测试。"""

    def test_output_shape(self, device):
        """输出形状应与输入一致。"""
        dim, grid_h, grid_w = 32, 8, 8
        pe = PositionalEncoding2D(dim, grid_h, grid_w).to(device)
        x = torch.randn(2, grid_h * grid_w, dim, device=device)
        out = pe(x)
        assert out.shape == x.shape

    def test_pe_buffer_shape(self, device):
        """位置编码buffer形状应正确。"""
        dim, grid_h, grid_w = 32, 8, 8
        pe = PositionalEncoding2D(dim, grid_h, grid_w).to(device)
        assert pe.pe.shape == (1, grid_h * grid_w, dim)

    def test_odd_dim_handled(self, device):
        """奇数dim应被正确处理（补零）。"""
        dim, grid_h, grid_w = 33, 8, 8
        pe = PositionalEncoding2D(dim, grid_h, grid_w).to(device)
        x = torch.randn(2, grid_h * grid_w, dim, device=device)
        out = pe(x)
        assert out.shape == x.shape


class TestCCMEncoder:
    """CCM编码器测试。"""

    def test_output_shapes(self, device):
        """输出scale和bias形状应正确。"""
        dim = 32
        encoder = CCMEncoder(dim).to(device)
        ccm1 = torch.randn(4, 3, 3, device=device) + torch.eye(3, device=device) * 5
        ccm2 = torch.randn(4, 3, 3, device=device) + torch.eye(3, device=device) * 5
        scale, bias = encoder(ccm1, ccm2)
        assert scale.shape == (4, dim)
        assert bias.shape == (4, dim)

    def test_numerical_stability(self, device):
        """输入接近奇异矩阵时不应产生NaN。"""
        dim = 32
        encoder = CCMEncoder(dim).to(device)
        # 创建接近奇异的矩阵
        ccm1 = torch.zeros(4, 3, 3, device=device)
        ccm1[:, 0, 0] = 1.0
        ccm1[:, 1, 1] = 1.0
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(4, -1, -1)
        scale, bias = encoder(ccm1, ccm2)
        assert not torch.isnan(scale).any()
        assert not torch.isnan(bias).any()


class TestFocalLengthEncoder:
    """焦距编码器测试。"""

    def test_output_shape(self, device):
        """输出形状应正确。"""
        dim = 16
        encoder = FocalLengthEncoder(dim).to(device)
        focal = torch.tensor([50.0, 26.0, 13.0], device=device)
        out = encoder(focal)
        assert out.shape == (3, dim)

    def test_different_focals_different_output(self, device):
        """不同焦距应产生不同编码。"""
        dim = 16
        encoder = FocalLengthEncoder(dim).to(device)
        focal1 = torch.tensor([50.0], device=device)
        focal2 = torch.tensor([13.0], device=device)
        out1 = encoder(focal1)
        out2 = encoder(focal2)
        assert not torch.allclose(out1, out2)


class TestSensorEncoder:
    """Sensor编码器测试。"""

    def test_output_shapes(self, device):
        """输出scale和bias形状应正确。"""
        dim = 32
        focal_embed_dim = 16
        encoder = SensorEncoder(dim, focal_embed_dim).to(device)
        focal = torch.tensor([50.0, 26.0, 13.0], device=device)
        scale, bias = encoder(focal)
        assert scale.shape == (3, dim)
        assert bias.shape == (3, dim)


class TestCCMHead:
    """CCM预测头测试。"""

    def test_output_shape(self, device):
        """输出形状应为[B, 3, 3]。"""
        dim = 32
        head = CCMHead(dim).to(device)
        cls_token = torch.randn(4, 1, dim, device=device)
        ccm_delta = head(cls_token)
        assert ccm_delta.shape == (4, 3, 3)


class TestAWBTransformer:
    """AWB Transformer主模型测试。"""

    def test_forward_without_ccm(self, device):
        """不预测CCM时的前向传播。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=True,
            focal_embed_dim=8,
            predict_ccm=False,
        ).to(device)
        model.eval()

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        focal = torch.tensor([50.0, 26.0], device=device)

        gain_map, ccm_delta = model(raw, mcs, ccm1, ccm2, focal)

        assert gain_map.shape == (B, H, W, 3)
        assert ccm_delta is None

    def test_forward_with_ccm(self, device):
        """预测CCM时的前向传播。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=True,
            focal_embed_dim=8,
            predict_ccm=True,
        ).to(device)
        model.eval()

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        focal = torch.tensor([50.0, 26.0], device=device)

        gain_map, ccm_delta = model(raw, mcs, ccm1, ccm2, focal)

        assert gain_map.shape == (B, H, W, 3)
        assert ccm_delta.shape == (B, 3, 3)

    def test_forward_without_focal(self, device):
        """无焦距输入时的前向传播。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=True,
            focal_embed_dim=8,
            predict_ccm=False,
        ).to(device)
        model.eval()

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)

        gain_map, ccm_delta = model(raw, mcs, ccm1, ccm2, focal_length=None)

        assert gain_map.shape == (B, H, W, 3)
        assert ccm_delta is None

    def test_gain_map_range(self, device):
        """gain map应在[1e-4, 4.0]范围内。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=True,
            focal_embed_dim=8,
            predict_ccm=False,
        ).to(device)
        model.eval()

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        focal = torch.tensor([50.0, 26.0], device=device)

        gain_map, _ = model(raw, mcs, ccm1, ccm2, focal)

        assert gain_map.min() >= 1e-4
        assert gain_map.max() <= 4.0

    def test_gradient_flow(self, device):
        """梯度应能正常回传。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=True,
            focal_embed_dim=8,
            predict_ccm=True,
        ).to(device)

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        focal = torch.tensor([50.0, 26.0], device=device)

        gain_map, ccm_delta = model(raw, mcs, ccm1, ccm2, focal)
        loss = gain_map.sum() + ccm_delta.sum()
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_without_positional_encoding(self, device):
        """不使用位置编码时应正常工作。"""
        model = AWBTransformer(
            dim=32, num_heads=4, grid_size=8,
            use_positional_encoding=False,
            focal_embed_dim=8,
            predict_ccm=False,
        ).to(device)
        model.eval()

        B, H, W = 2, 32, 32
        raw = torch.randn(B, H, W, 3, device=device)
        mcs = torch.randn(B, H, W, 10, device=device)
        ccm1 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        ccm2 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        focal = torch.tensor([50.0, 26.0], device=device)

        gain_map, _ = model(raw, mcs, ccm1, ccm2, focal)

        assert gain_map.shape == (B, H, W, 3)
