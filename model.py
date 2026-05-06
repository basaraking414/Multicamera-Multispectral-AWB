import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 2D Sinusoidal Positional Encoding
# =============================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.dim = dim
        self.grid_h = grid_h
        self.grid_w = grid_w

        pe = torch.zeros(grid_h * grid_w, dim)
        y_pos = torch.arange(grid_h).unsqueeze(1).repeat(1, grid_w).flatten()
        x_pos = torch.arange(grid_w).unsqueeze(0).repeat(grid_h, 1).flatten()

        dim_half = dim // 2
        if dim_half % 2 != 0:
            dim_half -= 1
        dim_y = dim_half
        dim_x = dim - dim_y

        div_y = torch.exp(torch.arange(0, dim_y, 2) * (-math.log(10000.0) / dim_y))
        div_x = torch.exp(torch.arange(0, dim_x, 2) * (-math.log(10000.0) / dim_x))

        pe_y = torch.zeros(grid_h * grid_w, dim_y)
        pe_x = torch.zeros(grid_h * grid_w, dim_x)

        pe_y[:, 0::2] = torch.sin(y_pos.unsqueeze(1) * div_y.unsqueeze(0))
        pe_y[:, 1::2] = torch.cos(y_pos.unsqueeze(1) * div_y.unsqueeze(0))
        pe_x[:, 0::2] = torch.sin(x_pos.unsqueeze(1) * div_x.unsqueeze(0))
        pe_x[:, 1::2] = torch.cos(x_pos.unsqueeze(1) * div_x.unsqueeze(0))

        pe = torch.cat([pe_y, pe_x], dim=1)  # [grid_h*grid_w, dim]
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, N, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


# =============================================================================
# CCM 编码器
# =============================================================================
class CCMEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.scale_head = nn.Linear(dim, dim)
        self.bias_head = nn.Linear(dim, dim)

    def forward(self, ccm1, ccm2):
        ccm1_inv = torch.linalg.inv(ccm1.float())
        ccm2_inv = torch.linalg.inv(ccm2.float())
        B = ccm1.shape[0]
        ccm_flat = torch.cat([ccm1_inv.reshape(B, -1), ccm2_inv.reshape(B, -1)], dim=-1)
        feat = self.net(ccm_flat)
        scale = self.scale_head(feat)
        bias = self.bias_head(feat)
        return scale, bias


# =============================================================================
# 焦距连续编码器
# =============================================================================
class FocalLengthEncoder(nn.Module):
    """将连续焦距值编码为固定维度的特征向量（正弦编码 + MLP）。"""

    def __init__(self, dim=16, max_focal=200.0):
        super().__init__()
        self.max_focal = max_focal
        freq = 2.0 ** torch.linspace(0, 5, dim // 2)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, focal_length: torch.Tensor) -> torch.Tensor:
        # focal_length: [B], 归一化到 [0, 1]
        norm = focal_length / self.max_focal  # [B]
        enc = norm.unsqueeze(1) * self.freq.unsqueeze(0)  # [B, dim/2]
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=1)  # [B, dim]
        return self.mlp(enc)


# =============================================================================
# Sensor 编码器（FiLM 调制）
# =============================================================================
class SensorEncoder(nn.Module):
    """将连续焦距编码为 FiLM scale/bias，调制 RAW tokens。

    焦距本身足以唯一标识三摄模组（tele/main/wide 各不相同的固定焦距），
    无需离散的 sensor_id embedding。推理时只需从 EXIF 读取焦距即可。
    """

    def __init__(self, dim=64, focal_embed_dim=16):
        super().__init__()
        self.focal_encoder = FocalLengthEncoder(dim=focal_embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(focal_embed_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.scale_head = nn.Linear(dim, dim)
        self.bias_head = nn.Linear(dim, dim)

    def forward(self, focal_length: torch.Tensor):
        fl_feat = self.focal_encoder(focal_length)  # [B, focal_embed_dim]
        feat = self.fusion(fl_feat)
        scale = self.scale_head(feat)
        bias = self.bias_head(feat)
        return scale, bias


# =============================================================================
# CCM 预测头（可选）
# =============================================================================
class CCMHead(nn.Module):
    """从 CLS token 预测 3x3 CCM 修正量 delta。

    最终 CCM = inverse(xyz2camera_rgb) + delta，实现 cameraRGB → XYZ。
    """

    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 9),  # 3x3 = 9
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        # cls_token: [B, 1, dim]
        delta = self.net(cls_token)  # [B, 1, 9]
        return delta.view(-1, 3, 3)  # [B, 3, 3]


# =============================================================================
# AWB Transformer 主模型
# =============================================================================
class AWBTransformer(nn.Module):
    def __init__(
        self,
        dim=64,
        num_heads=4,
        grid_size=16,
        use_positional_encoding=True,
        focal_embed_dim=16,
        predict_ccm=False,
    ):
        super().__init__()

        self.grid = grid_size
        self.dim = dim
        self.use_positional_encoding = use_positional_encoding
        self.predict_ccm = predict_ccm

        # ===== RAW encoder =====
        self.raw_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        # ===== MCS encoder（9 光谱通道 + 1 置信度通道）=====
        self.mcs_encoder = nn.Sequential(
            nn.Conv2d(10, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # ===== 2D Positional Encoding =====
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding2D(dim, grid_size, grid_size)
        else:
            self.pos_enc = None

        # ===== tokens =====
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # ===== CCM 编码器 =====
        self.ccm_encoder = CCMEncoder(dim)

        # ===== Sensor 编码器（FiLM 调制，仅依赖焦距）=====
        self.sensor_encoder = SensorEncoder(dim, focal_embed_dim)

        # ===== attention =====
        self.raw_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # CLS聚合
        self.cls_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # ===== CLS调制 =====
        self.cls_scale = nn.Linear(dim, dim)
        self.cls_bias = nn.Linear(dim, dim)

        # ===== 输出 head（per-token gain map）=====
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )

        # ===== CCM 预测头（可选）=====
        if predict_ccm:
            self.ccm_head = CCMHead(dim)

    def forward(self, raw, mcs, ccm1, ccm2, focal_length=None):
        """
        Args:
            raw:            [B, H, W, 3]  线性 RAW 图像
            mcs:            [B, H, W, 10] 多光谱数据（9 光谱 + 1 置信度）
            ccm1, ccm2:     [B, 3, 3]     XYZ→cameraRGB CCM 矩阵
            focal_length:   [B]           焦距值（mm），唯一标识模组
        Returns:
            gain_map:  [B, H, W, 3]  空间 AWB 增益
            ccm_delta: [B, 3, 3] 或 None  CCM 修正量（predict_ccm=True 时返回）
        """
        B, H, W, C = raw.shape

        # ===== RAW =====
        raw = raw.permute(0, 3, 1, 2)  # B,C,H,W
        raw_feat = self.raw_encoder(raw)
        raw_feat = self.pool(raw_feat)
        raw_tokens = raw_feat.flatten(2).transpose(1, 2)  # B,N,C

        # ===== MCS（10 通道：9 光谱 + 1 置信度）=====
        mcs = mcs.permute(0, 3, 1, 2)  # B,10,H,W
        mcs_feat = self.mcs_encoder(mcs)
        mcs_feat = self.pool(mcs_feat)
        mcs_tokens = mcs_feat.flatten(2).transpose(1, 2)

        # ===== 2D Positional Encoding =====
        if self.pos_enc is not None:
            raw_tokens = self.pos_enc(raw_tokens)
            mcs_tokens = self.pos_enc(mcs_tokens)

        # ===== CCM 校准（消除传感器色彩特性）=====
        scale, bias = self.ccm_encoder(ccm1, ccm2)
        calibrated_tokens = raw_tokens * scale.unsqueeze(1) + bias.unsqueeze(1)

        # ===== Sensor 调制（用焦距区分三摄模组）=====
        if focal_length is not None:
            s_scale, s_bias = self.sensor_encoder(focal_length)
            calibrated_tokens = calibrated_tokens * s_scale.unsqueeze(1) + s_bias.unsqueeze(1)

        # ===== CLS =====
        cls_token = self.cls_token.expand(B, -1, -1)

        # ===== Stage1: 标准颜色特征 self-attention =====
        refined_tokens, _ = self.raw_self_attn(calibrated_tokens, calibrated_tokens, calibrated_tokens)

        # ===== Stage2: cross-attention (标准颜色 ← MCS) =====
        fused_tokens, _ = self.cross_attn(refined_tokens, mcs_tokens, mcs_tokens)

        # ===== Stage3: CLS聚合 =====
        all_tokens = torch.cat([fused_tokens, mcs_tokens], dim=1)
        cls_out, _ = self.cls_attn(cls_token, all_tokens, all_tokens)

        # ===== CCM 预测 =====
        ccm_delta = None
        if self.predict_ccm:
            ccm_delta = self.ccm_head(cls_out)  # [B, 3, 3]

        # ===== CLS调制 =====
        scale = torch.sigmoid(self.cls_scale(cls_out))  # B,1,C
        bias = self.cls_bias(cls_out)
        fused_tokens = fused_tokens * scale + bias

        # ===== 输出 spatial AWB gain =====
        gain_map = self.head(fused_tokens)  # B,N,3
        gain_map = gain_map.view(B, self.grid, self.grid, 3)
        gain_map = F.softplus(gain_map) + 1e-4

        # ===== upsample到原图大小 =====
        gain_map = gain_map.permute(0, 3, 1, 2)  # B,3,H,W
        gain_map = F.interpolate(gain_map, size=(H, W), mode='bilinear', align_corners=False)
        gain_map = gain_map.permute(0, 2, 3, 1)  # B,H,W,3

        return gain_map, ccm_delta
