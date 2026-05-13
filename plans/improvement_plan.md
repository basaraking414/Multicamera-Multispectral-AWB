# AWB Transformer 改进实施计划

## 背景

通过代码审查和 ML 架构分析，发现以下可改进方向：
- 模型分辨率瓶颈（64×64 极低）
- 训练效率有优化空间（标准 MHSA 可替换）
- MCS 光谱通道未被充分利用
- Loss 权重组合未充分调优
- 代码存在少量重复逻辑

## 第一阶段：零成本 / 低成本改动（立即可试）

### 1.1 图像分辨率提升（高优先级）
- **改动**：config.yaml 中 `data.img_size` 和 `data.mcs_size` 从 `[64, 64]` 改为 `[128, 128]`
- **验证**：运行训练 5-10 epoch，观察 loss 下降趋势和显存占用
- **预期收益**：空间建模精度提升，angular loss 预期改善 5-15%
- **风险**：显存占用约增加 4x，需要检查是否溢出；若溢出改为 96×96
- **验证点**：loss 是否正常下降，显存是否足够，打印 pred_gain 的数值范围是否正常

### 1.2 Loss 权重 Ablation（零成本）
- **改动**：在 config.yaml 中尝试以下组合：
  - 组合A（当前）：awb=1.0, reconstruction=10.0, consistency=2.0
  - 组合B：awb=2.0, reconstruction=5.0, consistency=2.0
  - 组合C：awb=1.0, reconstruction=5.0, consistency=1.0
- **验证**：每组训练 10 epoch，比较 angular_loss 和 reconstruction_loss 的 tradeoff
- **预期收益**：找到更优的 loss 平衡点
- **风险**：低（纯配置改动）
- **验证点**：angular_loss 越低越好，同时 reconstruction 不应显著恶化

### 1.3 grid_size 调优（中优先级）
- **改动**：config.yaml 中 `model.grid_size` 从 16 尝试 24 或 32
- **验证**：配合 1.1 分辨率提升一起测试
- **预期收益**：减少 upsample 比例（从 4x-8x 降到 2x-4x），减少插值伪影
- **风险**：中等（token数量增加，计算量上升 ~2-4x）
- **验证点**：训练速度下降是否可接受，gain_map 的空间细节是否更丰富

### 1.4 Focal Length Encoding 维度提升（中优先级）
- **改动**：config.yaml 中 `model.focal_embed_dim` 从 16 改为 32
- **验证**：单独训练 10 epoch 观察 impact
- **预期收益**：更强的摄像头身份编码能力，支持更精细的模组差异化
- **风险**：低（参数量增加很小）
- **验证点**：不同焦距摄像头的预测 gain 差异是否更合理

## 第二阶段：代码级改动（需审查后实施）

### 2.1 Flash Attention 替换标准 MHSA
- **改动**：model.py 中引入 `torch.nn.attention.sdpa` 或 `flash_attn`
  ```python
  # 替换 nn.MultiheadAttention 为 SDPA 实现
  from torch.nn.attention import SDPBackend, sdpa_kernel
  ```
- **验证**：确认输出数值等价，运行 benchmark 测速
- **预期收益**：训练速度 2-3x 提升（同等显存下可支持更大分辨率）
- **风险**：低（数学等价，数值误差极小）
- **实现位置**：model.py 的 self-attention 层
- **验证点**：loss 曲线是否与标准 MHSA 一致，速度提升是否符合预期

### 2.2 MCS 光谱通道注意力（SE-Block）
- **改动**：在 model.py 的 `mcs_encoder` 后添加 Squeeze-Excite 块
  ```python
  class SEBlock(nn.Module):
      def __init__(self, channels, reduction=4):
          super().__init__()
          self.fc1 = nn.Linear(channels, channels // reduction)
          self.fc2 = nn.Linear(channels // reduction, channels)
      def forward(self, x):
          w = F.adaptive_avg_pool2d(x, 1).flatten(1)
          w = F.relu(self.fc1(w))
          w = torch.sigmoid(self.fc2(w)).unsqueeze(-1).unsqueeze(-1)
          return x * w
  ```
- **验证**：观察训练 loss 下降是否更快
- **预期收益**：更好地利用 9 个光谱通道，尤其混合光源场景
- **风险**：低（引入轻量模块，不破坏现有结构）
- **实现位置**：model.py，AWBTransformer.\_\_init\_\_ 和 forward
- **验证点**：SE 权重分布是否合理（不是所有通道都平均激活）

### 2.3 抽取重复 sRGB 前向逻辑到 model.py
- **改动**：将 train.py:119-134 和 eval.py:119-134 的 sRGB 前向代码抽取为 model.py 的一个方法
- **验证**：确保输出完全一致（数值误差 < 1e-5）
- **预期收益**：消除重复代码，架构更清晰，sRGB 路径可被 inference.py 复用
- **风险**：低（等价的代码移动）
- **实现位置**：model.py 新增 `forward_srgb()` 方法
- **验证点**：运行 eval.py 确认输出与改动前完全一致

### 2.4 配置校验增强
- **改动**：config_loader.py 中增加类型检查和必需字段校验
  ```python
  required_fields = ["data.root_dir", "model.dim", "training.epochs", ...]
  type_checks = {"data.img_size": (list, tuple), "training.loss_weights": dict, ...}
  ```
- **验证**：故意写错 config.yaml，检查报错信息是否友好
- **预期收益**：更早发现配置错误，减少调试时间
- **风险**：极低（只影响加载阶段）
- **验证点**：错误信息是否清晰指出哪个字段有问题

## 第三阶段：实验性改动（需验证后决定）

### 3.1 scene_batch_size > 1
- **当前问题**：scene_batch_size=1 无法跨场景学习
- **改动**：dataloader.py 支持返回多 scene 并行，train.py 适配 batch 维度和 loss 计算
- **风险**：高（涉及数据pipeline和loss的较大改动）
- **优先级**：低，可后续探索

### 3.2 Hierarchical Feature Pyramid
- **改动**：在 raw_encoder 和 mcs_encoder 中增加多尺度特征融合
- **风险**：高（架构性改动，训练稳定性需要重新验证）
- **优先级**：低，作为后续方向

## 实施顺序建议

```
阶段一（无需改代码，1天内可验证）:
  1.1 img_size 64→128      ← 最优先，马上看效果
  1.2 Loss权重 Ablation    ← 并行，零成本
  1.3 grid_size 调优       ← 与 1.1 结合测试
  1.4 focal_embed_dim 调优 ← 单独测试

阶段二（需改代码，2-3天）:
  2.1 Flash Attention      ← 性能关键，优先做
  2.2 SE-Block for MCS    ← 快速实现
  2.3 抽取 sRGB 逻辑      ← 代码质量
  2.4 配置校验            ← 收尾

阶段三（视阶段一/二结果决定）:
  3.1 scene_batch_size > 1
  3.2 Hierarchical Features
```

## 验证指标

每个改动需记录：
- angular_loss（主要）
- reconstruction_loss（辅助）
- consistency_loss（辅助）
- 训练速度（steps/sec）
- 显存峰值占用

建议用 tensorboard 或 wandb 记录训练曲线，方便对比。