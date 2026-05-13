# AWB Transformer 全盘复查与改进计划

> 创建日期: 2026-05-10
> 来源: Claude 双代理审查（代码质量审查 + 模型架构分析）

---

## 一、Critical（必须修复）

### C1. DataLoader 场景分组依赖排序文件名

**文件**: `dataloader.py:21-31`

**问题**: 通过 `sorted(os.listdir(...))` 再按位置 `[i:i+3]` 分组来形成 scene。任何文件缺失或命名不一致会导致所有后续场景错位，sensor_id 分配错误 → **静默数据损坏**。

**修复方案**:
- 从文件名解析 scene_id（如前缀 `scene001_tele`、`scene001_main`），按 scene_id 显式分组
- 验证每组恰好 3 个文件且焦距符合 tele/main/wide 预期

---

### C2. Transformer 缺少残差连接和 LayerNorm

**文件**: `model.py:267-273`

**问题**: 3 个 `MultiheadAttention` 都是裸调用，无 skip connection + 无 LayerNorm。梯度必须完全通过注意力权重路径传播，易梯度消失/训练震荡。

**修复方案**:
- 将每个 attention 包装为 Pre-LN 残差块：

```python
raw_norm = nn.LayerNorm(dim)
attn_out = self.raw_self_attn(raw_norm(calibrated_tokens), ...)[0]
refined_tokens = calibrated_tokens + attn_out
```

---

### C3. CCM 矩阵求逆无数值保护

**文件**: `model.py:62-63`, `loss.py:113-114`

**问题**: `torch.linalg.inv(ccm1.float())` 在矩阵奇异时产生 NaN，NaN 通过梯度污染全部参数。

**修复方案**:
- 加入小对角正则化：`ccm_reg = ccm1 + eps * eye(3)` 再求逆
- 训练中加 `torch.isnan(ccm1_inv).any()` 断言及早捕获

---

### C4. PositionalEncoding2D 在 dim 奇数时崩溃

**文件**: `model.py:22-37`

**问题**: `dim_x` 可能为奇数导致 `pe_x[:, 1::2]` 与 cos 输出列数不匹配。当前 dim=64 不触发，属潜伏 bug。

**修复方案**:
- 保证 `dim_x` 和 `dim_y` 均为偶数；或加 `assert dim % 4 == 0`

---

### C5. Cosine LR 在 T_max 后回升

**文件**: `config_loader.py:265-271`

**问题**: `CosineAnnealingLR(T_max=100)` 在 step > 100 后 LR 重新上升。若 epochs 超过 100 会破坏收敛。

**修复方案**:
- 将 `T_max` 默认设为 `cfg.epochs`，或加验证 `T_max >= epochs`

---

## 二、Important（应该修复）

### I1. Loss 不监督空间变化（P0 优先级）

**文件**: `dataloader.py:153`, `loss.py:65-68, 79-80`

**问题**:
- `angular_loss` 对 pred_gain 做 `spatial_mean` → 空间信息全丢
- `reconstruction_loss` 的 GT 是 `image * uniform_gain` → 模型被训练成输出全局均匀 gain
- **架构支持空间变化但 loss 在惩罚它**

**修复方案**:
1. Patch-wise angular loss：将 gain map 分 N×N 块，每块独立算角度损失
2. 或只在 angular loss 上监督全局色温，用弱监督（如 gray-world）允许空间变化

---

### I2. MCS token 在 CLS 路径中被重复使用

**文件**: `model.py:273`

**问题**: `all_tokens = concat([fused_tokens, mcs_tokens])` — mcs_tokens 已在 cross_attn 中用过，这里又直接拼接到 CLS 的序列中，序列长度翻倍。

**修复方案**:
- 去掉 mcs_tokens 拼接，改为 `all_tokens = fused_tokens`
- 或增加专门的 CLS↔MCS cross-attention

---

### I3. build_srgb_gt 平均两个 CCM 物理上无意义

**文件**: `loss.py:114-116`

**问题**: `(ccm1_inv + ccm2_inv) / 2` — 两个不同光照的 CCM 平均不对应任何真实光照。

**修复方案**:
- 只用其中一个 CCM，或基于场景元数据加权混合，或跳过 sRGB loss

---

### I4. DataLoader NaN 检查在 worker 中抛异常会静默挂起

**文件**: `dataloader.py:125-134`

**问题**: `__getitem__` 中 raise ValueError，多 worker 模式下异常不传播到主进程。

**修复方案**:
- 将验证移至预处理阶段，或在 `__getitem__` 中 try/except + 返回 sentinel

---

### I5. Angular loss 与 Reconstruction loss 作用在不同空间区域

**文件**: `loss.py:170, 181-185`

**问题**: angular_loss 用全图 gain，reconstruction_loss 用裁剪后的重叠区域。

**修复方案**:
- 统一：angular loss 也在裁剪区域上计算，或明确文档说明这是有意的

---

### I6. Gain 预测无上界

**文件**: `model.py:289`

**问题**: `F.softplus(gain) + 1e-4` 只有下界（1e-4）无上界。

**修复方案**: `F.softplus(gain).clamp(1e-4, 4.0)` 或使用 `sigmoid * (max-min) + min`

---

### I7. Debug 可视化可能因非连续 tensor 崩溃

**文件**: `train.py:31`, `test.py:40`

**问题**: reshape/permute 后的 view 直接 `.numpy()` 若非连续则抛异常。

**修复方案**: `.contiguous().detach().cpu().numpy()`

---

### I8. sRGB loss 路径中重复计算 CCM

**文件**: `train.py:210, 228-229`

**问题**: `flat_ccm1` 已计算，又在 sRGB 段重新 `reshape(batch_size * scene_size, 3, 3)`。

**修复方案**: 复用 `flat_ccm1` / `flat_ccm2`

---

### I9. crop_boxes 死代码

**文件**: `dataloader.py:160`

**问题**: 创建了 `np.zeros(4)` 并 stacked 到 batch 中，但从未使用。

**修复方案**: 删除

---

## 三、Minor

| # | 问题 | 文件 |
|---|------|------|
| m1 | focal_embed_dim 奇数时 FocalLengthEncoder 维度不匹配 | `model.py:81` |
| m2 | 无 EMA 模型权重用于推理 | — |
| m3 | `allow_pickle=True` 安全风险 | `dataloader.py:39,58` |
| m4 | 梯度裁剪 `max_norm=1.0` 对 200K 参数模型可能过强 | `train.py:269` |

---

## 四、模型架构改进建议（按优先级排序）

### P0. 修复 Loss 空间监督（见 I1）
**前提条件**：必须先修复此问题，否则所有空间改进无效。

### P1. 增加 Pre-LN 残差连接（见 C2）
**预期**：训练稳定，可堆叠多层

### P2. 多尺度 Token 金字塔
**描述**：粗分支 16×16（全局）+ 细分支 32×32（局部细节），cross-scale attention 融合
**预期**：4× 空间 token 数，真实空间变化增益
**工作量**：~150 行，修改 `model.py`

### P3. CNN Decoder 替代双线性上采样
**描述**：用 ConvTranspose2d 渐进上采样，学习增益图高频结构
**预期**：减少上采样伪影
**工作量**：~80 行，修改 `model.py`

### P4. 边缘感知梯度平滑 Loss
**描述**：在 gain map 上加入边缘感知的平滑约束
**预期**：空间连贯的增益图
**工作量**：~30 行，修改 `loss.py`

### P5. Gated Fusion 替代简单 Cross-attention
**描述**：`gate * raw + (1-gate) * mcs_attn_out` 自适应融合
**预期**：模型学习 RAW/MCS 权重分配
**工作量**：~50 行，修改 `model.py`

### P6. MCS Self-attention
**描述**：MCS token 间做 self-attention 建模光谱关系
**预期**：更好利用 MCS 光谱信息

### P7. 焦距编码去 MLP
**描述**：去掉 FocalLengthEncoder 的 MLP，用纯正弦编码 + 可学习 camera embedding
**预期**：避免过拟合到 3 个离散焦距值

---

## 五、推荐执行顺序

```
Phase 1（数据安全）:      C1 修复 dataloader 场景分组
Phase 2（训练稳定）:      C2 + C3 + C4 + C5 + I6
Phase 3（Loss 正确性）:   I1 + I3 + I5 + P0 patch-wise loss
Phase 4（空间能力）:      P2 + P4（前提是 Phase 3 完成）
Phase 5（性能优化）:      P3 + P5 + m2（EMA）
Phase 6（清理）:          I8 + I9 + m3 + m4
```

---

## 六、验证标准

- **修复后**
  - `python train.py` 从 epoch 0 到 100 无 NaN/inf，loss 平稳下降
  - `python test.py` 所有场景通过且数值合理
  - Angular error 不再有异常峰值

- **架构改进后**
  - Gain map 可视化显示出空间变化（不同区域不同增益）
  - Debug 图像中叠加热力图后能看到有意义的空间模式
  - Angular error + 重建误差均下降