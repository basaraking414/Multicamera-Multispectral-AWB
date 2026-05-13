# AWB Transformer — 多光谱辅助自动白平衡

基于 Transformer 和多光谱数据（MCS）的自动白平衡（AWB）校正项目。  
利用 9 通道多光谱传感器数据辅助 RGB 图像进行白平衡增益估计，支持三摄（tele/main/wide）同场景联合训练。

---

## 目录

- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [数据预处理全流程](#数据预处理全流程)
- [模型架构](#模型架构)
- [训练](#训练)
- [评估](#评估)
- [断点续训](#断点续训)
- [推理](#推理)
- [配置文件说明](#配置文件说明)
- [常见问题](#常见问题)

---

## 环境配置

项目使用 `.venv` 虚拟环境，所有依赖已预装。

```bash
# 激活虚拟环境
source .venv/Scripts/activate

# 验证关键依赖
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### 依赖清单

| 包名 | 用途 |
|------|------|
| `torch` (2.11+cu126) | 深度学习框架 |
| `opencv-python` | 图像处理、resize、可视化 |
| `rawpy` | DNG/RAW 解码 |
| `PyExifTool` + exiftool | EXIF 元数据提取（CCM、焦距等） |
| `numpy` | 数据处理 |
| `PyYAML` | 配置解析 |
| `natsort` | 自然排序 |

---

## 项目结构

```
demo/
├── config.yaml                     # 统一配置文件
├── config_loader.py                # 配置加载器（YAML → dataclass）
├── dataloader.py                   # 数据集加载 + MCS 空间对齐
├── model.py                        # AWBTransformer（Pre-LN 残差 Transformer）
├── loss.py                         # 损失函数（逐像素角度 + 边缘感知平滑）
├── train.py                        # 训练入口
├── eval.py                         # 评估脚本（推荐）
├── test.py                         # 评估脚本（兼容性入口，调用 eval.py）
├── infer.py                        # 推理入口
├── geometry_utils.py               # 几何工具 + MCS 对齐 + CCM 矩阵工具
├── visualization.py                # 可视化工具（训练/评估共用）
│
├── tests/                          # 单元测试（pytest）
│   ├── conftest.py                 # 共享 fixtures
│   ├── test_loss.py                # 损失函数测试
│   ├── test_geometry_utils.py      # 几何工具测试
│   └── test_model.py               # 模型组件测试
│
├── data/
│   ├── image/                       # 匹配后的 JPG
│   ├── image_dng/                   # 匹配后的 DNG
│   ├── McsBin/                      # 匹配后的 MCS .bin
│   ├── Mcsnpy/                      # 解析后的 MCS .npy (9ch)
│   └── image_processed/             # 处理完成的 .npz
│
├── checkpoints/                 # 训练断点
│   ├── latest.pth
│   ├── best_model.pth
│   └── model_epoch_*.pth
│
├── debug_outputs/
│   └── train/                    # 训练可视化
│
└── test_outputs/                 # 评估输出
    ├── summary.txt               # 平均指标
    ├── per_scene.txt             # 每场景详细指标
    └── debug/                    # 可视化对比
```

---

## 数据预处理全流程

```
第一步: 数据匹配
  demo_ImageMatchBindata.py
  → data/image/ + data/image_dng/ + data/McsBin/

第二步: MCS 解析
  demo_readMcsRaw.py
  data/McsBin/*.bin → data/Mcsnpy/*.npy (9 通道光谱)

第三步A: DNG 解码（解耦）
  dng_decoder.py — 只解码 RAW + 提取元数据，不含 GT
  → data/image_raw/*.npz

第三步B: GT 提取（解耦，独立运行）
  gt_extractor.py — 在解码后的 .npz 上添加白平衡 GT
  支持色卡自动检测 (colorchecker) 或启发式白块 (white_patch)
  → data/image_processed/*.npz（兼容 dataloader）
```

详细步骤见 `readme.txt`。

---

## 模型架构

### 架构概览

```
Inputs: raw [B, H, W, 3]    MCS [B, H, W, 10]   CCM1/2 [B, 3, 3]    Focal [B]
         │                        │                    │                  │
         ▼                        ▼                    ▼                  ▼
    ┌─────────┐             ┌──────────┐         ┌──────────┐      ┌──────────────┐
    │Conv2d(3→64)│          │Conv2d(10→64)│      │CCMEncoder│      │SensorEncoder │
    │ AvgPool  │             │ AvgPool  │         │(inv+MLP) │      │(sin embed)   │
    └────┬────┘             └────┬─────┘         └────┬─────┘      └──────┬───────┘
         │                      │                     │                   │
         ▼                      ▼                     ▼                   ▼
    raw_tokens              mcs_tokens           scale/bias           scale/bias
    [B,256,64]              [B,256,64]            [B,64]               [B,64]
         │                      │                     │                   │
         └──────────┬───────────┘                     │                   │
                    │                                 ▼                   ▼
                    │                    calibrated = raw * scale_ccm + bias_ccm
                    │                    calibrated *= scale_sensor + bias_sensor
                    │                                 │
                    └──────────┬──────────────────────┘
                               ▼
          ┌──────────────────────────────────────────────────┐
          │  Stage 1: Self-Attention (RAW→RAW, Pre-LN + residual) │
          │  Stage 2: Cross-Attention (RAW→MCS, Pre-LN + residual)│
          │  Stage 3: CLS Aggregation (Pre-LN + residual)         │
          └──────────────────────┬───────────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────────┐
          │  Head → Gain Map [B, H, W, 3]        │
          │  (clamp [1e-4, 4.0])                 │
          │  [可选] CCMHead → CCM Delta [B, 3, 3] │
          └──────────────────────────────────────┘
```

### 关键设计

#### 1. Pre-LN 残差 Transformer

三个注意力层（Self-Attention、Cross-Attention、CLS Attention）均采用 **Pre-LN 残差连接**：

```
output = x + Attention(LayerNorm(q), LayerNorm(k), LayerNorm(v))
```

- 每个注意力层独立 LayerNorm，不共享
- 残差连接确保梯度直通，支持更深网络
- 标准 Transformer 设计，训练更稳定

#### 2. CCM 编码 + 数值保护

CCMEncoder 将两个 3×3 CCM（XYZ→cameraRGB）的逆矩阵展平为 18 维向量，通过 MLP 生成 FiLM scale/bias 调制 RAW token。求逆时加入 `eps * eye(3)`（eps=1e-6）防止奇异矩阵产生 NaN。

#### 3. 焦距连续编码

`SensorEncoder` 使用正弦编码将连续焦距映射为特征向量，经 MLP 后通过 FiLM 调制 RAW tokens：

```
focal_length → sin/cos 编码 → MLP → scale / bias
```

不依赖离散 sensor_id，推理时仅需 EXIF 焦距值，泛化到未见过的模组。

#### 4. MCS 空间对齐 + 置信度

MCS 传感器固定 FOV ≈ Main 模组，Tele/Wide 需要空间对齐并生成置信度图（第 10 通道）：

| 模组 | 对齐方式 | 置信度 |
|------|---------|--------|
| **Main** | 不变 | 全 1.0 |
| **Tele** | 裁剪中心 + resize 放大 | 全 1.0 |
| **Wide** | 缩小 + reflect padding | 中心 1.0 → 边缘高斯衰减至 ~0.0 |

#### 5. 后裁剪 Loss

模型输出全图 gain map，loss 计算时裁出三摄共同覆盖区域（重叠区域）。所有 loss 项统一在裁剪区域上计算：

```
Model 前向 → 全图 gain_map [B*S, H, W, 3] 
              ↓
        用 crop_ratio 裁出重叠区域
              ↓
        resize 到统一尺寸 loss_crop_size
              ↓
         统一尺寸上计算各项 loss
```

#### 6. CCM 预测头（可选，默认开启）

从 CLS token 预测 3×3 CCM 修正量 delta，配合 EXIF CCM（仅 ccm1/D65 基准）计算 sRGB 输出。

---

## 训练

### 数据组织

数据集按"场景"组织。每个场景包含 3 张图像（tele/main/wide），通过焦距降序排列自动分配 sensor_id。

### 启动训练

```bash
python train.py
```

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `angular_loss` | 1.0 | **逐像素**角度损失（1-cos），空间平均前在每个像素上计算 |
| `reconstruction_loss` | 10.0 | 裁剪重叠区域上的 L1 |
| `consistency_loss` | 2.0 | 同场景三摄色度一致性 |
| `smoothness_loss` | 0.05 | 边缘感知空间平滑（raw 强边缘处降低惩罚） |
| `srgb_loss` | 1.0 | [可选，需 predict_ccm] sRGB 空间 L1 |

### 训练配置

```yaml
training:
  epochs: 100
  scene_batch_size: 1
  learning_rate: 0.0001
  lr_scheduler: "cosine"
  loss_crop_size: 32
  smoothness_weight: 0.05    # 空间平滑损失权重
  num_workers: 0             # DataLoader 工作进程数，0=主进程加载
```

- LR Scheduler: CosineAnnealingLR，T_max 自动约束 ≤ epochs，防止 LR 回升
- 梯度裁剪: `max_norm=5.0`
- 检查点: 每 epoch 保存 latest，最佳保存 best，每 10 epoch 周期性保存

### 输出

- `checkpoints/latest.pth` — 每 epoch 更新
- `checkpoints/best_model.pth` — 总 loss 最小时
- `checkpoints/model_epoch_XXX_loss_X.XXXX.pth` — 周期性保存
- `debug_outputs/train/epoch_XXX.png` — 可视化对比

---

## 评估

在独立测试集上评估模型，计算与训练完全一致的 loss 指标（无梯度更新）。

### 启动评估

```bash
# 使用默认配置
python eval.py

# 指定配置文件
python eval.py --config config.yaml

# 指定模型 checkpoint
python eval.py --checkpoint checkpoints/best_model.pth
```

### 输出

| 文件 | 说明 |
|------|------|
| `test_outputs/summary.txt` | 测试集平均 loss 指标 |
| `test_outputs/per_scene.txt` | 每个场景的详细 loss |
| `test_outputs/debug/scene_XXXX.png` | input\|pred\|gt 三行三列对比 |
| `test_outputs/debug/scene_XXXX_mcs_alignment.png` | MCS 对齐可视化 |

---

## 断点续训

```yaml
checkpoint:
  resume_from: "./checkpoints/latest.pth"
```

自动加载模型权重和优化器状态，从断点 epoch+1 继续训练。  
支持加载旧版本检查点（新增 LayerNorm 参数自动跳过，`strict=False`）。

---

## 推理

```bash
# 默认配置
python infer.py

# 指定输入文件和主摄焦距
python infer.py --npz data/image_processed/*.npz --mcs data/Mcsnpy/*.npy --ref-focal 26.0
```

输出（到 `inference_outputs/`）：

| 文件 | 说明 |
|------|------|
| `*_input.png` | 原始图像 |
| `*_corrected.png` | AWB 校正后（gain 受限 [1e-4, 4.0]） |
| `*_gain_map.png` | Gain map 可视化（3 通道） |
| `*_comparison.png` | input / AWB corrected 对比拼接 |

---

## 配置文件说明

| 配置块 | 关键参数 | 默认值 | 说明 |
|--------|----------|--------|------|
| `model.dim` | Transformer 维度 | 64 | 所有 token 和 attention 的隐层维数 |
| `model.num_heads` | 注意力头数 | 4 | 每个注意力层的 head 数 |
| `model.grid_size` | 网格尺寸 | 16 | 16×16 = 256 tokens |
| `model.focal_embed_dim` | 焦距编码维度 | 16 | FocalLengthEncoder 输出维数 |
| `model.predict_ccm` | 启用 CCM 预测 | true | 开启 sRGB 输出路径 |
| `training.loss_crop_size` | 裁剪统一尺寸 | 32 | 所有 loss 在裁剪后的统一尺寸上计算 |
| `training.smoothness_weight` | 空间平滑权重 | 0.05 | 0 则关闭 |
| `training.loss_weights` | 各项损失权重 | awb:1/rec:10/cons:2 | 角度、重建、一致性 |
| `training.num_workers` | DataLoader 进程数 | 0 | 0=主进程加载，Windows建议0-4 |
| `test.loss_crop_size` | 测试裁剪尺寸 | 32 | 默认与训练一致 |

---

## 常见问题

### MCS 对齐不正确

确保 `image_processed/` 中每场景 3 张 `.npz` 的排序与 `Mcsnpy/` 中的 `.npy` 一致。  
对齐比例 `align_ratio = focal_main / focal_camera`，main 模组焦距由场景中中间焦距自动确定。

### Loss 出现 NaN

- 检查 `loss_crop_size` 是否过小（< 4）
- 检查 focal_length 是否为 0（导致除零）
- 降低 learning_rate
- 检查输入数据是否包含 NaN/Inf

### exiftool 找不到

在 `config.yaml` 的 `external_tools.exiftool_path` 中配置 exiftool 路径，或将其加入系统 PATH。

exiftool 下载：https://exiftool.org/