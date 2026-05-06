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
├── model.py                        # AWBTransformer（含 SensorEncoder + 可选 CCMHead）
├── loss.py                         # 损失函数（含后裁剪重叠区域 + 可选 sRGB 损失）
├── train.py                        # 训练入口
├── infer.py                        # 推理入口
├── geometry_utils.py               # 几何工具 + MCS 对齐
├── gt_utils.py                     # GT 提取工具
│
├── data/
│   ├── demo_ImageMatchBindata.py    # dump 数据匹配
│   ├── demo_readMcsRaw.py           # MCS .bin → .npy 解析
│   ├── visualize_mcs.py             # MCS 可视化
│   ├── dng_process_rawpy.py         # [传统] DNG 解码 + GT 一步完成
│   ├── dng_decoder.py               # [解耦] 仅 DNG 解码
│   ├── gt_extractor.py              # [解耦] 独立 GT 提取
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
└── debug_outputs/
    └── train/
```

---

## 数据预处理全流程

```
第一步: demo_ImageMatchBindata.py
  dump/Camera/*.{jpg,dng} + dump/MultispectralSensor/*.bin
  → data/image/ + data/image_dng/ + data/McsBin/

第二步: demo_readMcsRaw.py
  data/McsBin/*.bin → data/Mcsnpy/*.npy (9通道光谱数据)

第三步A [传统耦合]: dng_process_rawpy.py
  一步完成 DNG 解码 + GT 提取 → data/image_processed/*.npz

第三步B [解耦推荐]: dng_decoder.py + gt_extractor.py
  dng_decoder → data/image_raw/*.npz (仅解码)
  gt_extractor → data/image_processed/*.npz (后加GT)
```

详细步骤见 `readme.txt`。

---

## 模型架构

### 架构概览

```
Inputs: raw [B, H, W, 3]    MCS [B, H, W, 10]   CCM1/2 [B, 3, 3]    SensorID [B]    Focal [B]
         │                        │                    │                    │              │
         ▼                        ▼                    ▼                    ▼              ▼
    ┌─────────┐             ┌──────────┐         ┌──────────┐         ┌──────────────┐
    │Conv2d(3→64)│          │Conv2d(10→64)│      │CCMEncoder│         │SensorEncoder │
    │ AvgPool  │             │ AvgPool  │         │(inv+MLP) │         │(embed+MLP)   │
    └────┬────┘             └────┬─────┘         └────┬─────┘         └──────┬───────┘
         │                      │                     │                      │
         ▼                      ▼                     ▼                      ▼
    raw_tokens              mcs_tokens           scale/bias              scale/bias
    [B,256,64]              [B,256,64]            [B,64]                  [B,64]
         │                      │                     │                      │
         └──────────────┬───────┘                     │                      │
                        │                             ▼                      ▼
                        │                   calibrated_tokens = raw_tokens * scale_ccm + bias_ccm
                        │                             │
                        │                             ▼
                        │                   calibrated *= scale_sensor + bias_sensor
                        │                             │
                        └──────────┬──────────────────┘
                                   ▼
                    ┌────────────────────────────┐
                    │  Self-Attention (RAW→RAW)   │
                    │  Cross-Attention (RAW→MCS)  │
                    │  CLS Aggregation            │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────┐
                    │  Head → Gain Map [B,H,W,3] │
                    │  [可选] CCMHead → CCM Delta │
                    └────────────────────────────┘
```

### 关键设计

#### 1. 全视场输入（取消裁剪）

三个模组的图像统一 resize 到 128x128，**不做视场裁剪**，保留每张图的完整内容。

#### 2. MCS 空间对齐（核心改进）

MCS 传感器固定 FOV ≈ Main 模组，Tele/Wide 需要空间对齐：

| 模组 | 对齐方式 | 几何意义 |
|------|---------|---------|
| **Main** | 不变 | MCS FOV ≈ Main FOV |
| **Tele** | 裁剪中心 + resize 放大 | Tele 视场更窄，只看 MCS 中心 |
| **Wide** | resize 缩小 + **reflect** padding | Wide 视场更宽，MCS 只覆盖中心 |

对齐比例：`align_ratio = focal_main / focal_camera`

**MCS 置信度图**（第 10 通道）：Wide 模组的 MCS 外推区域（padding 部分）附带了从中心 1.0 到边缘 ~0.0 的高斯衰减置信度图，让模型知道该区域的 MCS 是外推而非实测数据。Tele 和 Main 的置信度全为 1.0。

#### 3. Sensor ID + 焦距编码

`SensorEncoder` 通过 FiLM 调制（scale/bias）注入模组身份和焦距信息：

- `sensor_id` (0/1/2) → learned embedding
- `focal_length` (连续值) → sinusoidal 编码 + MLP
- 两者融合后调制 RAW tokens，使模型感知当前模组

#### 4. 后裁剪 Loss

模型输出全图 gain_map 后，在 loss 计算时裁出三摄共同覆盖区域：

```
Model 前向 → 全图 gain_map → 全图 pred_image
                              ↓
              用 crop_ratio 裁出重叠区域
                              ↓
              resize 到统一尺寸 (64x64)
                              ↓
              在统一尺寸上计算 L1 + consistency
```

- `angular_loss`：全图 gain 做 spatial mean，保持不变
- `reconstruction_loss`：在裁剪后的重叠区域计算 L1
- `scene_consistency_loss`：在裁剪后的重叠区域计算三摄色度一致性

#### 5. CCM 预测头（可选，默认关闭）

从 CLS token 预测 3x3 CCM 修正量 delta，配合已知 EXIF CCM 得到完整 CCM → 可输出 sRGB。

---

## 训练

### 数据组织

数据集按 "场景" 组织。每个场景包含 3 张图像，通过焦距自动分配 sensor_id：

| sensor_id | 模组 | 焦距排序 |
|-----------|------|----------|
| 0 | Tele | 最大焦距 |
| 1 | Main | 中间焦距 |
| 2 | Wide | 最小焦距 |

### 启动训练

```bash
# 默认配置
python train.py

# 启用 CCM 预测（sRGB 输出）
# 需在 config.yaml 中设置 predict_ccm: true
python train.py
```

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `angular_loss` | 1.0 | 预测增益与 GT 增益的余弦角度差 |
| `reconstruction_loss` | 10.0 | 在重叠区域计算 L1 |
| `consistency_loss` | 2.0 | 同场景三摄色度一致性（无需GT） |
| `srgb_loss` | 1.0 | [可选] sRGB 空间 L1 损失 |

### 输出

- `checkpoints/latest.pth` — 每 epoch 更新
- `checkpoints/best_model.pth` — 总 loss 最小时
- `checkpoints/model_epoch_XXX_loss_X.XXXX.pth` — 周期性保存
- `debug_outputs/train/epoch_XXX.png` — 可视化对比

---

## 断点续训

```yaml
checkpoint:
  resume_from: "./checkpoints/latest.pth"
```

自动加载模型权重和优化器状态，从断点 epoch+1 继续训练。

---

## 推理

```bash
python infer.py
```

输出（到 `inference_outputs/`）：

| 文件 | 说明 |
|------|------|
| `*_input.png` | 原始图像 |
| `*_corrected.png` | AWB 校正后 |
| `*_gain_map.png` | Gain map 可视化 |
| `*_comparison.png` | 对比拼接图 |

---

## 配置文件说明

| 配置块 | 关键参数 | 说明 |
|--------|----------|------|
| `model` | `dim`, `num_heads`, `grid_size` | Transformer 架构 |
| `model` | `sensor_embed_dim`, `predict_ccm` | Sensor 编码维度 / 是否启用 CCM 预测 |
| `training` | `epochs`, `learning_rate` | 训练超参数 |
| `training` | `loss_crop_size` | 后裁剪重叠区域的统一尺寸 |
| `loss_weights` | `awb`, `reconstruction`, `consistency` | 各项损失权重 |

---

## 常见问题

### MCS 对齐不正确

确保 `image_processed/` 中每场景 3 张 `.npz` 的排序与 `Mcsnpy/` 中的 `.npy` 一致。对齐比例 `align_ratio = focal_main / focal_camera`，main 模组焦距由场景中中间焦距自动确定。

### loss 出现 NaN

- 检查 `loss_crop_size` 是否过小（< 4）
- 检查 focal_length 是否为 0（导致除零）
- 降低 learning_rate

### exiftool 找不到

`dng_process_rawpy.py` 和 `dng_decoder.py` 需要 exiftool。在 `config.yaml` 中配置路径或加入系统 PATH。

exiftool 下载：https://exiftool.org/
