# AWB Transformer — 项目规范

## 项目概述

基于 Transformer 和多光谱数据（MCS）的自动白平衡（AWB）校正项目。利用 9 通道多光谱传感器数据辅助 3 摄（tele/main/wide）RGB 图像进行白平衡增益估计。

---

## 环境

- **Python**: 虚拟环境 `.venv/`（已预装所有依赖）
- **激活**: `source .venv/Scripts/activate`
- **关键依赖**: torch, opencv-python, rawpy, PyExifTool, numpy, PyYAML
- **exiftool**: `E:/ricky/research/oppo-project-2025/exiftool-13.53_64/exiftool-13.53_64/exiftool.exe`

---

## 数据预处理流水线

```
Step 1: 数据匹配
  demo_ImageMatchBindata.py
  → data/image/ (JPG) + data/image_dng/ (DNG) + data/McsBin/ (MCS.bin)

Step 2: MCS 解析
  demo_readMcsRaw.py
  → data/Mcsnpy/*.npy (9ch 光谱)

Step 3A [传统]: 解码+GT 一步
  dng_process_rawpy.py
  → data/image_processed/*.npz

Step 3B [解耦推荐]: 解码 + GT 分离
  dng_decoder.py → data/image_raw/*.npz
  gt_extractor.py → data/image_processed/*.npz
```

**MCS 旋转校正**: `demo_readMcsRaw.py` 中 `MCS_ROT90_K = 1`（np.rot90 k=1）。**修改旋转参数后必须删除旧 .npy 重新运行**。不在 dataloader 或 model 中做旋转。

---

## 代码规范

### 通用规则
- 所有路径用相对路径（相对于项目根目录），通过 `config.yaml` 统一管理
- 中文注释用于解释 WHY（非显而易见的原因），不写 WHAT（代码本身已表达）
- 不做过早抽象、不必要的错误处理、或为了"未来需求"的设计

### 命名规范
- 文件/函数: snake_case
- 类名: PascalCase
- 配置 key: snake_case
- pytorch tensor shape 注释: `[B, C, H, W]` 或 `[B, N, C]`

### 配置管理
- `config.yaml` 是唯一配置入口
- `config_loader.py` 解析为结构化 dataclass (`Config`)
- 新增配置项需同步修改 `config_loader.py` 和 `config.yaml`

---

## 三摄场景结构

每个"场景"包含 3 张图像（tele/main/wide），按焦距降序排列：

| sensor_id | 模组 | 焦距排序 | MCS 对齐方式 |
|-----------|------|----------|-------------|
| 0 | Tele | 最大 | 裁剪中心 + resize (置信度全1) |
| 1 | Main | 中间 | 不变 (置信度全1) |
| 2 | Wide | 最小 | 缩小 + reflect 外推 (置信度高斯衰减) |

**对齐公式**: `align_ratio = focal_main / focal_camera`

---

## 模型架构关键约束

### 可修改的
- `model.py` 中 encoder/attention 的结构细节
- `loss.py` 中损失权重、裁剪尺寸
- `train.py` 中训练循环逻辑
- `infer.py` 中推理后处理
- `config.yaml` 中所有参数

### 不可破坏的
- **MCS 10 通道格式**: 9 光谱 + 1 置信度，最后一维必须为 10
- **raw_tokens 和 mcs_tokens 共用 2D 位置编码**: 两者 `grid_size` 必须一致，否则 cross-attention 空间对应关系断裂
- **后裁剪机制**: 模型输出全图 gain_map，loss 时裁剪重叠区域。angular_loss 在全图 spatial mean 上计算，reconstruction/consistency 在裁剪区域计算
- **焦距编码 (FocalLengthEncoder)**: 即使 MCS 已空间对齐，焦距编码仍需保留作为 camera identity 条件输入
- **MCS 对齐只发生在 dataloader 层**: `align_mcs_to_fov()` 在 `dataloader.py` 中，model 接收的是已经对齐好的 mcs tensor

---

## 训练流程

```bash
python train.py                    # 默认训练
python train.py --config config.yaml  # 指定配置文件
```

### 断点续训
在 config.yaml 中设置：
```yaml
checkpoint:
  resume_from: "./checkpoints/latest.pth"
```
自动加载 `model_state_dict` + `optimizer_state_dict`，从 `epoch+1` 恢复训练。

### 输出产物
- `checkpoints/best_model.pth` — 总 loss 最小时（手动选择用于推理的权重）
- `checkpoints/latest.pth` — 每 epoch 更新
- `checkpoints/model_epoch_XXX_loss_X.XXXX.pth` — 周期性保存（间隔在 config 中设置）
- `debug_outputs/train/epoch_XXX.png` — input|pred|gt 三列三行对比
- `debug_outputs/train/epoch_XXX_mcs_alignment.png` — MCS 空间对齐可视化

---

## 推理流程

```bash
python infer.py
# 或指定文件:
python infer.py --npz data/image_processed/IMG*.npz --mcs data/Mcsnpy/IMG*.npy
```

推理时 MCS 对齐行为必须与训练时一致（使用相同 `align_mcs_to_fov` 逻辑）。

---

## 扩展指南

### 添加新模块
1. 在 `model.py` 中添加 `nn.Module` 子类
2. 如果新增配置项，同步修改 `config.yaml` 和 `config_loader.py`
3. 如果影响训练流程，修改 `train.py` 相应部分
4. 确保 MCS 10 通道格式和 cross-attention 空间对应不被破坏

### 添加新的预处理步骤
1. 在 `data/` 下创建独立脚本
2. 更新 README.md 中的预处理流水线
3. 遵循"每个预处理步骤独立可重复"原则

### 调试可视化
- MCS 对齐验证: 查看 `debug_outputs/train/epoch_*_mcs_alignment.png`
- 训练收敛: 查看 `debug_outputs/train/epoch_*.png`
- MCS 各通道: 运行 `data/visualize_mcs.py`
