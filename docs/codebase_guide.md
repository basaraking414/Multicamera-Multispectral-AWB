# AWB Transformer 代码库深入指南

> 最后更新: 2026-05-11
> 覆盖范围: 全部 11 个源文件 + 测试框架，每个函数/类的实现细节

---

## 目录

1. [项目全景](#1-项目全景)
2. [config.yaml — 配置中心](#2-configyaml--配置中心)
3. [config_loader.py — 配置加载器](#3-config_loaderpy--配置加载器)
4. [geometry_utils.py — 几何变换工具](#4-geometry_utilspy--几何变换工具)
5. [data/ 预处理脚本](#5-data-预处理脚本)
6. [dataloader.py — 场景数据集](#6-dataloaderpy--场景数据集)
7. [model.py — 核心模型](#7-modelpy--核心模型)
8. [loss.py — 损失函数](#8-losspy--损失函数)
9. [train.py — 训练脚本](#9-trainpy--训练脚本)
10. [test.py — 测试评估](#10-testpy--测试评估)
11. [visualization.py — 可视化工具](#11-visualizationpy--可视化工具)
12. [infer.py — 单张推理](#12-inferpy--单张推理)
13. [完整数据流走查](#13-完整数据流走查)
14. [pytest 测试框架](#14-pytest-测试框架)

---

## 1. 项目全景

### 1.1 核心问题

三摄手机（tele/长焦, main/主摄, wide/广角）拍同一个场景，不同模组的传感器对不同光照的色彩响应不同。传统 AWB 只能每个模组独立做，导致三张照片的色温不一致。

本项目的解法：利用 MCS 9 通道多光谱传感器的光谱信息，让 Transformer 学习跨模组的光照一致性。

### 1.2 端到端数据流

```
预处理阶段（训练前，一次性完成）
═══════════════════════════════════════════
DNG 原片 → 解码 → RAW 线性图像
                → 提取 CCM (EXIF 两个光照矩阵)
                → 提取焦距 (EXIF)
                → 提取 GT 白平衡增益 (白块检测)
                → 保存为 .npz

MCS .bin → 解析 → 9 通道光谱 .npy

训练阶段
════════
.npz + .npy → AWBDataset._prepare_scene()
  ├── 按焦距降序排列 (tele→main→wide)
  ├── 图像 resize → [64, 64, 3]
  ├── MCS 空间对齐 + 置信度 → [64, 64, 10]
  ├── 合成 GT 图像 (image * gt_gain)
  └── 返回 batch {image, mcs, ccm1, ccm2, focal_length, ...}
           │
           ▼
   AWBTransformer.forward()
  ├── RAW encoder → 256 tokens [B, 256, 64]
  ├── MCS encoder → 256 tokens [B, 256, 64]
  ├── PositionalEncoding2D (共享)
  ├── CCM FiLM 调制 (消除传感器色彩差异)
  ├── Sensor FiLM 调制 (用焦距标识模组)
  ├── Stage1: Self-Attention (RAW 内部交互)
  ├── Stage2: Cross-Attention (RAW 查询 MCS)
  ├── Stage3: CLS Attention (全局聚合)
  ├── CLS 调制 fused_tokens
  ├── Head → per-token gain → reshape → clamp [1e-4, 4.0]
  └── Bilinear upsample → [B, H, W, 3]
           │
           ▼
   total_loss()
  ├── crop_to_overlap (裁三摄共同区域 → 32×32)
  ├── angular_loss (逐像素角度损失)
  ├── reconstruction_loss (L1)
  ├── scene_consistency_loss (三摄色度一致)
  └── spatial_smoothness_loss (边缘感知平滑)

推理阶段
════════
单个 .npz + .npy → 同编码流程 → 模型 → gain map → 上采样到原图 → 保存
```

### 1.3 关键设计决策汇总

| 设计点 | 决策 | 原因 |
|--------|------|------|
| CCM 编码 | 只求 ccm1 逆（D65），不用 ccm2 | 平均两个光照矩阵物理无意义 |
| CCM 求逆 | `+ eps * eye(3)`, eps=1e-6 | 防止奇异矩阵产生 NaN |
| Sensor 编码 | 只用焦距正弦编码，不用 sensor_id | 焦距连续值泛化性更好 |
| 注意力结构 | Pre-LN 残差连接 | 训练稳定，支持加深网络 |
| Gain 输出 | softplus + clamp [1e-4, 4.0] | 约束合理增益范围 |
| 损失空间 | 后裁剪重叠区域，所有 loss 统一 | 避免非重叠区域的干扰 |
| 角度损失 | 逐像素计算再平均 | 保留空间变化信息 |
| 数据分组 | 焦距降序排列 → tele/main/wide | 统一 sensor_id 分配 |

---

## 2. config.yaml — 配置中心

**作用**: 项目的唯一配置入口，所有超参数和路径集中管理。

**设计**: 扁平 YAML 结构，按功能分 8 个区块，支持注释说明每个参数用途。

```
config.yaml 区块结构:
├── project          # 项目名、随机种子
├── data             # 数据路径、输入尺寸、支持格式
├── external_tools   # exiftool 路径
├── model            # 模型架构参数
├── training         # 训练超参、损失权重、调度器
├── checkpoint       # 检查点保存策略
├── debug            # 调试可视化间隔
├── test             # 测试集路径和参数
└── inference         # 推理输出配置
```

**关键数值约定**:
- `img_size: [64, 64]` — 模型输入分辨率，影响 token 数量和计算量
- `grid_size: 16` — 16×16=256 tokens，与 64×64 分辨率匹配（每 token 覆盖 4×4 像素）
- `loss_crop_size: 32` — 裁剪到 32×32 统一尺寸计算 loss
- `smoothness_weight: 0.05` — 空间平滑正则化权重，可设为 0 关闭

---

## 3. config_loader.py — 配置加载器

**作用**: 将 config.yaml 解析为结构化 Python 对象，提供类型安全的配置访问。

**架构**: 8 个 dataclass → 1 个顶层 Config dataclass → YAML 解析函数。

### 3.1 数据结构: 8 个 Config Dataclass

每个 dataclass 对应 config.yaml 的一个区块，字段有默认值保证向后兼容。

```python
@dataclass
class DataConfig:
    root_dir: str = "./data"                # 数据集根目录
    img_size: Tuple[int, int] = (64, 64)   # (H, W)
    mcs_size: Tuple[int, int] = (64, 64)
    # ... 子目录路径字段 ...
```

```python
@dataclass
class ModelConfig:
    dim: int = 64            # Transformer 维度 (必须能被 2 整除)
    num_heads: int = 4       # 注意力头数 (dim 能被 num_heads 整除)
    grid_size: int = 16      # 空间网格尺寸
    focal_embed_dim: int = 16  # 焦距编码输出维度
    predict_ccm: bool = True   # 是否启用 CCM 预测头
```

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.0001
    lr_scheduler: str = "cosine"   # "cosine" | "plateau" | "none"
    loss_weights: Dict[str, float]  # awb/reconstruction/consistency 权重
    loss_crop_size: int = 32       # 后裁剪统一尺寸
    smoothness_weight: float = 0.0 # 空间平滑权重
```

### 3.2 函数详解

#### `_resolve_path(base_dir, value) -> Optional[str]`
将 YAML 中的相对路径转为绝对路径。`None` 保持 `None`（用于可选路径如 exiftool_path、resume_from）。

#### `_parse_size(value, name) -> Tuple[int, int]`
灵活解析尺寸配置：支持 `64`（正方形）、`[64]`（正方形）、`[64, 64]`（长宽），统一返回 `(H, W)` 二元组。非法值抛出 `ValueError`。

#### `load_config(config_path="config.yaml") -> Config`
逐区块读 YAML → 构造对应 dataclass。每个字段用 `.get(key, default)` 保证缺省时的兼容性。

**实现要点**:
- 用 `base_dir = os.path.dirname(os.path.abspath(config_path))` 将相对路径转为绝对路径
- `root_dir` 等路径字段调用 `_resolve_path` 归一化
- `img_size`/`mcs_size` 调用 `_parse_size` 解析

#### `build_lr_scheduler(optimizer, cfg) -> Optional[scheduler]`
工厂函数，根据 `cfg.lr_scheduler` 字符串创建对应调度器。

**cosine 模式**:
- `T_max = min(params["T_max"], cfg.epochs)` — 约束不超过总 epoch，防止 LR 回升
- `eta_min = params["eta_min"]` — 余弦最低点 LR

**plateau 模式**: `ReduceLROnPlateau(mode="min", factor, patience)` — loss 停滞时减半

**none 模式**: 返回 `None`

### 3.3 当前审查发现

代码质量良好。唯一的注意点：`_resolve_path` 只在顶层路径字段调用，子目录字段（`image_dir`、`mcs_npy_dir` 等）保持为相对路径字符串，由 dataloader 在运行时用 `os.path.join(root_dir, ...)` 组合。这是有意的设计——子目录必须与 root_dir 拼接才有意义。

---

## 4. geometry_utils.py — 几何变换工具

**作用**: 提供所有空间几何操作：裁剪比例计算、中心裁剪、图像缩放、MCS 空间对齐。

### 4.1 函数详解

#### `focal_to_crop_ratio(focal_length, reference_focal) -> float`
计算裁剪比例 `focal_length / reference_focal`。结果裁剪到 [0.05, 1.0]。

**物理含义**: 以最大焦距模组为基准（FOV 最小），其他模组的视场比例。Tele 比例最大（≈1），Wide 最小（<1）。

#### `center_crop(image, crop_ratio) -> (cropped_image, (x0,y0,x1,y1))`
按比例从图像中心裁出子区域。`crop_h = round(h * ratio)`，最小 4 像素。返回裁剪图像和裁剪框坐标。

#### `resize_image(image, target_hw, interpolation) -> ndarray`
智能缩放：如果通道数 > 4（如 MCS 9 通道），逐通道缩放后堆叠；否则直接 cv2.resize。

**注意**: OpenCV 的 resize 期望 `(W, H)` 顺序（不是 `(H, W)`）。

#### `crop_and_resize(image, crop_ratio, target_hw, interpolation) -> (resized, crop_box)`
组合操作：先 `center_crop`，再 `resize_image`。一次性完成裁剪+缩放到目标尺寸。

#### `compute_scene_crop_plan(focal_lengths) -> List[Dict]`
为场景中每张图计算裁剪计划。以场景内最大焦距为基准，逐张算 `crop_ratio`。

```python
# 输入: [tele_focal=120, main_focal=26, wide_focal=16]
# 基准: max = 120
# 输出: [{focal=120, ref=120, ratio=1.00},    # tele: 全图
#        {focal=26,  ref=120, ratio=0.22},    # main: 裁 22%
#        {focal=16,  ref=120, ratio=0.13}]    # wide: 裁 13%
```

#### `compute_crop_boxes_from_ratios(img_h, img_w, crop_ratios) -> List[Tuple]`
给定裁剪比例，计算每张图的具体裁剪框坐标。不实际裁剪，只算坐标，供后裁剪 loss 使用。

#### `align_mcs_to_fov(mcs, align_ratio, target_hw) -> (aligned_mcs, confidence)`

**这是项目最核心的数据预处理函数。** 将 MCS 数据从 Main 模组视场空间对齐到各模组视场。

**为什么需要**: MCS 传感器物理 FOV 固定 ≈ Main 模组。Tele 视角更窄、Wide 视角更宽，RAW 图像的像素和 MCS 数据的像素不是一一对应的。不处理会导致用错误的 MCS 值指导当前像素的 AWB。

**实现逻辑**（三种情况）:

1. `align_ratio ≈ 1.0` (Main): 直接 resize → 置信度全 1.0
2. `align_ratio < 1.0` (Tele): 中心裁剪 + resize 放大 → 数据全真实 → 置信度全 1.0
3. `align_ratio > 1.0` (Wide): 缩小 + reflect padding 外推 → 置信度从中心 1.0 高斯衰减到边缘 0.0

**置信度图的计算**:
```python
confidence = zeros(th, tw)
confidence[valid_region] = 1.0  # 中心真实数据区域
sigma = max_pad * 0.4            # 过渡带宽
confidence = GaussianBlur(confidence, sigmaX=sigma)
```

Wide 的 MCS 边缘是外推/猜测值，置信度告诉模型："这里我不确定，别全信"。

### 4.2 CCM 矩阵工具

#### `safe_inv_ccm(ccm, eps_factor=10.0) -> Tensor`
数值稳定的CCM矩阵求逆。通过 `eps * eye(3)` 正则化避免奇异矩阵求逆导致的 NaN/Inf。

**实现**:
```python
eps = max(1e-6, torch.finfo(ccm.dtype).eps * eps_factor)
eye = torch.eye(3, device=ccm.device, dtype=ccm.dtype).unsqueeze(0)
return torch.linalg.inv(ccm.float() + eps * eye)
```

**使用位置**: model.py (CCMEncoder)、loss.py (build_srgb_gt)、train.py、test.py

### 4.3 可视化辅助工具

本章开头的 `auto_expose` 和 `ensure_float01` 原在 `gt_utils.py` 中，精简后移入此处供训练/测试/推理脚本共用。

#### `ensure_float01(image) -> ndarray`
将图像归一化到 [0,1] 范围。若 max > 1.0（如 16-bit RAW），除以 max 归一化。用于 debug 可视化前的预处理。

#### `auto_expose(image, percentile=99.5) -> ndarray`
自动曝光显示：按 P99.5 分位数做白归一化，再施加 gamma 2.2 暗部补偿。**仅用于可视化，不影响训练数据**。

---

## 5. data/ 预处理脚本

**作用**: 6 个独立脚本组成预处理管线，完成从 DNG 原片到模型可读的 .npz 的一站式转换。

### 5.1 `demo_ImageMatchBindata.py` — 数据匹配

**作用**: 将 dump 目录中的 Camera（JPG/DNG）和 MultispectralSensor（MCS .bin）文件按时间戳或文件名匹配对齐。

**输入/输出**:
```
dump/Camera/*.{jpg,dng} + dump/MultispectralSensor/*.bin
  → data/image/ + data/image_dng/ + data/McsBin/
```

**为什么需要**: 实际采集中，同一场景的 RGB 图像和 MCS 数据来自不同传感器，文件名可能不一致。需要对齐确保同一场景的三摄数据和 MCS 数据一一对应。

### 5.2 `demo_readMcsRaw.py` — MCS 解析

**作用**: 将 MCS .bin 二进制文件解析为 NumPy .npy。

**输入/输出**: `data/McsBin/*.bin → data/Mcsnpy/*.npy (9 通道光谱)`

**9 通道光谱波段**: 从近红外到近紫外的离散光谱采样，具体波段范围取决于 MCS 传感器型号。

### 5.3 `dng_decoder.py` — 解耦 DNG 解码器

**作用**: **只做** RAW 解码和 EXIF 元数据提取，不包含任何白平衡/白块 GT 计算。输出干净的 NPZ 文件供 `gt_extractor.py` 后续独立添加 GT。

#### 辅助函数

##### `_parse_float(value, default=0.0) -> float32`
从 EXIF 文本解析浮点值。处理分数格式（如 `"50/1"` → 50.0）和带 `mm` 后缀的值。如果 EXIF 字段缺失，返回默认值。

##### `_parse_matrix(value) -> ndarray [3, 3]`
从 EXIF 字符串解析 3×3 矩阵。支持空格分隔的 9 个浮点数。如果字段缺失，返回单位矩阵（后备方案）。

#### 核心函数

##### `decode_dng(raw_file_path, size=256) -> Dict`

**DNG 解码 + 元数据提取的核心实现。**

```
1. 用 exiftool 读取所有 EXIF 元数据
2. 用 rawpy 解码 DNG:
   - AAHD 去马赛克
   - 16-bit 输出、无自动亮度/缩放、无相机白平衡
   - raw 色彩空间（不经任何色彩转换）
   - 手动黑电平/白点校准
3. 归一化: image / white_level → [0, 1]
4. resize 到统一 size×size (保留原图在 image_full 中)
5. 提取元数据:
   - focal_length, focal_length_35mm
   - xyz2camera_rgb1 (D65) + xyz2camera_rgb2 (TL84)
   - file_name, raw_resolution, processed_resolution
```

**返回值字典**:
```python
{
    "image":         [size, size, 3]      # 缩放的 RAW
    "image_full":    [H_orig, W_orig, 3]  # 原始分辨率 RAW
    "focal_length":  float32              # 物理焦距 (mm)
    "focal_length_35mm": float32          # 等效 35mm 焦距
    "xyz2camera_rgb1": [3, 3]             # D65 CCM
    "xyz2camera_rgb2": [3, 3]             # TL84 CCM
    "file_name":     str                  # 源文件名
    "raw_resolution": [H, W]              # 原始分辨率
    "processed_resolution": [H, W]        # 处理后分辨率
    "crop_strategy": "none"               # 占位
}
```

##### `batch_decode(input_dir, output_dir, size=256) -> None`
遍历 `input_dir` 中所有 `.dng`/`.nef`/`.arw` 文件，逐张调用 `decode_dng`，将结果保存到 `output_dir/*.npz`。

### 5.4 `gt_extractor.py` — 独立 GT 提取器

**作用**: 在解码后的 RAW NPZ 上独立运行，添加白平衡 GT 字段。支持两种 GT 获取方法：色卡自动检测和启发式白块。

**设计**: 完全自包含（不依赖项目中任何其他 Python 文件），"从 gt_utils 移植的独立版本"。

#### 方法 A: X-Rite ColorChecker 检测（推荐）

##### `detect_colorchecker_auto(image) -> Optional[(gray_patches, centers)]`
在全分辨率 RAW 图像中自动定位 X-Rite ColorChecker Classic 色卡，提取灰阶块。

**实现步骤**:
```
1. Canny 边缘检测 → 找 4 边形轮廓
2. 验证宽高比 ≈ 1.543（216mm / 140mm）
3. 透视变换矫正视角 → 4×6 网格分割
4. 取第 4 行（灰阶行）6 个灰块 → 6 个 RGB 均值
5. 将网格中心映射回原图坐标
```

##### `detect_colorchecker_roi(image, roi) -> (gray_patches, centers)`
用户手动指定色卡区域，跳过自动检测，直接在 ROI 内做 4×6 网格分割。

##### `compute_gain_from_gray_patches(gray_patches) -> ndarray`
从 6 个灰阶块（从白到黑）计算增益。策略：排除最亮（可能过曝）和最暗（低信噪比）的块，用中间 4 块平均。

#### 方法 B: 启发式白块检测（备用）

白块检测算法与 4.2 节 `ensure_float01` 等一同移入 `geometry_utils.py`，此处 `gt_extractor.py` 有独立副本以保持自包含。

#### 统一接口

##### `extract_awb_gt(image, method="colorchecker", roi=None) -> Dict`
```python
# 自动检测色卡 → colorchecker_auto
# 指定 ROI → colorchecker_roi
# 自动检测失败 → white_patch_fallback (回退)
# 指定 white_patch → white_patch
返回: {"awb_gt_gain": [3], "white_patch_rgb": [3],
       "white_patch_box": [4], "white_patch_score": [3],
       "gt_method": str}
```

#### 批处理接口

##### `process_raw_npz(npz_path, output_dir, debug_dir, method, roi, size) -> None`
单文件处理：读取 decoder 输出的 raw NPZ → 调用 `extract_awb_gt` → 合并 GT 字段 → 保存为完整 NPZ → 生成调试可视化。

##### `batch_process(input_dir, output_dir, method, roi, size) -> None`
批量处理目录下所有 raw NPZ 文件。

### 5.5 `visualize_mcs.py` — MCS 可视化

**作用**: 将 9 通道 MCS 数据可视化为 RGB 伪彩色图像，用于检查 MCS 数据质量和空间对齐效果。

### 5.6 管线使用方式

```bash
# 完整数据预处理管线:
python demo_ImageMatchBindata.py      # Step 1: 匹配
python demo_readMcsRaw.py             # Step 2: MCS 解析

# 解耦方案 (推荐):
python dng_decoder.py                 # Step 3A: 仅解码 → image_raw/
python gt_extractor.py --method colorchecker  # Step 3B: 加 GT → image_processed/

# 查看 MCS:
python visualize_mcs.py
```

---

## 6. dataloader.py — 场景数据集

**作用**: PyTorch Dataset，按"场景"组织三摄数据，完成 MCS 对齐、数据验证和 batch 构造。

### 6.1 类: `AWBDataset(Dataset)`

#### `__init__(self, root_dir, img_size, mcs_size)`

**执行流程**:

1. 列出 `image_processed/*.npz` 和 `Mcsnpy/*.npy`，排序后验证数量一致性
2. 按位置每 3 个一组形成场景
3. 每组内读取 EXIF 焦距，按焦距降序排列（tele → main → wide），分配 sensor_id=0,1,2
4. 为每个样本加载所有字段并保存到 `self.scenes` 列表中

**数据结构**:
```python
self.scenes = [
    [  # scene 0
        {"image": [H,W,3], "focal_length": 120.0, "ccm1": [3,3], ...},  # tele
        {"image": [H,W,3], "focal_length": 26.0,  "ccm1": [3,3], ...},  # main
        {"image": [H,W,3], "focal_length": 16.0,  "ccm1": [3,3], ...},  # wide
    ],
    [  # scene 1
        ...
    ],
]
```

**已知问题**（未修复，等数据收集完成后处理）: 场景分组基于文件名字母排序 + 位置切片，如果某个场景文件缺失会导致后续场景全部错位。后续需改为按文件名中的 scene_id 显式分组。

#### `__len__() -> int`
返回场景数量。

#### `_prepare_scene(scene_samples) -> Dict[str, Tensor]`

**这是训练前每次 batch 构造的核心函数。**

执行步骤:

1. **计算裁剪计划**: `compute_scene_crop_plan(focal_lengths)` — 以最大焦距为基准，每张图算 crop_ratio

2. **确定 MCS 对齐基准焦距**: 取 index=1（main）的焦距作为 ref_focal

3. **逐传感器处理**（for sample in scene_samples）:
   - 数据验证：检查 image 和 MCS 是否有 NaN/Inf
   - 图像 resize：`resize_image(image, img_size)` — 仅缩放不裁剪
   - MCS 空间对齐：`align_mcs_to_fov(mcs, align_ratio, mcs_size)` — 关键步骤
   - 拼接置信度通道：`concat([mcs_aligned, confidence], axis=-1)` → `[H, W, 10]`
   - 合成 GT 图像：`gt_image = clip(image * gain, 0, 1)`

4. **构造 batch 字典**（numpy → torch）:
```python
batch = {
    "image":       torch [S, H, W, 3],       # 原始 RAW
    "gt_image":    torch [S, H, W, 3],       # GT 校正后 (uniform gain)
    "mcs":         torch [S, H, W, 10],      # MCS 10 通道 (9光谱 + 1置信)
    "awb_gt_gain": torch [S, 3],             # GT 增益值
    "crop_ratio":  torch [S],                # 后裁剪比例
    "ccm1":        torch [S, 3, 3],          # D65 CCM
    "ccm2":        torch [S, 3, 3],          # 第二光照 CCM
    "focal_length": torch [S],               # 焦距 (mm)
    "sensor_id":   torch [S],                # 0=tele, 1=main, 2=wide
    "scene_id":    torch [S],                # 场景编号
}
```

**注意**: `gt_image = image * gt_gain` 使用了均匀增益，这意味着当前 GT 假设场景光照是全局均匀的。这是后续改进的方向。

#### `__getitem__(idx) -> Dict`
直接调用 `_prepare_scene(self.scenes[idx])`。

---

## 7. model.py — 核心模型

**作用**: 实现完整的 AWBTransformer 架构，包含位置编码、CCM 编码、Sensor 编码、注意力 Pipeline 和输出头。

### 7.1 类: `PositionalEncoding2D(nn.Module)`

**作用**: 为 2D 空间 token 提供可区分位置信息的正弦位置编码。

**为什么需要**: Transformer 的 Self-Attention 是位置无关的（permutation invariant）。不加 PE，模型不知道哪个 token 对应图像的哪个位置。

**实现细节**:

1. 生成 Y 坐标和 X 坐标序列
2. 将 dim 均分给 y 和 x 方向（各 dim/2）
3. 分别生成 y 方向和 x 方向的正弦/余弦编码
4. `pe[:, 0::2] = sin(pos * div)`, `pe[:, 1::2] = cos(pos * div)` — 交错存放
5. 拼接 y 和 x 部分的 PE，用 `register_buffer` 注册（不参与梯度，但随模型保存）

**数值保护**: 确保 dim_y 和 dim_x 均为偶数（避免切片宽度不匹配）。如有余数用零填充。

**forward**: `x + pe[:, :N, :]` — 简单加法注入位置信息。

**raw 和 mcs 共用同一个 PE 的原因**: 两者 grid_size 一致（都是 16×16），token 的空间位置完全对应。

### 7.2 类: `CCMEncoder(nn.Module)`

**作用**: 将两个 3×3 Color Correction Matrix 编码为 FiLM 调制参数，消除不同传感器的色彩响应差异。

**物理背景**: 每个摄像头有 xyz2camera_rgb 矩阵（EXIF 中有 D65 和 TL84 两组），定义了标准 XYZ 颜色空间到这个摄像头 RGB 的转换。逆矩阵 = cameraRGB → XYZ。不同传感器的 CCM 不同，表示它们"看"颜色的方式不同。

**实现**:
```
ccm1 [B,3,3] → inv(+eps*eye) → reshape [B,9] ─┐
ccm2 [B,3,3] → inv(+eps*eye) → reshape [B,9] ─┤→ cat [B,18] → MLP(18→dim→dim) → scale/bias head
```

**数值保护** (`eps = max(1e-6, finfo.eps * 10)`): 对角加微小扰动防止近奇异矩阵产生 NaN。`1e-6` 在 float32 下足够显著又几乎不改变物理含义。

### 7.3 类: `FocalLengthEncoder(nn.Module)`

**作用**: 将连续焦距值映射到高维特征空间，作为 SensorEncoder 的输入。

**设计**: 使用正弦位置编码的变体——不同频率的正弦波捕获不同尺度的特征。
```python
focal_range = 2^0 ~ 2^5 = 1 ~ 32  # 频率范围
norm = focal / 200  # 归一化到 [0,1]
enc = sin(norm * freq), cos(norm * freq)  # dim 维编码
→ MLP(dim → dim → dim)  # 非线性变换
```

**为什么不用 sensor_id embedding**: 焦距是连续值，可以泛化到训练时没见过的焦距值。sensor_id (0,1,2) 只覆盖 3 个离散值，对新手机无用。

### 7.4 类: `SensorEncoder(nn.Module)`

**作用**: 将焦距编码转为 FiLM scale/bias，调制 RAW tokens 以注入模组身份信息。

**注意**: 即使 MCS 在 dataloader 层已做空间对齐，仍需要焦距编码。因为 MCS 对齐是几何层面的，而焦距编码提供的是 camera identity——模型需要知道"当前处理的是哪个模组的信号"以做出正确判断。

**实现**:
```
focal_length [B] → FocalLengthEncoder → [B, focal_embed_dim]
  → fusion MLP(focal_embed_dim → dim → dim) → 2 heads → scale, bias [B, dim]
```

### 7.5 类: `CCMHead(nn.Module)`

**作用**: 从 CLS token 预测 3×3 CCM delta，实现端到端的相机色彩空间标定。

```
CLS token [B, 1, dim] → MLP(dim → dim → 9) → reshape [B, 3, 3]
```

**最终 CCM** = `inverse(xyz2camera_rgb1) + delta`，仅用 ccm1（D65）作基准。

### 7.6 类: `AWBTransformer(nn.Module)` — 主模型

#### `__init__(dim=64, num_heads=4, grid_size=16, ...)`

构造顺序:
```
raw_encoder (Conv2d 3→64→64)          # 3ch RAW → 64ch feature
mcs_encoder (Conv2d 10→64→64)         # 10ch MCS → 64ch feature  
pool (AdaptiveAvgPool2d → 16×16)      # 空间降采样 → 256 tokens
pos_enc (PositionalEncoding2D)        # 2D 位置编码 (可选)
cls_token (learnable [1, 1, 64])      # CLS token (全局信息聚合)
ccm_encoder (CCMEncoder)              # CCM → FiLM
sensor_encoder (SensorEncoder)        # Focal → FiLM
raw_self_attn (MultiheadAttention)     # Stage1: RAW 自注意力
cross_attn (MultiheadAttention)       # Stage2: RAW→MCS 交叉注意力
cls_attn (MultiheadAttention)         # Stage3: CLS 聚合注意力
5× LayerNorm                          # Pre-LN 归一化层
cls_scale/bias (Linear)               # CLS 调制融合 token
head (MLP 64→64→3)                    # 逐 token 增益输出
ccm_head (CCMHead, 可选)               # CCM delta 预测
```

**Pre-LN 设计理由**: 标准 Transformer 的最佳实践。在注意力之前做 LayerNorm，残差连接加在之后：
```
output = input + Attention(LayerNorm(input))
```
优点: 梯度可以通过残差路径直接回传，训练更稳定。

#### `forward(raw, mcs, ccm1, ccm2, focal_length) -> (gain_map, ccm_delta)`

**完整前向步骤**（按实际代码顺序）:

```
Step 1: 编码
  raw [B,H,W,3] → permute(B,C,H,W) → raw_encoder → pool → tokens [B,256,64]
  mcs [B,H,W,10] → permute(B,10,H,W) → mcs_encoder → pool → tokens [B,256,64]
  
Step 2: 位置编码
  raw_tokens += PE, mcs_tokens += PE  (共享同一个 PE)

Step 3: 双重 FiLM 调制
  calibrated = raw_tokens * scale_ccm + bias_ccm  # CCM: 消除传感器色彩差异
  calibrated *= scale_sensor + bias_sensor         # Sensor: 焦距标识模组

Step 4: 三级注意力 Pipeline (全部 Pre-LN + 残差)
  Stage1: refined = calibrated + SelfAttn(norm(calibrated))
  Stage2: fused    = refined + CrossAttn(norm_q(refined), norm_kv(mcs))
  Stage3: cls_out  = cls_token + CLSAttn(norm_q(cls), norm_kv([fused; mcs]))
  
  # 注意: all_tokens = cat([fused_tokens, mcs_tokens]) 让 CLS 同时看到融合特征和原始 MCS

Step 5: CCM 预测 (可选)
  ccm_delta = CCMHead(cls_out)  # [B, 3, 3]

Step 6: CLS 调制
  fused *= sigmoid(cls_scale(cls_out)) + cls_bias(cls_out)
  # sigmoid 将 scale 约束到 (0,1)，防止过度放大

Step 7: 输出 gain
  gain = Head(fused_tokens)          # [B, 256, 3]
  gain = reshape → [B, 16, 16, 3]
  gain = softplus(gain).clamp(1e-4, 4.0)  # 正值有界约束
  gain = upsample(bilinear, 16×16 → H×W)  # [B, H, W, 3]
```

**逐 token 增量上采样**: `F.interpolate(mode='bilinear')` 做平滑上采样，比最近邻插值能保持增益空间连续性。

**softplus 选择**: `softplus(x) = log(1+e^x)` 确保输出始终大于 0（增益必须为正），且梯度平滑。

---

## 8. loss.py — 损失函数

**作用**: 定义所有损失项并聚合，实现后裁剪重叠区域机制。

### 8.1 函数详解

#### `crop_to_overlap(images, crop_ratios, target_size, min_size=4) -> Tensor`

**三摄重叠区域后裁剪的核心实现。**

```
输入: images [B, S, H, W, C]  # B=场景数, S=3 摄
逐场景、逐传感器:
  crop_size = max(4, round(H * ratio))  # ratio 来自焦距比
  center_crop → [crop_h, crop_w, C]
  bilinear resize → [target_size, target_size, C]  # 统一到相同尺寸
输出: [B, S, target_size, target_size, C]
```

**为什么需要**: 三摄的视场大小不同（Tele 窄、Wide 宽），直接在原始图像上算 loss 会出现"这个像素在 Tele 中存在但 Wide 中不存在"的情况。裁剪到共同区域后，所有像素在所有摄像头中都有对应。

#### `srgb_gamma(linear, eps=1e-8) -> Tensor`

标准 sRGB gamma 校正分段函数:
- `linear ≤ 0.0031308`: `12.92 * linear` (暗部线性)
- `linear > 0.0031308`: `1.055 * linear^(1/2.4) - 0.055` (亮部幂律)

#### `angular_loss(pred_gain, gt_gain, eps=1e-6) -> Tensor`

**逐像素角度损失**（修复后版本）:

```python
pred_vec = normalize(pred_gain [B*S, H, W, 3])  # 通道维归一化
gt_vec   = normalize(gt_gain   [B*S, 3])        # 广播到空间维
cosine = (pred_vec * gt_vec).sum(dim=-1)         # [B*S, H, W] 逐像素纯余弦
loss = (1 - cosine).mean()                       # 空间 + batch 平均
```

**为什么用 1-cos 而不是 acos**: `arccos` 在 cos→±1 时梯度趋近无穷大，`1-cos` 梯度有界 `[-1, 1]`，训练更稳定。

**逐像素 vs 空间平均的区别**:
- 旧版（空间平均后算角度）: 如果左上角 gain=[1.2,1.0,0.8]，右下角 gain=[0.8,1.0,1.2]，空间平均=[1.0,1.0,1.0]，与 GT=[1.0,1.0,1.0] 完美匹配 → 损失 = 0，但两个像素都没做对
- 新版（逐像素算角度）: 每个像素独立与 GT 比较 → 损失 > 0，能捕捉到空间变化信号

#### `reconstruction_loss(pred_image, gt_image) -> Tensor`
L1 loss。比 L2 对异常值更鲁棒，更适合图像任务。

#### `scene_consistency_loss(pred_image, eps=1e-6) -> Tensor`

**三摄色度一致性约束**（无监督信号，不需要 GT）:

```python
scene_mean = pred_image.mean(spatial_axes)        # [B, S, 3] 每摄全局平均色
chroma = scene_mean / scene_mean.sum(dim=-1)      # 归一化为色度 (R+G+B=1)
center = chroma.mean(dim=1)                        # [B, 3] 三摄色度中心
loss = |chroma - center|.mean()                    # 每摄偏离中心的平均距离
```

**物理意义**: 同一个场景不同摄像头拍出来的"正确白平衡"结果，色度应该一致。如果 Tele 偏蓝、Wide 偏黄 → 损失高 → 模型学习调整。

#### `spatial_smoothness_loss(gain_map, raw_image, edge_weight=10.0) -> Tensor`

**边缘感知空间平滑**（新增）:

```python
# 1. 计算 gain 的垂直和水平梯度
gain_grad_y = gain[y+1] - gain[y]
gain_grad_x = gain[x+1] - gain[x]

# 2. 计算 raw 灰度图的梯度作为边缘感知权重
raw_grad_y = raw_gray[y+1] - raw_gray[y]
weight_y = exp(-10 * |raw_grad_y|)  # 边缘处 → 权重 → 0

# 3. 损失 = 加权 L1 梯度
loss = (weight * |gain_grad|).mean()
```

**为什么需要**: gain map 作为空间变化量，可能产生不自然的突变。通过平滑约束让 gain 空间连续变化，同时用 raw 图像的边缘来控制"哪里可以不平滑"。

**edge_weight=10.0**: 控制边缘处惩罚衰减速度。太大 → 边缘完全被忽略；太小 → 平滑过度。

#### `build_srgb_gt(raw, gt_gain, ccm1, ccm2) -> Tensor`

**合成 sRGB Ground Truth**（修复后版本）:

```
raw [B,H,W,3] * gt_gain [B,3]       → cameraRGB (AWB 校正后)
cameraRGB * inv(ccm1 + eps*eye)      → XYZ (标准色彩空间)
XYZ * xyz_to_srgb 标准矩阵           → linear sRGB
linear sRGB → gamma → sRGB [B,H,W,3]
```

**关键修复**: 只使用 ccm1（D65），不再平均 ccm1 和 ccm2。平均两个不同光照的矩阵在物理上没有意义——得到的矩阵不对应任何真实光照。

#### `total_loss(...) -> Dict[str, Tensor]`

**损失聚合函数**，所有损失项的编排中心。

**参数**:
```python
pred_gain:        [B*S, H, W, 3]  预测增益
gt_gain:          [B*S, 3]        GT 增益
pred_image:       [B*S, H, W, 3]  预测图像 (gain*raw)
gt_image:         [B*S, H, W, 3]  GT 图像
scene_pred_image: [B, S, H, W, 3] 按场景分组
weights:          {"awb":1.0, "reconstruction":10.0, "consistency":2.0}
crop_ratios:      [B, S] 或 None
raw_image:        [B*S, H, W, 3]  (smoothness 的边缘参考)
smoothness_weight: float
pred_srgb:        [B*S, H, W, 3]  预测 sRGB (可选)
gt_srgb:          [B*S, H, W, 3]  GT sRGB (可选)
```

**执行分支**:

1. **crop_ratios 不为 None** (启用后裁剪):
   - pred_image, gt_image, pred_gain 全部 reshape → crop_to_overlap → 统一变为 [B*S, t, t, 3]
   - 所有 loss 在裁剪后的统一尺寸上计算

2. **crop_ratios 为 None**:
   - 直接在全图上计算

3. **smoothness** (可选):
   - 同样遵循裁剪规则——当启用裁剪时在裁剪区域上计算
   - 使用 `result["total"]` 累加（不是覆盖），确保与 sRGB loss 共存

4. **sRGB** (可选):
   - 同样累加到 `result["total"]`

**返回值**: `{"awb": ..., "reconstruction": ..., "consistency": ..., "total": ..., "smoothness": ...(可选), "srgb": ...(可选)}`

---

## 9. train.py — 训练脚本

**作用**: 完整的训练循环，包含数据加载、模型训练、损失计算、可视化、检查点保存。

### 9.1 导入依赖

```python
from visualization import save_debug_scene, save_mcs_alignment_debug
```

可视化函数已提取到 `visualization.py` 模块，train.py 和 test.py 共用。

### 9.2 函数详解

#### `train(config_path="config.yaml")`

**完整训练循环**:

```
初始化阶段:
├── load_config → 结构化配置
├── set_all_seeds → 可复现
├── AWBDataset + DataLoader → 数据流
├── AWBTransformer → 模型 (所有参数在 GPU)
├── Adam optimizer (lr=0.0001)
├── scheduler (cosine/plateau/none)
└── 断点续训 (如果 resume_from 存在)

训练循环 (for epoch in 0..99):
├── for batch in loader:
│   ├── 展平场景维度: [B, S, H, W, C] → [B*S, H, W, C]
│   ├── model.forward() → pred_gain, ccm_delta
│   ├── pred_image = pred_gain * flat_image
│   ├── 可选 sRGB 路径 (构建 pred_srgb)
│   ├── total_loss() → {awb, rec, cons, total, ...}
│   ├── backward() + clip_grad_norm(max=5.0) + step()
│   └── 累计 epoch 指标
│
├── scheduler.step() (cosine/plateau)
├── 打印 epoch 日志
└── 保存检查点 (latest/best/periodic)
```

**sRGB 路径细节**:
```python
# 只在 predict_ccm=True 且 srgb_weight > 0 时启用
gt_srgb = build_srgb_gt(raw, gt_gain, ccm1, ccm2)  # 与 loss.py 一致的 D65 基准

# predicted sRGB:
effective_ccm = inv(ccm1 + eps*eye) + ccm_delta    # 只用 ccm1 做基准
xyz = bmm(corrected_camera_rgb, effective_ccm.T)   # cameraRGB → XYZ  
linear_srgb = bmm(xyz, xyz_to_srgb.T)              # XYZ → linear sRGB
pred_srgb = srgb_gamma(linear_srgb)                 # gamma 校正
```

**检查点内容**:
```python
checkpoint_state = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": epoch_total_loss,
    "config": {  # 保存构造参数，确保 test/infer 能重建相同架构
        "dim": 64, "grid_size": 16, "num_heads": 4,
        "use_positional_encoding": True,
        "focal_embed_dim": 16, "predict_ccm": True,
    },
}
```

**加载策略**:
- `latest.pth` → 每 epoch 覆盖
- `best_model.pth` → 仅 loss 更优时
- `model_epoch_XXX_loss_X.XXXX.pth` → 每 10 epoch
- `load_state_dict(strict=False)` → 兼容旧检查点（新 LayerNorm 参数跳过）

---

## 10. eval.py — 测试评估

**作用**: 在测试集上评估模型，计算与训练一致的 loss 指标，生成可视化对比。

### 10.1 导入依赖

```python
from visualization import save_debug_scene, save_mcs_alignment_debug
```

可视化函数已提取到 `visualization.py` 模块，train.py 和 eval.py 共用。

### 10.2 函数详解

#### `test(config_path="config.yaml")`

**核心逻辑**（与 train 的差别）:

1. 无梯度: `with torch.no_grad()` 包裹所有前向操作
2. 无优化器、无 scheduler、无 checkpoint 保存
3. 使用 `cfg.test.root_dir` 加载测试集
4. 从 checkpoint 的 `"config"` 字段恢复模型参数
5. 使用 `strict=False` 加载旧检查点
6. **Loss 计算与训练完全一致**: 同样的 `total_loss` 调用、同样的权重、同样的裁剪参数

**输出文件**:
```
test_outputs/
├── summary.txt         # 格式: Metric + Value 表格
├── per_scene.txt       # 格式: Scene 编号 + 每个 loss 值
└── debug/
    ├── scene_0000.png                 # input|pred|gt 三行对比
    └── scene_0000_mcs_alignment.png   # MCS 对齐可视化
```

---

## 11. visualization.py — 可视化工具

**作用**: 提供训练和测试共用的 debug 可视化函数，消除 train.py 和 test.py 之间的代码重复。

### 11.1 函数详解

#### `save_debug_scene(save_dir, identifier, input_image, pred_image, gt_image, prefix="scene")`

将模型预测结果可视化为对比图。

**参数**:
- `save_dir`: 保存目录
- `identifier`: 场景ID或epoch编号
- `input_image [S, H, W, 3]`: RAW 输入
- `pred_image [S, H, W, 3]`: 模型预测
- `gt_image [S, H, W, 3]`: Ground Truth
- `prefix`: 文件名前缀（"scene" 或 "epoch"）

**布局**: 3 行 (tele/main/wide) × 3 列 (input/pred/gt)

**文件名格式**: `{prefix}_{identifier:04d}.png`

#### `save_mcs_alignment_debug(save_dir, identifier, batch, prefix="scene")`

可视化 MCS 与 RAW 的空间对齐情况。

**3 列内容**:
1. RAW 图像（自动曝光）
2. MCS 前 3 通道伪彩色（逐通道归一化）
3. 置信度热力图叠加（红色 = 低置信外推区域）

**用途**: 验证 `align_mcs_to_fov` 是否产生了合理的结果——Tele 应该全红（全置信）、Wide 应该中心蓝边缘红（中心可信边缘外推）。

**文件名格式**: `{prefix}_{identifier:04d}_mcs_alignment.png`

---

## 12. infer.py — 单张推理

**作用**: 使用训练好的模型对单张或多张图像进行 AWB 校正推理。

### 11.1 函数详解

#### `load_model(cfg, device) -> AWBTransformer`

从 checkpoint 加载模型权重。复用 checkpoint 中保存的 `config` 字段重建模型架构。

```python
ckpt = torch.load(checkpoint_path)
model_cfg = ckpt["config"]  # {"dim":64, "grid_size":16, ...}
model = AWBTransformer(**model_cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

#### `load_single_sample(npz_path, npy_path, img_size, mcs_size, sensor_id=1, ref_focal=None)`

加载单个 .npz + .npy 样本，返回原始图像 + 模型输入张量。

**处理流程**:
1. 加载 npz 元数据（EXIF 字段、CCM、焦距）和图像数据
2. 优先加载 `image_full`（原始尺寸），回退到 `image`（已 resize）
3. 缩放图像到 `img_size` 供模型推理
4. MCS 空间对齐（如果提供了 ref_focal，用与训练相同的方式）
5. 拼接置信度通道 → 10 通道 MCS

#### `infer_one(model, image_t, mcs_t, ccm1_t, ccm2_t, device, original_size, focal_length)`

单张推理核心函数:

```
image_t [1, H_small, W_small, 3] → model → gain [1, H_small, W_small, 3]
  → permute → bilinear upsample → 还原到 original_size → numpy [H_orig, W_orig, 3]
```

**上采样到原图**: 模型在 64×64 上推理（速度快），通过 `F.interpolate` 将 gain map 上采样到原始图像尺寸（如 4032×3024）。

#### `save_results(output_dir, base_name, image, gain_map, cfg)`

保存推理结果: gain map 可视化 + 校正后图像 + 输入图 + 对比拼接。

**增益图可视化**: 归一化到 [0,1] 后保存为彩色图像。
**校正图像**: `image * gain_map` → clip [0,1] → 自动曝光 → 保存。

#### `main()`

支持命令行参数:
```bash
python infer.py --config config.yaml                              # 默认配置
python infer.py --npz *.npz --mcs *.npy --ref-focal 26.0        # 指定文件 + 主摄焦距
```

如果没有指定文件，自动从 `data.image_processed_dir` 和 `data.mcs_npy_dir` 加载。

---

## 13. 完整数据流走查

### 从一张 DNG 原片到最终 AWB 校正输出

```
┌─────────────────────────────────────────────────────────────┐
│  预处理阶段（训练前 1 次）                                     │
│                                                             │
│  DNG 文件                                                    │
│    ├── rawpy 解码 → 线性 16-bit RAW 图像                      │
│    ├── exiftool → 提取 xyz2camera_rgb1 (D65 CCM)             │
│    ├── exiftool → 提取 xyz2camera_rgb2 (TL84 CCM)            │
│    ├── exiftool → 提取 FocalLengthIn35mmFormat                │
│    └── gt_extractor.extract_awb_gt() → GT 白平衡增益          │
│          保存为 .npz: {image, ccm1, ccm2, focal, awb_gt_gain} │
│                                                             │
│  MCS .bin → 解析 → 9 通道光谱 .npy                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  训练时每次 __getitem__                                       │
│                                                             │
│  3 个 .npz + 3 个 .npy（同一场景）                             │
│    │                                                        │
│    ├── 按焦距降序排序 → [tele(120mm), main(26mm), wide(16mm)]  │
│    │                                                        │
│    ├── 每张图:                                               │
│    │   image → resize(64,64) → [64, 64, 3]                   │
│    │   mcs → align_ratio=26/focal → align_mcs_to_fov()       │
│    │       → [64, 64, 9] + confidence[64, 64, 1]             │
│    │       → concat → [64, 64, 10]                            │
│    │   gt_image = image * gt_gain (均匀)                      │
│    │                                                        │
│    └── batch: {image[3,64,64,3], mcs[3,64,64,10], ccm[3,3,3], │
│                focal[3], gain[3,3], crop_ratio[3], ...}       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  AWBTransformer.forward()                                    │
│                                                             │
│  输入展平: [3, H, W, C] (视为 B=3)                            │
│                                                             │
│  RAW:  permute→encoder(3→64)→pool(16×16)→tokens[3,256,64]    │
│  MCS:  permute→encoder(10→64)→pool(16×16)→tokens[3,256,64]   │
│                                                             │
│  位置编码: PE(tokens) [3, 256, 64]                            │
│                                                             │
│  CCM FiLM: tokens *= scale_ccm + bias_ccm [3, 256, 64]       │
│  Sensor FiLM: tokens *= scale_sen + bias_sen [3, 256, 64]     │
│                                                             │
│  Stage1: SelfAttn(norm(tokens)) + tokens → refined            │
│  Stage2: CrossAttn(norm(refined), norm(mcs)) + refined → fuse │
│  Stage3: CLSAttn(norm(cls), norm([fuse;mcs])) + cls → cls_out │
│                                                             │
│  输出:                                                       │
│    gain = Head(fuse * sigmoid(scale(cls)) + bias(cls))        │
│      → [3, 256, 3] → reshape → [3, 16, 16, 3]                │
│      → softplus.clamp(1e-4,4.0) → upsample → [3, 64, 64, 3]  │
│                                                             │
│  pred_image = gain * raw [3, 64, 64, 3]                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  total_loss()                                                │
│                                                             │
│  Step 1: 重塑为场景维度                                       │
│    pred_image [3,64,64,3] → reshape(1,3,64,64,3)             │
│    pred_gain [3,64,64,3] → reshape(1,3,64,64,3)              │
│                                                             │
│  Step 2: 后裁剪重叠区域                                       │
│    crop_to_overlap([1,3,64,64,3], crop_ratio, target=32)     │
│    → [1, 3, 32, 32, 3]                                      │
│                                                             │
│  Step 3: 计算各项损失                                         │
│    awb = angular_loss(gain_cropped[3,32,32,3], gt_gain[3,3]) │
│    rec = L1(pred_cropped[3,32,32,3], gt_cropped[3,32,32,3])  │
│    cons = chroma_consistency(pred_cropped[1,3,32,32,3])       │
│    smooth = smoothness(gain_cropped, raw_cropped) (如果启用)   │
│                                                             │
│  total = 1.0*awb + 10.0*rec + 2.0*cons + 0.05*smooth        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  反向传播                                                     │
│  backward() → clip_grad_norm(max_norm=5.0) → step()          │
└─────────────────────────────────────────────────────────────┘
```

### 推理时的简化流程

```
.npz + .npy → 加载 image_full (原始尺寸) + MCS + CCM + Focal
  → image resize 64×64 → MCS 对齐 64×64 → model → gain [64,64,3]
  → gain upsample → 原图分辨率 → image * gain → 保存
```

**关键差别**: 推理时保留原图尺寸用于最终输出，模型在 64×64 上快速推理后上采样 gain map。不需要 crop_ratio 和场景分组（单张图片独立推理）。

---

## 14. pytest 测试框架

**作用**: 自动化测试核心函数，确保代码修改不会引入回归错误。

### 14.1 测试文件结构

```
tests/
├── conftest.py              # 共享 fixtures（测试夹具）
├── test_loss.py             # loss.py 的测试 (24 个用例)
├── test_geometry_utils.py   # geometry_utils.py 的测试 (18 个用例)
└── test_model.py            # model.py 的测试 (18 个用例)
```

### 14.2 基础命令

```bash
# 激活虚拟环境
source .venv/Scripts/activate

# 运行所有测试
pytest

# 运行并显示详细输出（推荐）
pytest -v

# 运行并显示 print 输出
pytest -s

# 显示简短的错误信息
pytest --tb=short

# 显示完整的错误堆栈
pytest --tb=long
```

### 14.3 运行特定测试

```bash
# 运行单个文件
pytest tests/test_loss.py

# 运行单个类
pytest tests/test_loss.py::TestAngularLoss

# 运行单个函数
pytest tests/test_loss.py::TestAngularLoss::test_identical_vectors_returns_zero

# 按关键字匹配
pytest -k "angular"           # 运行名称包含 "angular" 的测试
pytest -k "not slow"          # 运行名称不包含 "slow" 的测试
pytest -k "model or loss"     # 运行名称包含 "model" 或 "loss" 的测试

# 运行最近失败的测试
pytest --lf

# 第一个失败后停止
pytest -x
```

### 14.4 测试覆盖率

```bash
# 运行并生成覆盖率报告
pytest --cov=. --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest --cov=. --cov-report=html

# 打开 HTML 报告查看详细覆盖情况
# Windows: start htmlcov/index.html
# Mac/Linux: open htmlcov/index.html
```

### 14.5 conftest.py — 共享 Fixtures

Fixtures 是 pytest 的核心概念，用于提供测试的前置条件和共享资源。

```python
# conftest.py 中定义的 fixtures 可在所有测试文件中使用

@pytest.fixture
def device():
    """返回可用的设备（CUDA 或 CPU）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_model(device):
    """创建一个小模型用于快速测试。"""
    model = AWBTransformer(dim=32, num_heads=4, grid_size=8).to(device)
    model.eval()
    return model

@pytest.fixture
def synthetic_batch(device):
    """创建合成的测试数据 batch。"""
    return {
        "image": torch.randn(2, 3, 32, 32, 3, device=device),
        "mcs": torch.randn(2, 3, 32, 32, 10, device=device),
        # ...
    }
```

**使用方式**:
```python
def test_model_forward(self, small_model, device):
    """测试中直接使用 fixture 名称作为参数。"""
    raw = torch.randn(2, 32, 32, 3, device=device)
    gain, _ = small_model(raw, ...)
    assert gain.shape == (2, 32, 32, 3)
```

### 14.6 编写测试的模式

#### 基本结构

```python
import pytest
import torch

class TestAngularLoss:
    """角度损失测试（类用于组织相关测试）。"""

    def test_identical_vectors_returns_zero(self, device):
        """测试相同向量应返回 0 损失。"""
        # 1. 准备数据
        pred = torch.randn(4, 32, 32, 3, device=device)
        gt = pred.clone()

        # 2. 执行测试
        loss = angular_loss(pred, gt)

        # 3. 断言结果
        assert loss.item() < 1e-5
```

#### 参数化测试

```python
@pytest.mark.parametrize("ratio,expected_mode", [
    (1.0, "main"),
    (0.5, "tele"),
    (2.0, "wide"),
])
def test_align_mcs_to_fov_modes(self, ratio, expected_mode):
    """测试 MCS 对齐的三种模式。"""
    mcs = np.random.randn(64, 64, 9).astype(np.float32)
    aligned, confidence = align_mcs_to_fov(mcs, ratio, (32, 32))

    if expected_mode == "main":
        assert np.allclose(confidence, 1.0)
    elif expected_mode == "tele":
        assert np.allclose(confidence, 1.0)
    elif expected_mode == "wide":
        assert confidence.max() > 0.5
```

### 14.7 常用断言

```python
# 相等/不等
assert x == y
assert x != y

# 浮点近似
assert abs(x - y) < 1e-6
assert torch.allclose(tensor1, tensor2, atol=1e-6)

# 范围检查
assert 0.0 <= x <= 1.0
assert tensor.min() >= 0.0
assert tensor.max() <= 4.0

# 形状检查
assert tensor.shape == (B, H, W, 3)
assert tensor.dim() == 4

# 异常检查
with pytest.raises(ValueError, match="必须是3的倍数"):
    some_function_that_raises()

# 布尔检查
assert torch.isnan(tensor).any() == False
assert torch.isinf(tensor).any() == False
```

### 14.8 标记测试

```python
# 在测试函数上添加标记
@pytest.mark.slow
def test_large_model_training():
    """标记为慢速测试。"""
    pass

@pytest.mark.gpu
def test_cuda_specific():
    """标记为需要 GPU。"""
    pass

# 运行时排除标记的测试
# pytest -m "not slow"
```

### 14.9 实际工作流

#### 场景 1：修改 loss.py 后验证

```bash
# 1. 修改 loss.py 中的代码

# 2. 运行相关测试
pytest tests/test_loss.py -v

# 3. 如果失败，查看详细错误
pytest tests/test_loss.py -v --tb=long

# 4. 修复后再次运行直到全部通过
pytest tests/test_loss.py -v
```

#### 场景 2：重构前后对比

```bash
# 1. 重构前运行测试，确保全部通过
pytest tests/ -v > before.txt

# 2. 执行重构

# 3. 重构后再次运行
pytest tests/ -v > after.txt

# 4. 对比结果
diff before.txt after.txt
```

#### 场景 3：检查测试覆盖率

```bash
# 1. 生成覆盖率报告
pytest --cov=. --cov-report=term-missing

# 2. 输出示例:
# ----------- coverage: platform win32 -----------
# Name                     Stmts   Miss  Cover   Missing
# ------------------------------------------------------
# loss.py                    120      15    88%   45-50, 78-82
# geometry_utils.py           95       8    92%   120-127
# model.py                   180      20    89%   250-269
# ------------------------------------------------------

# 3. 为未覆盖的代码添加测试
```

### 14.10 当前测试覆盖

| 模块 | 测试文件 | 用例数 | 覆盖内容 |
|------|----------|--------|----------|
| loss.py | test_loss.py | 24 | angular_loss, reconstruction_loss, scene_consistency_loss, srgb_loss, spatial_smoothness_loss, srgb_gamma, crop_to_overlap, build_srgb_gt, total_loss |
| geometry_utils.py | test_geometry_utils.py | 18 | focal_to_crop_ratio, center_crop, resize_image, align_mcs_to_fov, safe_inv_ccm, ensure_float01 |
| model.py | test_model.py | 18 | PositionalEncoding2D, CCMEncoder, FocalLengthEncoder, SensorEncoder, CCMHead, AWBTransformer |

### 14.11 常见问题

**Q: 测试失败怎么办？**
```bash
# 查看详细错误信息
pytest tests/test_xxx.py::TestClass::test_method -v --tb=long
```

**Q: 如何只运行失败的测试？**
```bash
pytest --lf  # 只运行上次失败的测试
pytest --ff  # 先运行上次失败的测试，再运行其他的
```

**Q: 如何并行运行测试加速？**
```bash
pip install pytest-xdist
pytest -n auto  # 自动检测 CPU 核心数并行运行
```

**Q: 如何调试测试？**
```bash
# 在测试中插入断点
import pdb; pdb.set_trace()

# 或使用 pytest 的调试模式
pytest --pdb  # 失败时自动进入调试器
```