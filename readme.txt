================================================================================
AWB Transformer Demo — 项目说明与执行方案
================================================================================

一、项目做什么
--------------------------------------------------------------------------------
多摄（tele / main / wide）同一场景下，利用：
  - 线性域 RAW 图像（.npz）
  - 多光谱 MCS（.npy，9 通道；训练时在 Dataset 中对齐后扩展为 10 通道，含置信度）
  - EXIF 中的焦距、CCM（xyz2camera_rgb1/2）
训练 AWBTransformer：预测空间增益图 gain，使 pred_image = gain * input 更接近 GT；
可选预测 CCM 修正量，用于 sRGB 监督（由 config.yaml 中 predict_ccm 与 loss_weights 控制）。

核心代码（项目根目录）：
  config.yaml / config_loader.py   全局配置与加载
  train.py                          训练入口（断点、调度器、debug、checkpoint）
  infer.py                          推理入口
  dataloader.py                     按 scene 组 batch；RAW 全视场；MCS 与各摄视场对齐
  model.py                          AWBTransformer（RAW/MCS/CCM/焦距编码 + attention）
  loss.py                           AWB 角误差、重建、scene 一致性、可选 sRGB
  geometry_utils.py                 焦距重叠比例、MCS 对齐 align_mcs_to_fov
  gt_utils.py                       白块启发式检测与调试图显示映射

数据脚本（data/ 目录）：
  demo_ImageMatchBindata.py         从 dump 匹配复制 jpg/dng/bin 到 data 子目录
  demo_readMcsRaw.py                MCS .bin → Mcsnpy/*.npy（按通道稳健归一化）
  visualize_mcs.py                  MCS 9 通道调试图
  dng_process_rawpy.py              一体化：DNG 解码 + EXIF + 白块 GT → image_processed
  dng_decoder.py                    解耦：仅解码 + 元数据 → image_raw（无 GT）
  gt_extractor.py                   解耦：在 image_raw 上独立提取 GT → image_processed


二、环境与依赖
--------------------------------------------------------------------------------
建议使用项目自带虚拟环境（根目录 .venv），避免系统 Python 缺少 torch/cv2：

  Windows（PowerShell，在项目根目录 demo/ 下）：
    .\.venv\Scripts\python.exe -c "import torch, cv2; print(torch.__version__, cv2.__version__)"

主要依赖：
  torch, numpy, opencv-python, rawpy, PyYAML；exiftool（Python 包 + 本机可执行文件）

ExifTool 路径：
  - 根目录 config.yaml → external_tools.exiftool_path
  - 若为 null，则使用系统 PATH 中的 exiftool
  - dng_process_rawpy.py 通过 load_config 读取上述配置
  - data/dng_decoder.py 内含 DEFAULT_EXIFTOOL 常量；若本机路径不同请修改或改为读取 config


三、目录与数据约定（由 config.yaml 的 data.root_dir 指定，默认 ./data）
--------------------------------------------------------------------------------
  data/image/                 匹配后的 JPG（调试用）
  data/image_dng/             匹配后的 DNG
  data/McsBin/                匹配后的 MCS .bin
  data/Mcsnpy/                解析后的 MCS .npy（9 通道）
  data/image_processed/       训练用 .npz（含 image、焦距、CCM、GT 等）
  data/image_raw/             解耦流程：仅解码的 .npz（由 dng_decoder 写入，若启用）

训练数据配对约定（dataloader.py）：
  - image_processed 与 Mcsnpy 下 .npz / .npy 数量一致，且总张数为 3 的倍数
  - 每连续 3 个文件视为一个 scene；再按 focal_length_35mm（无则 focal_length）降序
    重排为 [tele, main, wide]，保证 sensor_id 与焦距语义一致
  - 每个样本仍为「一张 RAW npz 对应一份 MCS npy」，不改为 scene 共用单份 MCS


四、详细执行步骤（推荐顺序）
================================================================================

步骤 0：配置
--------------------------------------------------------------------------------
编辑项目根目录 config.yaml：
  - data：root_dir、各子目录名、img_size / mcs_size、raw_extensions、调试输出路径
  - external_tools.exiftool_path
  - model：dim、num_heads、grid_size、use_positional_encoding、focal_embed_dim、predict_ccm
  - training：epochs、scene_batch_size、learning_rate、loss_weights、loss_crop_size、lr_scheduler
  - checkpoint / debug / inference

说明：config_loader.ModelConfig 中字段名为 sensor_embed_dim；YAML 使用 focal_embed_dim；
train.py 使用 getattr(cfg.model, "focal_embed_dim", 16) 兼容。建议团队统一命名。


步骤 1：从 dump 匹配并拷贝原始数据
--------------------------------------------------------------------------------
在 data/ 下运行（路径以 demo_ImageMatchBindata.py 内配置为准，可按机器修改）：

  .\.venv\Scripts\python.exe .\data\demo_ImageMatchBindata.py

产出：image/、image_dng/、McsBin/ 等对齐文件。


步骤 2：MCS .bin → .npy
--------------------------------------------------------------------------------
在 data/ 下运行：

  .\.venv\Scripts\python.exe .\data\demo_readMcsRaw.py

实现要点（demo_readMcsRaw.py）：
  - 解析 bin 头：黑电平、高宽
  - 重组为 H×W×9 多光谱块
  - 减黑电平后，按通道做分位数（默认 99.5%）归一化到 [0,1]，缓和通道动态范围差异


步骤 2b（可选）：MCS 调试图
--------------------------------------------------------------------------------
在 data/ 下运行：

  .\.venv\Scripts\python.exe .\data\visualize_mcs.py

输出：config 中 data.debug_mcs_dir 所设目录（默认 Mcsnpy/debug_mcs）下的 *_mcs_debug.png


步骤 3：DNG → 训练用 NPZ（二选一）
================================================================================

【流程 A — 一体化，简单】快速跑通
--------------------------------------------------------------------------------
在 data/ 下运行：

  .\.venv\Scripts\python.exe .\data\dng_process_rawpy.py

作用：
  - rawpy 解码 DNG → 线性图，按 white_level 归一化到 [0,1]
  - exiftool 读焦距、CCM、分辨率等
  - gt_utils.extract_awb_gt：启发式白块 + awb_gt_gain（以绿为基准 [G/R,1,G/B]）
  - 保存 image（resize）、image_full、元数据、GT 字段
  - 白块调试图：debug_white_patch/*.png（先做 GT 白平衡再自动曝光，便于肉眼核对）


【流程 B — 解耦，便于换 GT 方法】科研迭代推荐
--------------------------------------------------------------------------------
(1) 仅解码 + 元数据 → image_raw/

  .\.venv\Scripts\python.exe .\data\dng_decoder.py

(2) 独立 GT 提取 → image_processed/

  .\.venv\Scripts\python.exe .\data\gt_extractor.py --input_dir <image_raw> --output_dir <image_processed> --method colorchecker
  或：
  .\.venv\Scripts\python.exe .\data\gt_extractor.py ... --method white_patch [--roi "x0,y0,x1,y1"]

详见 gt_extractor.py 文件头部用法与参数说明。


步骤 4：训练
--------------------------------------------------------------------------------
在项目根目录运行：

  .\.venv\Scripts\python.exe .\train.py

默认读取根目录 config.yaml（见 train.py）。

训练数据流（摘要）：
  - DataLoader：每个 item 为一个 scene，含 3 张图（tele/main/wide）
  - RAW：resize 到 img_size，保留全视场（不按焦距先裁图）
  - MCS：每张仍对应各自 npy；对 MCS 做 align_mcs_to_fov（以 scene 内 main 焦距为 ref），
    再拼第 10 通道置信度；与「一张 raw 一份 mcs」一致，但空间上对齐到当前摄的视场表达
  - 模型前向：reshape 为 [B*3,...]；输入 raw、mcs(10ch)、ccm1、ccm2、focal_length
  - 输出：pred_gain；predict_ccm 时另有 ccm_delta
  - Loss：angular（pred_gain 空间均值 vs 标量 gt_gain）；reconstruction 与 consistency
    在重叠区后裁剪（crop_ratio + loss_crop_size，见 loss.crop_to_overlap）
  - 若启用 sRGB 损失：需 training.loss_weights 中含 srgb 且 predict_ccm 为 true

Checkpoint（默认 ./checkpoints/）：
  latest.pth、best_model.pth、按 interval 的周期保存

Debug（默认 ./debug_outputs/train/）：
  epoch_XXX.png：三行（三摄）× 三列（input | pred | gt），统一显示尺度
  epoch_XXX_mcs_alignment.png：RAW / MCS 前 3 通道伪彩 / 置信度叠加


步骤 5：推理
--------------------------------------------------------------------------------
  .\.venv\Scripts\python.exe .\infer.py
  或：
  .\.venv\Scripts\python.exe .\infer.py --config my_config.yaml
  或指定文件：
  .\.venv\Scripts\python.exe .\infer.py --npz data/image_processed/XXX.npz --mcs data/Mcsnpy/XXX.npy

推理侧应与训练一致：图像 resize、MCS resize、align_mcs_to_fov、从 checkpoint 读取保存的 model 超参。


五、实现功能细节（对照代码）
================================================================================

1）Scene 与 sensor 顺序（dataloader.py）
  - 文件名排序后每 3 条为一组 scene
  - 组内按 35mm 等效焦距（或物理焦距）降序：sensor_id 0=tele，1=main，2=wide

2）重叠视场与 loss 区域（geometry_utils.compute_scene_crop_plan + loss.crop_to_overlap）
  - crop_ratio = f / max(f)，表示相对最窄视场的重叠比例
  - RAW 训练输入为全视场；重建与一致性在重叠子域上监督，减轻 tele/wide 与 main 像素不对齐

3）MCS 对齐（geometry_utils.align_mcs_to_fov）
  - ref_focal 取 scene 内 main（排序后索引 1）
  - align_ratio = ref_focal / focal_current
  - main：近似不变，置信度全 1
  - tele：中心裁剪再放大，置信度全 1
  - wide：缩小 + reflect pad，置信度从中心到边缘高斯衰减（第 10 通道）

4）模型（model.py）
  - RAW / MCS 各自 Conv → adaptive pool 到 grid_size×grid_size tokens
  - 可选共享 2D sinusoidal PE（RAW 与 MCS token 共用同一套网格 PE）
  - CCMEncoder：对 xyz2camera 的逆编码后 FiLM 调制 RAW tokens
  - SensorEncoder：连续焦距正弦编码 + MLP → FiLM 调制 RAW tokens
  - Attention：RAW self-attn → cross(RAW←MCS) → CLS attn；CLS 再 FiLM；head 输出 gain，softplus

5）GT（gt_utils / gt_extractor / dng_process_rawpy）
  - 一体化流程：白块启发式；解耦流程：色卡或白块，可 ROI
  - Dataset 中 gt_image = clip(image * awb_gt_gain, 0, 1)（标量 gain 乘每像素）

6）predict_ccm 与 sRGB
  - config 中 predict_ccm: true 时，若需要显式 sRGB 监督，请在 training.loss_weights 中增加 srgb；
    否则 CCM 头可能缺少直接监督信号。


六、常见问题
================================================================================
Q：调试图仍不好辨认线性 RAW？
A：白块 debug 与 train debug 已使用显示映射；若需原始线性对照，可自行另存未 gamma 的版本。

Q：scene 错乱或 loss 异常？
A：检查三文件是否真同场景、文件名排序是否稳定；长期建议 manifest（CSV）显式列出 triple。

Q：MCS 与 RAW 视觉对不齐？
A：看 debug_outputs/train/*_mcs_alignment.png；检查焦距 EXIF、main 是否为排序后中间一档。


七、版本记录（建议自行维护）
================================================================================
可在此记录：数据集版本、config 改动、checkpoint 与实验名称对应关系。
