# AWB Transformer — 项目约束

## 环境
- 激活虚拟环境: `source .venv/Scripts/activate`
- exiftool: `E:/ricky/research/oppo-project-2025/exiftool-13.53_64/exiftool-13.53_64/exiftool.exe`

## 通用规则
- 路径用相对路径（相对于项目根），通过 `config.yaml` 统一管理
- 中文注释只写 WHY，不写 WHAT
- 文件/函数 snake_case，类名 PascalCase，tensor shape 注释 `[B, C, H, W]`
- 新增配置项需同步修改 `config.yaml` 和 `config_loader.py`

## 模型架构 — 不可破坏
- **MCS 10 通道格式**: 9 光谱 + 1 置信度，最后一维必须为 10
- **raw_tokens 和 mcs_tokens 共用 2D 位置编码**: 两者 `grid_size` 必须一致
- **后裁剪机制**: 模型输出全图 gain_map，loss 时裁剪重叠区域
- **焦距编码**: 即使 MCS 已空间对齐，仍需保留作为 camera identity 条件输入
- **MCS 对齐只在 dataloader 层**: `align_mcs_to_fov()` 在 `dataloader.py` 中，model 接收已对齐的 mcs tensor

