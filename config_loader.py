"""配置加载工具 —— 从 YAML 读取并解析所有项目配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# =============================================================================
# 结构化配置 dataclass
# =============================================================================

@dataclass
class DataConfig:
    root_dir: str = "./data"
    dump_camera_dir: str = "./data/image"
    dump_mcs_dir: str = "./data/McsBin"
    image_dir: str = "image"
    image_dng_dir: str = "image_dng"
    mcs_bin_dir: str = "McsBin"
    mcs_npy_dir: str = "Mcsnpy"
    image_processed_dir: str = "image_processed"
    debug_white_patch_dir: str = "image_processed/debug_white_patch"
    debug_mcs_dir: str = "Mcsnpy/debug_mcs"
    img_size: Tuple[int, int] = (128, 128)
    mcs_size: Tuple[int, int] = (128, 128)
    raw_extensions: List[str] = field(default_factory=lambda: [".dng", ".nef", ".arw"])


@dataclass
class ExternalToolsConfig:
    exiftool_path: Optional[str] = None


@dataclass
class ModelConfig:
    name: str = "AWBTransformer"
    dim: int = 64
    num_heads: int = 4
    grid_size: int = 16
    use_positional_encoding: bool = True
    sensor_embed_dim: int = 16
    predict_ccm: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 100
    scene_batch_size: int = 1
    learning_rate: float = 0.0001
    lr_scheduler: str = "cosine"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "awb": 1.0, "reconstruction": 10.0, "consistency": 2.0,
    })
    loss_crop_size: int = 64


@dataclass
class CheckpointConfig:
    dir: str = "./checkpoints"
    best_model: str = "best_model.pth"
    latest: str = "latest.pth"
    interval: int = 10
    resume_from: Optional[str] = None


@dataclass
class DebugConfig:
    output_dir: str = "./debug_outputs"
    save_interval: int = 5


@dataclass
class InferenceConfig:
    checkpoint: str = "./checkpoints/best_model.pth"
    output_dir: str = "./inference_outputs"
    save_gain_map: bool = True
    save_corrected_image: bool = True
    img_size: Tuple[int, int] = (128, 128)
    mcs_size: Tuple[int, int] = (128, 128)


@dataclass
class ProjectConfig:
    name: str = "awb_transformer"
    seed: int = 42


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    external_tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# =============================================================================
# 加载函数
# =============================================================================

def _resolve_path(base_dir: str, value: Optional[str]) -> Optional[str]:
    """将相对路径转为绝对路径；None 保持 None。"""
    if value is None:
        return None
    return os.path.normpath(os.path.join(base_dir, value))


def load_config(config_path: str = "config.yaml") -> Config:
    """从 YAML 文件加载配置，返回结构化 Config 对象。"""
    base_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # ---- project ----
    proj_raw = raw.get("project", {})
    project = ProjectConfig(
        name=proj_raw.get("name", "awb_transformer"),
        seed=proj_raw.get("seed", 42),
    )

    # ---- data ----
    d = raw.get("data", {})
    data = DataConfig(
        root_dir=_resolve_path(base_dir, d.get("root_dir", "./data")),
        dump_camera_dir=_resolve_path(base_dir, d.get("dump_camera_dir", "./data/image")),
        dump_mcs_dir=_resolve_path(base_dir, d.get("dump_mcs_dir", "./data/McsBin")),
        image_dir=d.get("image_dir", "image"),
        image_dng_dir=d.get("image_dng_dir", "image_dng"),
        mcs_bin_dir=d.get("mcs_bin_dir", "McsBin"),
        mcs_npy_dir=d.get("mcs_npy_dir", "Mcsnpy"),
        image_processed_dir=d.get("image_processed_dir", "image_processed"),
        debug_white_patch_dir=d.get("debug_white_patch_dir", "image_processed/debug_white_patch"),
        debug_mcs_dir=d.get("debug_mcs_dir", "Mcsnpy/debug_mcs"),
        img_size=tuple(d.get("img_size", [128, 128])),
        mcs_size=tuple(d.get("mcs_size", [128, 128])),
        raw_extensions=d.get("raw_extensions", [".dng", ".nef", ".arw"]),
    )

    # ---- external_tools ----
    et = raw.get("external_tools", {})
    ext_path = et.get("exiftool_path", None)
    external_tools = ExternalToolsConfig(
        exiftool_path=ext_path if ext_path else None,
    )

    # ---- model ----
    m = raw.get("model", {})
    model = ModelConfig(
        name=m.get("name", "AWBTransformer"),
        dim=m.get("dim", 64),
        num_heads=m.get("num_heads", 4),
        grid_size=m.get("grid_size", 16),
        use_positional_encoding=m.get("use_positional_encoding", True),
        sensor_embed_dim=m.get("sensor_embed_dim", 16),
        predict_ccm=m.get("predict_ccm", False),
    )

    # ---- training ----
    t = raw.get("training", {})
    training = TrainingConfig(
        epochs=t.get("epochs", 100),
        scene_batch_size=t.get("scene_batch_size", 1),
        learning_rate=t.get("learning_rate", 0.0001),
        lr_scheduler=t.get("lr_scheduler", "cosine"),
        lr_scheduler_params=t.get("lr_scheduler_params", {}),
        loss_weights=t.get("loss_weights", {"awb": 1.0, "reconstruction": 10.0, "consistency": 2.0}),
        loss_crop_size=t.get("loss_crop_size", 64),
    )

    # ---- checkpoint ----
    c = raw.get("checkpoint", {})
    checkpoint = CheckpointConfig(
        dir=_resolve_path(base_dir, c.get("dir", "./checkpoints")),
        best_model=c.get("best_model", "best_model.pth"),
        latest=c.get("latest", "latest.pth"),
        interval=c.get("interval", 10),
        resume_from=_resolve_path(base_dir, c["resume_from"]) if c.get("resume_from") else None,
    )

    # ---- debug ----
    dbg = raw.get("debug", {})
    debug = DebugConfig(
        output_dir=_resolve_path(base_dir, dbg.get("output_dir", "./debug_outputs")),
        save_interval=dbg.get("save_interval", 5),
    )

    # ---- inference ----
    inf = raw.get("inference", {})
    inference = InferenceConfig(
        checkpoint=_resolve_path(base_dir, inf.get("checkpoint", "./checkpoints/best_model.pth")),
        output_dir=_resolve_path(base_dir, inf.get("output_dir", "./inference_outputs")),
        save_gain_map=inf.get("save_gain_map", True),
        save_corrected_image=inf.get("save_corrected_image", True),
        img_size=tuple(inf.get("img_size", [128, 128])),
        mcs_size=tuple(inf.get("mcs_size", [128, 128])),
    )

    return Config(
        project=project,
        data=data,
        external_tools=external_tools,
        model=model,
        training=training,
        checkpoint=checkpoint,
        debug=debug,
        inference=inference,
    )


def build_lr_scheduler(
    optimizer,
    cfg: TrainingConfig,
) -> Optional[object]:
    """根据配置创建学习率调度器。"""
    name = cfg.lr_scheduler.lower()
    params = cfg.lr_scheduler_params

    if name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=params.get("T_max", cfg.epochs),
            eta_min=params.get("eta_min", 1e-6),
        )
    elif name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params.get("factor", 0.5),
            patience=params.get("patience", 10),
            threshold=params.get("threshold", 1e-4),
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown lr_scheduler: {name}")
