import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from geometry_utils import compute_scene_crop_plan, resize_image, align_mcs_to_fov


class AWBDataset(Dataset):
    def __init__(self, root_dir: str, img_size=(128, 128), mcs_size=(128, 128)):
        self.root = root_dir
        self.img_size = tuple(img_size)
        self.mcs_size = tuple(mcs_size)

        img_dir = os.path.join(root_dir, "image_processed")
        mcs_dir = os.path.join(root_dir, "Mcsnpy")

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
        mcs_files = sorted([f for f in os.listdir(mcs_dir) if f.endswith(".npy")])

        assert len(img_files) % 3 == 0, "dng数量必须是3的倍数"
        assert len(mcs_files) == len(img_files), "mcs和img数量不匹配"

        self.scenes: List[List[Dict[str, Any]]] = []

        for scene_id, i in enumerate(range(0, len(img_files), 3)):
            group = img_files[i : i + 3]
            mcs_group = mcs_files[i : i + 3]

            # ===== 按 focal_length 降序排列，分配 sensor_id =====
            # tele(焦距最大)=0, main=1, wide(焦距最小)=2
            scene_meta = []
            for img_name, mcs_name in zip(group, mcs_group):
                img_path = os.path.join(img_dir, img_name)
                img_data = np.load(img_path, allow_pickle=True)
                focal_length = float(
                    img_data["focal_length_35mm"]
                    if "focal_length_35mm" in img_data
                    else img_data["focal_length"]
                )
                scene_meta.append({
                    "img_name": img_name,
                    "mcs_name": mcs_name,
                    "focal_length": focal_length,
                })

            # 焦距降序：tele(最大) → main → wide(最小)
            scene_meta.sort(key=lambda x: x["focal_length"], reverse=True)

            scene_samples = []
            for sensor_id, meta in enumerate(scene_meta):
                img_path = os.path.join(img_dir, meta["img_name"])
                mcs_path = os.path.join(mcs_dir, meta["mcs_name"])

                img_data = np.load(img_path, allow_pickle=True)
                mcs_data = np.load(mcs_path).astype(np.float32)

                awb_gt_gain = (
                    img_data["awb_gt_gain"].astype(np.float32)
                    if "awb_gt_gain" in img_data
                    else np.ones(3, dtype=np.float32)
                )
                white_patch_rgb = (
                    img_data["white_patch_rgb"].astype(np.float32)
                    if "white_patch_rgb" in img_data
                    else np.ones(3, dtype=np.float32)
                )

                sample = {
                    "image": img_data["image"].astype(np.float32),
                    "focal_length": np.float32(meta["focal_length"]),
                    "ccm1": img_data["xyz2camera_rgb1"].astype(np.float32),
                    "ccm2": img_data["xyz2camera_rgb2"].astype(np.float32),
                    "mcs": mcs_data,
                    "sensor_id": np.int64(sensor_id),
                    "scene_id": np.int64(scene_id),
                    "file_name": str(img_data["file_name"]),
                    "awb_gt_gain": awb_gt_gain,
                    "white_patch_rgb": white_patch_rgb,
                    "white_patch_box": (
                        img_data["white_patch_box"].astype(np.int32)
                        if "white_patch_box" in img_data
                        else np.zeros(4, dtype=np.int32)
                    ),
                    "crop_strategy": (
                        str(img_data["crop_strategy"])
                        if "crop_strategy" in img_data
                        else "intrinsics_fallback_center_crop"
                    ),
                }
                scene_samples.append(sample)

            self.scenes.append(scene_samples)

    def __len__(self):
        return len(self.scenes)

    def _prepare_scene(self, scene_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        focal_lengths = [sample["focal_length"] for sample in scene_samples]
        crop_plan = compute_scene_crop_plan(focal_lengths)

        # MCS 固定 FOV ≈ Main 模组，取 main 的焦距作为对齐基准
        # scene_samples 已按焦距降序排列: [tele, main, wide]
        ref_focal = float(focal_lengths[1]) if len(focal_lengths) > 1 else float(focal_lengths[0])

        images = []
        gt_images = []
        mcs_maps = []
        awb_gt_gain = []
        white_patch_rgb = []
        crop_boxes = []
        crop_ratios = []
        sensor_ids = []
        scene_ids = []
        ccm1_list = []
        ccm2_list = []

        for sample, plan in zip(scene_samples, crop_plan):
            crop_ratio = plan["crop_ratio"]

            # 图像：仅 resize 不裁剪，保留全视场信息
            image = resize_image(sample["image"], self.img_size, interpolation=cv2.INTER_AREA)

            # MCS：空间对齐到对应模组的视场
            # align_ratio = ref_focal / focal_camera
            #   = 1.0 → main（不变）
            #   < 1.0 → tele（中心裁剪放大）
            #   > 1.0 → wide（缩小 + reflect 外推）
            focal = float(sample["focal_length"])
            align_ratio = ref_focal / max(focal, 1e-4)
            mcs_aligned, confidence = align_mcs_to_fov(sample["mcs"], align_ratio, self.mcs_size)

            # MCS 从 9 通道扩展为 10 通道（9 光谱 + 1 置信度）
            # 置信度让模型知道 Wide 边缘的外推 MCS 不完全可靠
            mcs = np.concatenate([mcs_aligned, confidence[..., None]], axis=-1).astype(np.float32)

            gain = sample["awb_gt_gain"].astype(np.float32)
            gt_image = np.clip(image * gain.reshape(1, 1, 3), 0.0, 1.0)

            images.append(image)
            gt_images.append(gt_image)
            mcs_maps.append(mcs.astype(np.float32))
            awb_gt_gain.append(gain)
            white_patch_rgb.append(sample["white_patch_rgb"])
            crop_boxes.append(np.zeros(4, dtype=np.int32))  # 占位，loss 阶段用 crop_ratio 计算
            crop_ratios.append(np.float32(crop_ratio))
            sensor_ids.append(sample["sensor_id"])
            scene_ids.append(sample["scene_id"])
            ccm1_list.append(sample["ccm1"])  # [3, 3]
            ccm2_list.append(sample["ccm2"])  # [3, 3]

        batch = {
            "image": torch.from_numpy(np.stack(images, axis=0)),
            "gt_image": torch.from_numpy(np.stack(gt_images, axis=0)),
            "mcs": torch.from_numpy(np.stack(mcs_maps, axis=0)),
            "awb_gt_gain": torch.from_numpy(np.stack(awb_gt_gain, axis=0)),
            "white_patch_rgb": torch.from_numpy(np.stack(white_patch_rgb, axis=0)),
            "crop_box": torch.from_numpy(np.stack(crop_boxes, axis=0)),
            "crop_ratio": torch.from_numpy(np.array(crop_ratios, dtype=np.float32)),
            "sensor_id": torch.from_numpy(np.array(sensor_ids, dtype=np.int64)),
            "scene_id": torch.from_numpy(np.array(scene_ids, dtype=np.int64)),
            "focal_length": torch.from_numpy(np.array(focal_lengths, dtype=np.float32)),
            "ccm1": torch.from_numpy(np.stack(ccm1_list, axis=0)),  # [S, 3, 3]
            "ccm2": torch.from_numpy(np.stack(ccm2_list, axis=0)),  # [S, 3, 3]
        }
        return batch

    def __getitem__(self, idx):
        return self._prepare_scene(self.scenes[idx])