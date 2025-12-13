# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register local dataloaders for camera-conditioned multiview training."""

from hydra.core.config_store import ConfigStore

import torch.distributed as dist

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2._src.predict2.camera.datasets.local_robot_multiview import LocalRobotMultiviewCameraDataset


def register_camera_data():
    cs = ConfigStore.instance()

    # Local robot multiview dataset (3 cameras) for post-training.
    train_dataset = L(LocalRobotMultiviewCameraDataset)(
        manifest_path="datasets/robot_multiview_local/train/manifest.jsonl",
        camera_order=("head", "hand_0", "hand_1"),
        num_frames_per_view=93,
        resolution_hw=(704, 1280),
        desired_fps=16,
        patch_spatial=16,
        shuffle=True,
        seed=0,
    )
    val_dataset = L(LocalRobotMultiviewCameraDataset)(
        manifest_path="datasets/robot_multiview_local/val/manifest.jsonl",
        camera_order=("head", "hand_0", "hand_1"),
        num_frames_per_view=93,
        resolution_hw=(704, 1280),
        desired_fps=16,
        patch_spatial=16,
        shuffle=False,
        seed=0,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="local_robot_multiview_train",
        node=L(get_generic_dataloader)(
            dataset=train_dataset,
            sampler=L(get_sampler)(dataset=train_dataset) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="local_robot_multiview_val",
        node=L(get_generic_dataloader)(
            dataset=val_dataset,
            sampler=L(get_sampler)(dataset=val_dataset) if dist.is_initialized() else None,
            batch_size=1,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
        ),
    )
