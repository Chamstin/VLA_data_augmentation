# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local multiview robot dataset for camera-conditioned Video2World training.

Each sample is defined by a JSONL manifest entry with:
  {
    "sample_id": "...",
    "prompt": "...",
    "videos": {"head": "...mp4", "hand_0": "...mp4", "hand_1": "...mp4"},
    "cameras": {
        "head": {"intrinsic": "...txt", "extrinsic": "...txt"},
        "hand_0": {...},
        "hand_1": {...}
    }
  }
Camera txt formats follow AgiBot convention:
  - intrinsic: T x 4, each row: fx fy cx cy
  - extrinsic: T x 12, each row is 3x4 camera-to-world matrix (row-major).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from einops import rearrange

from cosmos_predict2._src.imaginaire.modules.camera import Camera


@dataclass
class ViewSpec:
    name: str
    video_path: Path
    intrinsic_path: Path | None
    extrinsic_path: Path | None


class LocalRobotMultiviewCameraDataset(torch.utils.data.Dataset):
    """Loads synchronized 3-view robot videos + optional per-frame camera params."""

    def __init__(
        self,
        manifest_path: str,
        camera_order: Sequence[str] = ("head", "hand_0", "hand_1"),
        num_frames_per_view: int = 93,
        resolution_hw: tuple[int, int] = (704, 1280),
        desired_fps: int = 16,
        patch_spatial: int = 16,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.camera_order = list(camera_order)
        self.num_frames_per_view = int(num_frames_per_view)
        self.resolution_hw = tuple(resolution_hw)
        self.desired_fps = int(desired_fps)
        self.patch_spatial = int(patch_spatial)
        self.shuffle = shuffle
        self.seed = seed

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.samples: list[dict[str, Any]] = []
        with self.manifest_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(self.samples)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _open_vr(self, path: Path) -> VideoReader:
        return VideoReader(str(path), ctx=cpu(0), num_threads=2)

    def _pick_window(self, total_frames: int, downsample: int) -> int:
        needed = self.num_frames_per_view * downsample
        if total_frames < needed:
            return 0
        max_start = total_frames - needed
        return random.randint(0, max_start) if max_start > 0 else 0

    def _resize_center_crop(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        """Resize with aspect-preserving scale then center-crop to resolution_hw.

        frames_tchw: uint8 tensor [T, C, H, W]
        """
        target_h, target_w = self.resolution_hw
        _, _, h, w = frames_tchw.shape
        scale = max(target_w / w, target_h / h)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        frames = frames_tchw.float()
        try:
            frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        except TypeError:
            frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = max((new_h - target_h) // 2, 0)
        left = max((new_w - target_w) // 2, 0)
        frames = frames[:, :, top : top + target_h, left : left + target_w]
        frames = frames.clamp(0.0, 255.0).to(torch.uint8)
        return frames

    def _load_frames(
        self, vr: VideoReader, frame_ids: list[int]
    ) -> tuple[torch.Tensor, float]:
        frames = vr.get_batch(frame_ids).asnumpy()  # T,H,W,3 uint8
        frames = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = float(self.desired_fps)
        return frames, fps

    def _load_camera_params(self, path: Path | None) -> np.ndarray | None:
        if path is None:
            return None
        arr = np.loadtxt(str(path))
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _select_rows(self, arr: np.ndarray, row_ids: list[int]) -> np.ndarray:
        if arr.shape[0] == 0:
            raise ValueError("Empty camera parameter array")
        max_id = arr.shape[0] - 1
        sel = []
        for rid in row_ids:
            sel.append(arr[min(rid, max_id)])
        return np.stack(sel, axis=0)

    def _compute_plucker_rays(
        self,
        intrinsics_params: np.ndarray | None,
        extrinsics_params: np.ndarray | None,
        latent_row_ids: list[int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Compute plucker rays for one view. Returns [T_latent, H_p, W_p, patch^2*6]."""
        latent_frames = len(latent_row_ids)
        if intrinsics_params is None:
            fx = fy = float(width)
            cx = float(width) / 2.0
            cy = float(height) / 2.0
            intrinsics_params = np.tile(np.array([fx, fy, cx, cy], dtype=np.float32)[None, :], (latent_frames, 1))
        else:
            intrinsics_params = self._select_rows(intrinsics_params, latent_row_ids).astype(np.float32)

        if extrinsics_params is None:
            # Identity camera-to-world pose
            extrinsics_params = np.tile(
                np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32)[None, :],
                (latent_frames, 1),
            )
        else:
            extrinsics_params = self._select_rows(extrinsics_params, latent_row_ids).astype(np.float32)

        extrinsics_tgt = torch.tensor(extrinsics_params, dtype=torch.float32)
        extrinsics_tgt = torch.cat(
            (
                extrinsics_tgt,
                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0).expand(latent_frames, -1),
            ),
            dim=1,
        ).reshape(-1, 4, 4)

        intrinsics_tgt = torch.tensor(intrinsics_params, dtype=torch.float32)
        K = Camera.intrinsic_params_to_matrices(intrinsics_tgt)
        w2c = Camera.invert_pose(extrinsics_tgt[:, :3, :])

        plucker_flat = Camera.get_plucker_rays(w2c, K, (height, width))
        plucker_rays = plucker_flat.view(plucker_flat.shape[0], height, width, 6)
        plucker_rays = rearrange(
            plucker_rays,
            "T (H p1) (W p2) C -> T H W (p1 p2 C)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
        )
        return plucker_rays

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        prompt: str = sample.get("prompt") or sample.get("caption") or ""
        if not isinstance(prompt, str):
            prompt = str(prompt)

        videos: dict[str, str] = sample["videos"]
        cameras: dict[str, dict[str, str]] = sample.get("cameras", {})

        view_specs: list[ViewSpec] = []
        for view_name in self.camera_order:
            vpath = Path(videos[view_name])
            cam_info = cameras.get(view_name, {})
            ipath = Path(cam_info["intrinsic"]) if "intrinsic" in cam_info else None
            epath = Path(cam_info["extrinsic"]) if "extrinsic" in cam_info else None
            view_specs.append(ViewSpec(view_name, vpath, ipath, epath))

        # Open videos and determine sync window based on shortest view.
        vrs = [self._open_vr(v.video_path) for v in view_specs]
        total_frames = min(len(vr) for vr in vrs)

        # Determine downsample factor from first view.
        try:
            original_fps = float(vrs[0].get_avg_fps())
        except Exception:
            original_fps = float(self.desired_fps)
        downsample = max(int(round(original_fps / self.desired_fps)), 1)

        start_frame = self._pick_window(total_frames, downsample)
        end_frame = start_frame + self.num_frames_per_view * downsample
        frame_ids = [min(i, total_frames - 1) for i in range(start_frame, end_frame, downsample)]

        multiview_frames: list[torch.Tensor] = []
        for vr in vrs:
            frames, _fps = self._load_frames(vr, frame_ids)
            frames = self._resize_center_crop(frames)
            multiview_frames.append(frames)

        # Stack views along time: (V*T, C, H, W) -> (C, V*T, H, W)
        stacked_tchw = torch.cat(multiview_frames, dim=0)
        video_cthw = rearrange(stacked_tchw, "t c h w -> c t h w")

        # Camera rays per view (latent frames aligned to sampled window)
        latent_frames = self.num_frames_per_view // 4 + 1
        latent_row_ids = [start_frame + i * 4 * downsample for i in range(latent_frames)]

        camera_views: list[torch.Tensor] = []
        for v in view_specs:
            intr = self._load_camera_params(v.intrinsic_path)
            extr = self._load_camera_params(v.extrinsic_path)
            camera_views.append(
                self._compute_plucker_rays(
                    intrinsics_params=intr,
                    extrinsics_params=extr,
                    latent_row_ids=latent_row_ids,
                    height=self.resolution_hw[0],
                    width=self.resolution_hw[1],
                )
            )
        camera_tensor = torch.cat(camera_views, dim=0)  # (V*T_latent, H_p, W_p, feat)

        out: Dict[str, Any] = {
            "__key__": str(index),
            "__url__": str(sample.get("sample_id", index)),
            "video": video_cthw,  # uint8 C,T,H,W
            "ai_caption": prompt,
            "fps": float(self.desired_fps),
            "padding_mask": torch.zeros((1, *self.resolution_hw), dtype=torch.float32),
            "num_frames": torch.tensor(self.num_frames_per_view, dtype=torch.int64),
            "camera": camera_tensor,
            "image_size": torch.tensor(
                [self.resolution_hw[0], self.resolution_hw[1], self.resolution_hw[0], self.resolution_hw[1]],
                dtype=torch.float32,
            ),
        }
        return out
