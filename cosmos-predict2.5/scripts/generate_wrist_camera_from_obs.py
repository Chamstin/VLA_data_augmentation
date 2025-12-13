# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate per-frame camera intrinsics/extrinsics from obs.json for a wrist-mounted camera.

This is a helper for users who don't have IMU/calibration but do have robot TCP poses.

Assumptions (match your sample obs.json):
  - obs.json is a dict with key "data": list[frame]
  - each frame has "<arm>_tcp" as [x, y, z, qx, qy, qz, qw]
    representing TCP pose in robot base/world coordinates.
  - You provide a constant TCP->Camera mount offset (translation + rotation).

Outputs:
  cameras/intrinsic_<view>.txt : T x 4 lines (fx fy cx cy)
  cameras/extrinsic_<view>.txt : T x 12 lines (3x4 cam2world, row-major)

Example:
  python scripts/generate_wrist_camera_from_obs.py \
    --episode-dir fitune_dataset/sub_dataset2/0000 \
    --tcp-key right_arm_tcp \
    --view-name hand_0 \
    --offset-xyz 0.02 0.00 0.05 \
    --offset-rpy-deg 0 -90 0 \
    --fx-fy-cx-cy 950 950 640 352
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-dir", type=str, required=True, help="Directory containing obs.json")
    p.add_argument("--obs-name", type=str, default="obs.json")
    p.add_argument("--tcp-key", type=str, required=True, help="e.g., right_arm_tcp or left_arm_tcp")
    p.add_argument("--view-name", type=str, default="hand_0", help="hand_0 / hand_1 / head")

    p.add_argument("--offset-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="TCP->Camera translation (m)")
    p.add_argument(
        "--offset-rpy-deg",
        type=float,
        nargs=3,
        default=None,
        help="TCP->Camera rotation as roll pitch yaw in degrees (XYZ order, right-handed).",
    )
    p.add_argument(
        "--offset-quat",
        type=float,
        nargs=4,
        default=None,
        help="TCP->Camera rotation as quaternion qx qy qz qw (w-last). Overrides --offset-rpy-deg.",
    )

    p.add_argument(
        "--fx-fy-cx-cy",
        type=float,
        nargs=4,
        default=None,
        help="Camera intrinsics. If omitted, uses fx=fy=width, cx=width/2, cy=height/2.",
    )
    p.add_argument("--resolution-hw", type=int, nargs=2, default=(704, 1280), help="H W for default intrinsics")
    p.add_argument("--downsample", type=int, default=1, help="Write every Nth frame of obs")
    return p.parse_args()


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (qx,qy,qz,qw) to rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / n
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (rad), XYZ then yaw-pitch-roll composition."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_obs_frames(obs_path: Path) -> list[dict]:
    raw = json.loads(obs_path.read_text())
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"Unknown obs.json format at {obs_path}")


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    obs_path = episode_dir / args.obs_name
    frames = load_obs_frames(obs_path)
    frames = frames[:: max(args.downsample, 1)]

    # TCP->camera constant transform
    t_tcp_cam = np.array(args.offset_xyz, dtype=np.float64)
    if args.offset_quat is not None:
        R_tcp_cam = quat_to_rot(np.array(args.offset_quat, dtype=np.float64))
    elif args.offset_rpy_deg is not None:
        r, p, y = [np.deg2rad(v) for v in args.offset_rpy_deg]
        R_tcp_cam = rpy_to_rot(r, p, y)
    else:
        R_tcp_cam = np.eye(3, dtype=np.float64)
    T_tcp_cam = make_T(R_tcp_cam, t_tcp_cam)

    extrinsics: list[np.ndarray] = []
    for fr in frames:
        tcp = fr.get(args.tcp_key)
        if tcp is None:
            raise KeyError(f"{args.tcp_key} not found in frame keys: {list(fr.keys())[:10]}")
        tcp = np.asarray(tcp, dtype=np.float64)
        pos = tcp[:3]
        quat = tcp[3:7]
        R_base_tcp = quat_to_rot(quat)
        T_base_tcp = make_T(R_base_tcp, pos)

        T_base_cam = T_base_tcp @ T_tcp_cam  # cam2world (base)
        extrinsics.append(T_base_cam[:3, :4].reshape(-1))

    extrinsics_arr = np.stack(extrinsics, axis=0)

    # Intrinsics
    h, w = args.resolution_hw
    if args.fx_fy_cx_cy is None:
        fx = fy = float(w)
        cx = float(w) / 2.0
        cy = float(h) / 2.0
    else:
        fx, fy, cx, cy = args.fx_fy_cx_cy
    intrinsics_arr = np.tile(np.array([fx, fy, cx, cy], dtype=np.float64)[None, :], (extrinsics_arr.shape[0], 1))

    cameras_dir = episode_dir / "cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)
    intr_path = cameras_dir / f"intrinsic_{args.view_name}.txt"
    extr_path = cameras_dir / f"extrinsic_{args.view_name}.txt"

    np.savetxt(intr_path, intrinsics_arr, fmt="%.6f")
    np.savetxt(extr_path, extrinsics_arr, fmt="%.6f")

    print(f"Wrote intrinsics to {intr_path}")
    print(f"Wrote extrinsics to {extr_path}")


if __name__ == "__main__":
    main()

