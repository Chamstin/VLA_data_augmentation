# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert your 3‑view robot dataset (png sequences + obs.json) into a local
camera‑conditioned multiview dataset for Cosmos‑Predict2.5 robot/multiview‑agibot post‑training.

Input root layout (your raw data):
  fitune_dataset/
    sub_datasetA/
      left/0.png ...
      right/0.png ...
      low/0.png ...   # recommended as "main" view
      top/0.png ...   # optional (ignored unless you map it)
      obs.json
    sub_datasetB/
      0000/left/*.png ...
      0000/right/*.png ...
      0000/low/*.png ...
      0000/obs.json
      0001/...

This script:
  1) selects 3 views according to --view-map
  2) encodes each view’s frame sequence into an mp4
  3) generates dummy camera intr/extr files if none exist
  4) writes train/val manifests expected by LocalRobotMultiviewCameraDataset

Example:
  python scripts/convert_local_robot_multiview_dataset.py \
    --input-root fitune_dataset \
    --output-root datasets/robot_multiview_local \
    --prompt-map fitune_dataset/prompts.json \
    --view-map left=hand_0,right=hand_1,low=head \
    --fps 16 --val-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class RawSample:
    sample_id: str
    sub_dataset: str
    root: Path  # directory that contains view folders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=str, required=True, help="Raw dataset root (e.g., fitune_dataset)")
    p.add_argument("--output-root", type=str, required=True, help="Processed dataset root")
    p.add_argument(
        "--prompt-map",
        type=str,
        default=None,
        help="JSON mapping sub_dataset -> prompt text",
    )
    p.add_argument(
        "--view-map",
        type=str,
        default="left=hand_0,right=hand_1,low=head",
        help="Mapping from raw view folder to output view name, comma separated",
    )
    p.add_argument("--fps", type=int, default=16, help="Target fps for mp4 encoding")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=0, help="Random seed for split")
    p.add_argument("--resolution-hw", type=int, nargs=2, default=(704, 1280), help="H W for dummy intrinsics")
    p.add_argument(
        "--keep-existing-cameras",
        action="store_true",
        help="If raw sample has cameras/ folder, copy it instead of dummy.",
    )
    return p.parse_args()


def parse_view_map(view_map_str: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in view_map_str.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        mapping[k.strip()] = v.strip()
    return mapping


def discover_samples(input_root: Path) -> List[RawSample]:
    samples: List[RawSample] = []
    for sub in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        episode_dirs = [d for d in sub.iterdir() if d.is_dir() and d.name.isdigit()]
        if episode_dirs:
            for ep in sorted(episode_dirs):
                samples.append(RawSample(sample_id=f"{sub.name}_{ep.name}", sub_dataset=sub.name, root=ep))
        else:
            samples.append(RawSample(sample_id=sub.name, sub_dataset=sub.name, root=sub))
    return samples


def list_frame_files(view_dir: Path) -> List[Path]:
    files = list(view_dir.glob("*.png")) + list(view_dir.glob("*.jpg")) + list(view_dir.glob("*.jpeg"))
    def _key(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.name
    return sorted(files, key=_key)


def encode_mp4_from_frames(frames: List[Path], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) == 0:
        raise ValueError(f"No frames to encode for {out_path}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in frames:
            f.write(f"file '{p.as_posix()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path.as_posix(),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        Path(list_file).unlink(missing_ok=True)


def write_dummy_cameras(cameras_dir: Path, views: List[str], latent_frames: int, resolution_hw: Tuple[int, int]) -> Dict[str, Dict[str, str]]:
    cameras_dir.mkdir(parents=True, exist_ok=True)
    h, w = resolution_hw
    fx = fy = float(w)
    cx = float(w) / 2.0
    cy = float(h) / 2.0

    # simple offsets to distinguish views
    offsets = {
        "head": (0.0, 0.0, 0.0),
        "hand_0": (0.10, 0.0, 0.0),
        "hand_1": (-0.10, 0.0, 0.0),
    }

    cam_paths: Dict[str, Dict[str, str]] = {}
    for v in views:
        tx, ty, tz = offsets.get(v, (0.0, 0.0, 0.0))
        intrinsic_path = cameras_dir / f"intrinsic_{v}.txt"
        extrinsic_path = cameras_dir / f"extrinsic_{v}.txt"

        with intrinsic_path.open("w") as f:
            for _ in range(latent_frames):
                f.write(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

        with extrinsic_path.open("w") as f:
            row = [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz]
            line = " ".join(f"{x:.6f}" for x in row)
            for _ in range(latent_frames):
                f.write(line + "\n")

        cam_paths[v] = {"intrinsic": intrinsic_path.as_posix(), "extrinsic": extrinsic_path.as_posix()}
    return cam_paths


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    view_map = parse_view_map(args.view_map)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg before running this script.")

    prompt_map: Dict[str, str] = {}
    if args.prompt_map:
        prompt_map = json.loads(Path(args.prompt_map).read_text())

    samples = discover_samples(input_root)
    if len(samples) == 0:
        raise ValueError(f"No samples found under {input_root}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_ids = set(s.sample_id for s in samples[:n_val])

    train_entries: List[dict] = []
    val_entries: List[dict] = []

    for s in samples:
        split = "val" if s.sample_id in val_ids else "train"
        out_sample_dir = output_root / split / "samples" / s.sample_id
        videos_dir = out_sample_dir / "videos"
        cameras_dir = out_sample_dir / "cameras"

        videos: Dict[str, str] = {}
        frames_per_view: Dict[str, List[Path]] = {}
        for raw_view, out_view in view_map.items():
            raw_view_dir = s.root / raw_view
            if not raw_view_dir.exists():
                raise FileNotFoundError(f"Missing view folder {raw_view_dir} for sample {s.sample_id}")
            frames = list_frame_files(raw_view_dir)
            frames_per_view[out_view] = frames

        min_len = min(len(frames) for frames in frames_per_view.values())
        if min_len == 0:
            continue

        for out_view, frames in frames_per_view.items():
            frames = frames[:min_len]
            out_mp4 = videos_dir / f"{out_view}.mp4"
            encode_mp4_from_frames(frames, out_mp4, args.fps)
            videos[out_view] = out_mp4.as_posix()

        # cameras: copy if provided, else dummy
        cam_paths: Dict[str, Dict[str, str]] = {}
        raw_cameras_dir = s.root / "cameras"
        if args.keep_existing_cameras and raw_cameras_dir.exists():
            shutil.copytree(raw_cameras_dir, cameras_dir, dirs_exist_ok=True)
            for out_view in frames_per_view.keys():
                cam_paths[out_view] = {
                    "intrinsic": (cameras_dir / f"intrinsic_{out_view}.txt").as_posix(),
                    "extrinsic": (cameras_dir / f"extrinsic_{out_view}.txt").as_posix(),
                }
        else:
            latent_frames = min_len // 4 + 1
            cam_paths = write_dummy_cameras(
                cameras_dir=cameras_dir,
                views=list(frames_per_view.keys()),
                latent_frames=latent_frames,
                resolution_hw=tuple(args.resolution_hw),
            )

        prompt = prompt_map.get(s.sub_dataset, "")

        entry = {
            "sample_id": s.sample_id,
            "prompt": prompt,
            "videos": videos,
            "cameras": cam_paths,
        }
        if split == "train":
            train_entries.append(entry)
        else:
            val_entries.append(entry)

    # Write manifests
    for split, entries in [("train", train_entries), ("val", val_entries)]:
        manifest_path = output_root / split / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {len(entries)} samples to {manifest_path}")


if __name__ == "__main__":
    main()
