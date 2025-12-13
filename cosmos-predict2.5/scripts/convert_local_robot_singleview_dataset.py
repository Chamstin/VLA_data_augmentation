# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert your robot dataset into a single-view local dataset for Cosmos-Predict2.5 base post-training.

This script ONLY uses one view folder (default: "low") and produces a dataset compatible with
`cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video.VideoDataset` in JSON-caption mode:

output_root/
  train/
    videos/*.mp4
    captions/*.json
  validation/
    videos/*.mp4
    captions/*.json

Input root layout (your raw data):
  fitune_dataset/
    sub_datasetA/
      low/0.png ...
      obs.json
    sub_datasetB/
      0000/low/*.png obs.json
      0001/...

Prompt mapping:
  fitune_dataset/prompts.json
  {
    "sub_datasetA": "same prompt for this task",
    "sub_datasetB": "same prompt for this task"
  }

Example:
  python scripts/convert_local_robot_singleview_dataset.py \
    --input-root fitune_dataset \
    --output-root datasets/robot_low_singleview_json \
    --prompt-map fitune_dataset/prompts.json \
    --view low \
    --fps 16 \
    --min-frames 93 \
    --val-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RawSample:
    sample_id: str
    sub_dataset: str
    root: Path  # directory that contains the view folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=str, required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--prompt-map", type=str, required=True, help="JSON mapping sub_dataset -> prompt")
    p.add_argument("--view", type=str, default="low", help="Single view folder name to use (default: low)")
    p.add_argument("--fps", type=int, default=16, help="MP4 fps for encoding")
    p.add_argument("--min-frames", type=int, default=93, help="Skip samples with fewer frames than this")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--caption-model-key", type=str, default="manual", help="Top-level key in caption JSON")
    return p.parse_args()


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


def write_caption_json(path: Path, model_key: str, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {model_key: {"long": prompt, "medium": prompt, "short": prompt}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg before running this script.")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    prompt_map: Dict[str, str] = json.loads(Path(args.prompt_map).read_text())

    samples = discover_samples(input_root)
    if not samples:
        raise ValueError(f"No samples found under {input_root}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_ids = set(s.sample_id for s in samples[:n_val])

    stats = {"train": 0, "validation": 0, "skipped_short": 0, "skipped_missing_view": 0}

    for s in samples:
        split = "validation" if s.sample_id in val_ids else "train"
        view_dir = s.root / args.view
        if not view_dir.exists():
            stats["skipped_missing_view"] += 1
            continue

        frames = list_frame_files(view_dir)
        if len(frames) < args.min_frames:
            stats["skipped_short"] += 1
            continue

        out_videos_dir = output_root / split / "videos"
        out_captions_dir = output_root / split / "captions"
        out_mp4 = out_videos_dir / f"{s.sample_id}.mp4"
        out_caption = out_captions_dir / f"{s.sample_id}.json"

        encode_mp4_from_frames(frames, out_mp4, args.fps)
        prompt = prompt_map.get(s.sub_dataset, "")
        write_caption_json(out_caption, args.caption_model_key, prompt)

        stats[split] += 1

    print("Done.")
    print(json.dumps(stats, indent=2))
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()

