# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a LeRobot V2 dataset into a single-view (video+caption) dataset for Cosmos-Predict2.5 base post-training.

LeRobot V2 structure (GR00T LeRobot compatible, see GR00T-Dreams docs):
  lerobot_root/
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/info.json
    meta/modality.json
    videos/chunk-000/observation.images.<view>/episode_000000.mp4
    data/chunk-000/episode_000000.parquet

Output structure (compatible with VideoDataset caption_format="json"):
  output_root/
    train/
      videos/*.mp4
      captions/*.json
    validation/
      videos/*.mp4
      captions/*.json

Example:
  python scripts/convert_lerobot_to_singleview_video_dataset.py \
    --lerobot-root /path/to/your_lerobot_dataset \
    --output-root datasets/robot_low_singleview_json \
    --prefer-view low \
    --val-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


EP_RE = re.compile(r"episode_(\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-root", type=str, required=True, help="Path to LeRobot dataset root")
    p.add_argument("--output-root", type=str, required=True, help="Output dataset root")

    p.add_argument(
        "--video-dirname",
        type=str,
        default=None,
        help="Exact folder name under videos/chunk-*/ to use, e.g. observation.images.ego_view",
    )
    p.add_argument(
        "--prefer-view",
        type=str,
        default="low",
        help="If --video-dirname is not set, pick a video key whose name contains this (default: low).",
    )

    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=0, help="For debugging. 0 means no limit.")

    p.add_argument(
        "--link-mode",
        type=str,
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to place mp4 into output dataset.",
    )
    p.add_argument("--caption-model-key", type=str, default="lerobot", help="Top-level key in caption JSON")
    p.add_argument("--default-prompt", type=str, default="", help="Fallback prompt if no prompt found in meta")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_task_index_map(tasks_jsonl: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in read_jsonl(tasks_jsonl):
        if "task_index" in row and "task" in row:
            out[int(row["task_index"])] = str(row["task"])
    return out


def build_episode_prompt_map(episodes_jsonl: Path, task_index_to_task: dict[int, str]) -> dict[int, str]:
    """Best-effort mapping episode_index -> prompt.

    Supports:
      - episodes.jsonl tasks: ["some text prompt", ...]
      - episodes.jsonl tasks: [int_task_index, ...]
    """
    out: dict[int, str] = {}
    for row in read_jsonl(episodes_jsonl):
        if "episode_index" not in row:
            continue
        ep = int(row["episode_index"])
        tasks = row.get("tasks")
        prompt = ""
        if isinstance(tasks, list) and tasks:
            t0 = tasks[0]
            if isinstance(t0, str):
                prompt = t0
            elif isinstance(t0, int):
                prompt = task_index_to_task.get(int(t0), "")
            elif isinstance(t0, dict):
                # Try common patterns
                if "task" in t0 and isinstance(t0["task"], str):
                    prompt = t0["task"]
                elif "task_index" in t0:
                    prompt = task_index_to_task.get(int(t0["task_index"]), "")
        out[ep] = prompt
    return out


def read_modality_video_original_keys(modality_json: Path) -> list[str]:
    if not modality_json.exists():
        return []
    data = json.loads(modality_json.read_text())
    video = data.get("video")
    if not isinstance(video, dict):
        return []
    keys: list[str] = []
    for v in video.values():
        if isinstance(v, dict) and isinstance(v.get("original_key"), str):
            keys.append(v["original_key"])
    return keys


def select_video_dirname(lerobot_root: Path, args: argparse.Namespace) -> str:
    if args.video_dirname:
        return args.video_dirname

    # Prefer modality.json video original_key if present
    modality_json = lerobot_root / "meta" / "modality.json"
    candidates = read_modality_video_original_keys(modality_json)

    # Fallback: scan videos/chunk-*/ subfolders
    if not candidates:
        for p in lerobot_root.glob("videos/chunk-*/*"):
            if p.is_dir():
                candidates.append(p.name)

    if not candidates:
        raise ValueError(f"No video modalities found under {lerobot_root}/videos")

    prefer = (args.prefer_view or "").lower()
    for c in candidates:
        if prefer and prefer in c.lower():
            return c

    # Otherwise just pick the first one deterministically
    return sorted(set(candidates))[0]


def list_episode_videos(lerobot_root: Path, video_dirname: str) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for mp4 in sorted(lerobot_root.glob(f"videos/chunk-*/{video_dirname}/episode_*.mp4")):
        m = EP_RE.search(mp4.name)
        if not m:
            continue
        out.append((int(m.group(1)), mp4))
    return out


def ensure_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.as_posix(), dst.as_posix())
        return
    if mode == "hardlink":
        try:
            os.link(src.as_posix(), dst.as_posix())
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def write_caption_json(path: Path, model_key: str, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {model_key: {"long": prompt, "medium": prompt, "short": prompt}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    lerobot_root = Path(args.lerobot_root)
    output_root = Path(args.output_root)

    video_dirname = select_video_dirname(lerobot_root, args)
    print(f"Using video view: {video_dirname}")

    task_index_to_task = build_task_index_map(lerobot_root / "meta" / "tasks.jsonl")
    episode_to_prompt = build_episode_prompt_map(lerobot_root / "meta" / "episodes.jsonl", task_index_to_task)

    episodes = list_episode_videos(lerobot_root, video_dirname)
    if not episodes:
        raise ValueError(f"No episode videos found under videos/chunk-*/{video_dirname}/episode_*.mp4")

    import random

    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    n_val = int(len(episodes) * args.val_ratio)
    val_set = set(ep for ep, _ in episodes[:n_val])

    stats = {"train": 0, "validation": 0}
    for ep, src_mp4 in episodes:
        split = "validation" if ep in val_set else "train"
        base = f"episode_{ep:06d}"
        dst_mp4 = output_root / split / "videos" / f"{base}.mp4"
        dst_cap = output_root / split / "captions" / f"{base}.json"

        ensure_link(src_mp4, dst_mp4, args.link_mode)
        prompt = episode_to_prompt.get(ep) or args.default_prompt
        write_caption_json(dst_cap, args.caption_model_key, prompt)
        stats[split] += 1

    print("Done.")
    print(json.dumps(stats, indent=2))
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
