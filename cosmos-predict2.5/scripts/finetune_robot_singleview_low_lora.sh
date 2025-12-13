#!/usr/bin/env bash
set -euo pipefail

# Single-view LoRA post-training for Cosmos-Predict2.5 base (2B) on robot "low" view videos.
#
# Prereq: run scripts/convert_local_robot_singleview_dataset.py to create:
#   datasets/robot_low_singleview_json/train/{videos,captions}
#   datasets/robot_low_singleview_json/validation/{videos,captions}
#
# Usage:
#   CKPT_PATH=/path/to/base/pre-trained/*_ema_bf16.pt \
#   IMAGINAIRE_OUTPUT_ROOT=/path/to/output \
#   bash scripts/finetune_robot_singleview_low_lora.sh

: "${CKPT_PATH:?Set CKPT_PATH to the Cosmos-Predict2.5-2B base/pre-trained ema bf16 .pt checkpoint}"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}"

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \
  experiment=predict2_lora_training_2b_cosmos_nemo_assets_json \
  dataloader_train.dataset.dataset_dir="datasets/robot_low_singleview_json/train" \
  dataloader_val.dataset.dataset_dir="datasets/robot_low_singleview_json/validation" \
  checkpoint.load_path="$CKPT_PATH" \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  checkpoint.save_iter=200 \
  trainer.max_iter=5000 \
  trainer.logging_iter=50 \
  trainer.validation_iter=500 \
  model_parallel.context_parallel_size=1 \
  model.config.use_lora=True \
  model.config.lora_rank=64 \
  model.config.lora_alpha=64 \
  model.config.init_lora_weights=True \
  optimizer.lr=3.0e-5 \
  optimizer.weight_decay=0.1

