#!/usr/bin/env bash
set -euo pipefail

# Full-parameter post-training (heavier than LoRA).
# Usage:
#   CKPT_PATH=/path/to/model_ema_bf16.pt \
#   IMAGINAIRE_OUTPUT_ROOT=/path/to/output \
#   bash scripts/finetune_robot_multiview_local_full.sh

: "${CKPT_PATH:?Set CKPT_PATH to the base multiview-agibot .pt checkpoint}"

export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}"

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/camera/configs/multiview_camera/config.py -- \
  experiment=multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit \
  "override /data_train=local_robot_multiview_train" \
  "override /data_val=local_robot_multiview_val" \
  model_parallel.context_parallel_size=1 \
  checkpoint.load_path="$CKPT_PATH" \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  checkpoint.save_iter=200 \
  trainer.max_iter=3000 \
  trainer.logging_iter=50 \
  trainer.validation_iter=500 \
  model.config.use_lora=False \
  optimizer.lr=1.0e-5 \
  optimizer.weight_decay=0.1

