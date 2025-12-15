# LeRobot 数据集：只用 `low` 单视角微调 Cosmos‑Predict2.5 视频模型（A100 单卡 LoRA）

本文件面向你的真实情况：**数据已经是 LeRobot V2（GR00T LeRobot 兼容）格式**，你只想用其中一个能清楚看到机械臂动作的 **`low` 视角** 做单视角域适配微调 Cosmos‑Predict2.5 视频模型（base 2B）。

输出是一份可直接交给“可执行 agent”逐条跑的流程。

---

## 0. 目标

- **训练目标**：微调 `nvidia/Cosmos-Predict2.5-2B/base/pre-trained`（单视角 Video2World/Image2World）
- **输入数据**：LeRobot dataset 内某一个 view（优先包含 `low` 的 video key）
- **训练方式**：LoRA（单卡 A100 80G，参数与之前一致）
  - `lora_rank=64`, `lora_alpha=64`
  - `lr=3e-5`, `weight_decay=0.1`
  - `max_iter=5000`, `batch_size=1`

> 注意：base 模型不使用 multiview camera condition，所以**不需要相机内外参**。

---

## 1. 你 LeRobot 数据集的典型结构（用于确认）

LeRobot 根目录通常包含（GR00T-Dreams 的说明也一致）：
```
lerobot_root/
  meta/
    episodes.jsonl
    tasks.jsonl
    info.json
    modality.json
  videos/
    chunk-000/
      observation.images.low/episode_000000.mp4
      observation.images.low/episode_000001.mp4
      ...
  data/
    chunk-000/episode_000000.parquet
    ...
```

关键点：我们只需要 `videos/.../*.mp4` + meta 里的 prompt 信息（优先 `episodes.jsonl` 的 `tasks` 字段）。

---

## 2. 环境准备（云端 A100）

```bash
cd cosmos-predict2.5
pip install uv
uv sync --extra=cuda12   # 或 cuda11
ffmpeg -version || sudo apt-get install -y ffmpeg
pip install huggingface_hub
huggingface-cli login
```

---

## 3. 把 LeRobot 的 low 视角导出成 Cosmos 视频微调数据集

### 3.1 转换脚本（已新增）
`scripts/convert_lerobot_to_singleview_video_dataset.py`

输出结构（VideoDataset caption_format="json" 可直接读）：
```
datasets/robot_low_singleview_json/
  train/
    videos/*.mp4
    captions/*.json
  validation/
    videos/*.mp4
    captions/*.json
```

### 3.2 自动选择 low 视角（推荐）
脚本会：
- 优先从 `meta/modality.json` 里找到 video 的 `original_key`
- 若未找到则扫描 `videos/chunk-*/` 下的子目录
- 从候选中选择名称包含 `low` 的那个；否则选第一个

执行：
```bash
python scripts/convert_lerobot_to_singleview_video_dataset.py \
  --lerobot-root /path/to/lerobot_root \
  --output-root datasets/robot_low_singleview_json \
  --prefer-view low \
  --val-ratio 0.05 \
  --link-mode hardlink
```

如果你明确知道目录名（例如 `observation.images.ego_view`），可以强制指定：
```bash
python scripts/convert_lerobot_to_singleview_video_dataset.py \
  --lerobot-root /path/to/lerobot_root \
  --output-root datasets/robot_low_singleview_json \
  --video-dirname observation.images.ego_view \
  --val-ratio 0.05
```

检查输出：
```bash
ls datasets/robot_low_singleview_json/train/videos | head
ls datasets/robot_low_singleview_json/train/captions | head
```

> `--link-mode hardlink` 会尽量用硬链接节省空间；如果失败会自动 fallback 到 copy。

---

## 4. 下载 Cosmos‑Predict2.5 base 2B 权重（EMA BF16）

```bash
mkdir -p checkpoints/predict2_2b_base
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "base/pre-trained/*_ema_bf16.pt" \
  --local-dir checkpoints/predict2_2b_base

export CKPT_PATH=$(find checkpoints/predict2_2b_base -name "*_ema_bf16.pt" | head -n 1)
echo "CKPT_PATH=$CKPT_PATH"
```

---

## 5. 启动单卡 A100 LoRA 微调（命令与参数）

训练脚本：`scripts/finetune_robot_singleview_low_lora.sh`（已更新支持 `DATASET_ROOT`）

```bash
chmod +x scripts/finetune_robot_singleview_low_lora.sh

export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output
export DATASET_ROOT=datasets/robot_low_singleview_json

bash scripts/finetune_robot_singleview_low_lora.sh
```

---

## 6. 导出 pt checkpoint（推理/后续增强用）

```bash
CHECKPOINTS_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_json_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
ls -lh $CHECKPOINT_DIR/*.pt
```

推荐推理用：`$CHECKPOINT_DIR/model_ema_bf16.pt`

---

## 7. 推理验证（单视角）

准备 `assets/my_low.json`（示例）：
```json
{
  "inference_type": "video2world",
  "name": "low_test",
  "prompt": "你的prompt ...",
  "input_path": "path/to/low.mp4"
}
```

运行（覆盖 experiment + checkpoint）：
```bash
python examples/inference.py \
  -i assets/my_low.json \
  -o outputs/robot_low_singleview \
  --experiment predict2_lora_training_2b_cosmos_nemo_assets_json \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt
```

---

## 8. 交给可执行 agent 的“逐条执行清单”

```bash
cd cosmos-predict2.5
pip install uv
uv sync --extra=cuda12
ffmpeg -version || sudo apt-get install -y ffmpeg
pip install huggingface_hub
huggingface-cli login

python scripts/convert_lerobot_to_singleview_video_dataset.py \
  --lerobot-root /path/to/lerobot_root \
  --output-root datasets/robot_low_singleview_json \
  --prefer-view low \
  --val-ratio 0.05 \
  --link-mode hardlink

mkdir -p checkpoints/predict2_2b_base
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "base/pre-trained/*_ema_bf16.pt" \
  --local-dir checkpoints/predict2_2b_base

export CKPT_PATH=$(find checkpoints/predict2_2b_base -name "*_ema_bf16.pt" | head -n 1)
export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output
export DATASET_ROOT=datasets/robot_low_singleview_json

chmod +x scripts/finetune_robot_singleview_low_lora.sh
bash scripts/finetune_robot_singleview_low_lora.sh

CHECKPOINTS_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_json_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

