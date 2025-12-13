# 只用 `low` 单视角微调 Cosmos‑Predict2.5 视频模型（A100 单卡 LoRA）

本文件给出**可直接照着执行**的端到端流程：只取你数据中的 `low/` 视角（能清楚看到机械臂动作），把 png 序列转换为单视角 MP4 数据集，然后在 **1×RTX A100 80G** 上对 **Cosmos‑Predict2.5‑2B base/pre‑trained** 进行 LoRA 微调。

> 你可以把本文件交给一个有命令执行权限的 agent，让它逐条执行。

---

## 0. 你要微调的是什么？

- **模型**：`nvidia/Cosmos-Predict2.5-2B/base/pre-trained`
- **任务**：单视角视频生成/续写（Video2World / Image2World）
- **本流程特性**：
  - **不使用多视角**（不需要 `left/right/top`）
  - **不需要相机内外参**（base 模型不吃 camera condition）
  - 只用你的 `low` 视角视频对模型做域适配（robot domain + 机位视角）

---

## 1. 云端环境准备（A100）

在云端机器执行（进入仓库根目录 `cosmos-predict2.5/`）：

```bash
cd cosmos-predict2.5

# 1) uv
pip install uv

# 2) 安装 CUDA extra（按你的环境选 cuda12 或 cuda11）
uv sync --extra=cuda12

# 3) ffmpeg（用于 png -> mp4）
ffmpeg -version || sudo apt-get install -y ffmpeg

# 4) HuggingFace CLI（下载权重）
pip install huggingface_hub
huggingface-cli login
```

---

## 2. 原始数据要求（只看 low）

你的原始数据可以是两种结构之一（都支持）：

### 2.1 结构 A：任务下直接放 low
```
fitune_dataset/
  sub_dataset1/
    low/0.png 1.png ...
    obs.json
```

### 2.2 结构 B：任务下多 episode（推荐）
```
fitune_dataset/
  sub_dataset2/
    0000/low/*.png obs.json
    0001/low/*.png obs.json
```

> 本流程只使用 `low/`，其它视角目录会被忽略。

---

## 3. prompt 映射文件（sub_dataset -> prompt）

你在 `fitune_dataset/` 下维护：

`fitune_dataset/prompts.json`
```json
{
  "sub_dataset1": "prompt xxxxx",
  "sub_dataset2": "prompt yyyyy"
}
```

- key 必须等于任务目录名（`sub_dataset*`）
- value 是该任务的统一 prompt（你说真实数据中同一个 sub_dataset 共享 prompt）

---

## 4. 数据转换：low png 序列 → 单视角训练集（videos+captions）

### 4.1 转换脚本
`scripts/convert_local_robot_singleview_dataset.py`

它会生成一个 VideoDataset 可直接读取的数据集（JSON captions 模式）：
```
datasets/robot_low_singleview_json/
  train/
    videos/*.mp4
    captions/*.json
  validation/
    videos/*.mp4
    captions/*.json
```

### 4.2 运行命令
```bash
python scripts/convert_local_robot_singleview_dataset.py \
  --input-root fitune_dataset \
  --output-root datasets/robot_low_singleview_json \
  --prompt-map fitune_dataset/prompts.json \
  --view low \
  --fps 16 \
  --min-frames 93 \
  --val-ratio 0.05
```

关键参数：
- `--view low`：只取 low 视角
- `--min-frames 93`：少于 93 帧的视频会被跳过（因为训练默认采样 93 帧窗口）

检查输出：
```bash
ls datasets/robot_low_singleview_json/train/videos | head
ls datasets/robot_low_singleview_json/train/captions | head
```

---

## 5. 下载 base/pre-trained 2B 权重（EMA BF16）

```bash
mkdir -p checkpoints/predict2_2b_base
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "base/pre-trained/*_ema_bf16.pt" \
  --local-dir checkpoints/predict2_2b_base

export CKPT_PATH=$(find checkpoints/predict2_2b_base -name "*_ema_bf16.pt" | head -n 1)
echo "CKPT_PATH=$CKPT_PATH"
```

---

## 6. 单卡 A100 LoRA 微调（参数与之前一致）

### 6.1 训练脚本
`scripts/finetune_robot_singleview_low_lora.sh`

```bash
chmod +x scripts/finetune_robot_singleview_low_lora.sh

export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output
bash scripts/finetune_robot_singleview_low_lora.sh
```

### 6.2 默认训练超参（脚本内）
- `use_lora=True`
- `lora_rank=64`, `lora_alpha=64`
- `lr=3e-5`, `weight_decay=0.1`
- `max_iter=5000`
- `logging_iter=50`, `validation_iter=500`
- `context_parallel_size=1`
- 使用 `experiment=predict2_lora_training_2b_cosmos_nemo_assets_json`，但把 dataset_dir 覆盖为你的数据集目录

---

## 7. 导出 pt checkpoint（推理用）

训练输出是 DCP 目录，转换为 `.pt`：

```bash
# 通常输出在：
# $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_json_lora/checkpoints
# 如果你的环境里目录不同，可用下面的 find 自动定位 latest_checkpoint.txt：
# find $IMAGINAIRE_OUTPUT_ROOT -name latest_checkpoint.txt | head

CHECKPOINTS_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_json_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
ls -lh $CHECKPOINT_DIR/*.pt
```

推荐推理用：`$CHECKPOINT_DIR/model_ema_bf16.pt`

---

## 8. 推理验证（单视角）

用 base 推理脚本：`examples/inference.py`

准备一个 json（参考 `docs/inference.md`），例如 `assets/my_low.json`：
```json
{
  "inference_type": "video2world",
  "name": "my_low_test",
  "prompt": "你的prompt ...",
  "input_path": "path/to/low.mp4"
}
```

运行（关键是覆盖 experiment + checkpoint）：
```bash
python examples/inference.py \
  -i assets/my_low.json \
  -o outputs/robot_low_singleview \
  --experiment predict2_lora_training_2b_cosmos_nemo_assets_json \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt
```

---

## 9. 给执行型 agent 的“逐条执行清单”

```bash
cd cosmos-predict2.5
pip install uv
uv sync --extra=cuda12
ffmpeg -version || sudo apt-get install -y ffmpeg
pip install huggingface_hub
huggingface-cli login

python scripts/convert_local_robot_singleview_dataset.py \
  --input-root fitune_dataset \
  --output-root datasets/robot_low_singleview_json \
  --prompt-map fitune_dataset/prompts.json \
  --view low \
  --fps 16 \
  --min-frames 93 \
  --val-ratio 0.05

mkdir -p checkpoints/predict2_2b_base
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "base/pre-trained/*_ema_bf16.pt" \
  --local-dir checkpoints/predict2_2b_base

export CKPT_PATH=$(find checkpoints/predict2_2b_base -name "*_ema_bf16.pt" | head -n 1)
export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output

chmod +x scripts/finetune_robot_singleview_low_lora.sh
bash scripts/finetune_robot_singleview_low_lora.sh

CHECKPOINTS_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/lora/2b_cosmos_nemo_assets_json_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

---

## 10. 下一步：接入你的“后续数据增强流程”

你完成单视角模型微调后，后续数据增强流程通常就是：
- 用 `examples/inference.py` 批量对你的任务 prompt + 初始帧/短视频生成更多 low 视角视频
- 再把生成的视频/帧落盘到你自己的 augmentation pipeline（例如合成更多 episode、或再做 VLA 数据扩增）

如果你希望我把“批量推理 + 输出组织成你现有数据集结构”的脚本也补齐（含和 `sub_dataset -> prompt` 对应），告诉我你期望的输出目录结构即可。
