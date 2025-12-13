# 本地三视角机器人数据微调 Cosmos‑Predict2.5/robot/multiview‑agibot（A100 单卡版）

本文件提供**可直接照着执行**的端到端流程，用于把你当前的三视角机器人数据（`left/right/top` 或 `left/right/low` 等）转成 Cosmos‑Predict2.5 的本地 multiview‑agibot 训练格式，并在 **1×RTX A100 80G** 上进行 LoRA/全参 post‑training。

> 你可以把本文件交给另一个有命令执行权限的 agent，让它逐条执行即可。

---

## 0. 目标与模型说明

- **目标模型**：`nvidia/Cosmos-Predict2.5-2B/robot/multiview-agibot`
  - 三视角（3-camera view）+ 相机几何条件（Plücker rays）
  - 训练模式为 camera‑conditioned Video2World（同 checkpoint 对应实验名 `multicamera_*_agibot_frameinit`）
- **你当前数据**：
  - 每个 `sub_dataset` 代表一个任务；真实数据中同一个 `sub_dataset` 内所有 episode 共用同一个 prompt。
  - 三个有效视角（例如 `left/right/low` 或 `left/right/top`），逐帧 png。
  - `obs.json` 记录电机/末端 TCP 轨迹（本模型当前**不吃 action**，所以 obs.json 仅做未来扩展）。

---

## 1. 运行环境准备（云端 A100）

### 1.1 硬件
- 1×A100 80G（单卡）
- 推荐 BF16（仓库默认）

### 1.2 软件依赖
在云端机器上执行（进入仓库根目录 `cosmos-predict2.5/`）：

1. **Python & uv**
   ```bash
   python --version  # 建议 3.10/3.11
   pip install uv
   ```

2. **安装 Cosmos‑Predict2.5（带 CUDA extra）**
   > 你本地 import 报的 “CUDA extra not installed” 就是因为没装 extra。
   ```bash
   cd cosmos-predict2.5
   uv sync --extra=cuda12  # 如果你的驱动/torch 对应 cuda12
   # 或者：uv sync --extra=cuda11 视你的环境而定
   ```

3. **ffmpeg**
   转换脚本会调用 ffmpeg。
   ```bash
   ffmpeg -version  # 必须能找到
   # 若无：sudo apt-get install -y ffmpeg
   ```

4. **HuggingFace CLI**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

---

## 2. 原始数据要求

你的原始数据任一任务 `sub_datasetX` 可以是两种结构（下面以 `left/right/low` 为例；如果你还有 `top/`，可以不参与训练或用 `--view-map` 选择其一作为主视角）：

### 2.1 结构 A：直接放三视角
```
fitune_dataset/
  sub_dataset1/
    left/0.png 1.png ...
    right/0.png ...
    low/0.png ...
    obs.json
```

### 2.2 结构 B：任务下多 episode（推荐）
```
fitune_dataset/
  sub_dataset2/
    0000/
      left/*.png right/*.png low/*.png obs.json
    0001/
      ...
```

三视角文件夹名可不同，但需要在转换时用 `--view-map` 映射。

---

## 3. prompt 映射文件

你在 `fitune_dataset/` 下维护一个 json：

`fitune_dataset/prompts.json`
```json
{
  "sub_dataset1": "prompt xxxxx",
  "sub_dataset2": "prompt yyyyy"
}
```

- key 必须等于任务目录名 `sub_dataset*`
- value 是该任务的统一 prompt

---

## 4. 数据转换：png → mp4 + manifest（最关键）

### 4.1 转换脚本
`scripts/convert_local_robot_multiview_dataset.py`

### 4.2 运行命令
在 `cosmos-predict2.5/` 根目录执行：
```bash
python scripts/convert_local_robot_multiview_dataset.py \
  --input-root fitune_dataset \
  --output-root datasets/robot_multiview_local \
  --prompt-map fitune_dataset/prompts.json \
  --view-map left=hand_0,right=hand_1,low=head \
  --fps 16 \
  --val-ratio 0.05
```

参数解释：
- `--input-root`：你的原始数据根目录
- `--output-root`：输出到 Cosmos 训练格式的根目录
- `--prompt-map`：任务 prompt 映射 json
- `--view-map`：**原始视角文件夹名 → 模型三视角名**  
  - 模型内固定三视角名：`head`, `hand_0`, `hand_1`
  - 推荐把你的 **主视角 `low` 映射为 `head`**，例如：  
    `--view-map left=hand_0,right=hand_1,low=head`  
    如果你还有 `top/` 但想用它当主视角，则改为：`--view-map left=hand_0,right=hand_1,top=head`
- `--fps 16`：统一目标 fps（与 agibot 模型一致）
- `--val-ratio`：随机切分验证集比例（按 sample 粒度）

### 4.3 输出检查
执行完后应该得到：
```
datasets/robot_multiview_local/
  train/
    manifest.jsonl
    samples/<sample_id>/
      videos/head.mp4 hand_0.mp4 hand_1.mp4
      cameras/intrinsic_*.txt extrinsic_*.txt
  val/
    manifest.jsonl
    samples/...
```

你可以抽查 manifest 前几行：
```bash
head -n 3 datasets/robot_multiview_local/train/manifest.jsonl
```
每行形如：
```json
{
  "sample_id": "sub_dataset2_0001",
  "prompt": "prompt yyyyy",
  "videos": {"head": ".../head.mp4", "hand_0": ".../hand_0.mp4", "hand_1": ".../hand_1.mp4"},
  "cameras": {"head": {"intrinsic": "...txt", "extrinsic": "...txt"}, ...}
}
```

### 4.4 相机参数说明（强烈建议替换为真实标定）
- 脚本默认会生成 dummy intr/extr，只保证训练形状正确。
- 如果你有真实相机标定：
  - 在每个 episode 目录下放 `cameras/`：
    ```
    cameras/
      intrinsic_head.txt
      extrinsic_head.txt
      intrinsic_hand_0.txt
      extrinsic_hand_0.txt
      intrinsic_hand_1.txt
      extrinsic_hand_1.txt
    ```
  - 格式必须和 AgiBot 一致：
    - `intrinsic_*.txt`：T×4，每行 `fx fy cx cy`
    - `extrinsic_*.txt`：T×12，每行 3×4 **camera‑to‑world** 矩阵（row‑major）
  - 再运行转换时加：`--keep-existing-cameras`

---

## 5. 下载 multiview‑agibot base checkpoint

在云端下载 `.pt` EMA checkpoint：
```bash
mkdir -p checkpoints/robot_multiview_agibot
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "f740321e-*/model_ema_bf16.pt" \
  --local-dir checkpoints/robot_multiview_agibot
```

把实际文件路径记为：
`CKPT_PATH=/absolute/path/to/model_ema_bf16.pt`

---

## 6. LoRA 微调（推荐）

### 6.1 运行脚本
`scripts/finetune_robot_multiview_local_lora.sh`

```bash
chmod +x scripts/finetune_robot_multiview_local_lora.sh

export CKPT_PATH=/abs/path/to/model_ema_bf16.pt
export IMAGINAIRE_OUTPUT_ROOT=/abs/path/to/output_root

bash scripts/finetune_robot_multiview_local_lora.sh
```

### 6.2 默认超参（脚本内）
- batch_size=1（单卡 80G 最稳）
- context_parallel_size=1（避免 CP 导致额外显存开销）
- LoRA rank=64 / alpha=64
- lr=3e‑5, weight_decay=0.1
- max_iter=5000（小规模数据起步）

### 6.3 如何按数据量调整
| 数据规模（train sample 数） | 建议 max_iter | lr |
| --- | --- | --- |
| < 200 | 3k‑5k | 3e‑5 |
| 200‑2000 | 10k‑20k | 3e‑5 |
| > 2000 | 30k‑40k | 2e‑5‑3e‑5 |

修改方式：直接编辑脚本或 CLI 覆盖，例如：
```bash
bash scripts/finetune_robot_multiview_local_lora.sh \
  -- trainer.max_iter=20000 optimizer.lr=2.0e-5
```

---

## 7. 全参微调（可选，数据很大时用）

脚本：`scripts/finetune_robot_multiview_local_full.sh`

```bash
chmod +x scripts/finetune_robot_multiview_local_full.sh

export CKPT_PATH=/abs/path/to/model_ema_bf16.pt
export IMAGINAIRE_OUTPUT_ROOT=/abs/path/to/output_root

bash scripts/finetune_robot_multiview_local_full.sh
```

默认 lr=1e‑5，max_iter=3000 起步；数据大时可以到 1‑2 万 iter。

---

## 8. 导出 pt checkpoint

训练输出在：
`$IMAGINAIRE_OUTPUT_ROOT/cosmos_diffusion_v2/.../checkpoints/iter_xxxxx/model/`

用官方脚本转 pt：
```bash
CHECKPOINT_DIR=/abs/path/to/checkpoints/iter_00000xxxx
python scripts/convert_distcp_to_pt.py \
  $CHECKPOINT_DIR/model \
  $CHECKPOINT_DIR
```

得到：
- `model.pt`
- `model_ema_fp32.pt`
- `model_ema_bf16.pt`（推理推荐）

---

## 9. 推理验证

准备一个推理 json（参考官方 `assets/robot_multiview-agibot/*.json`），或复用你已有的 base‑path 格式。
注意：当前仓库已兼容把主视角命名为 `low`（即允许使用 `{input_name}_low.png`、`{input_name}_intrinsic_low.txt`、`{input_name}_extrinsic_low.txt` 来替代 `head`）。

单卡推理：
```bash
python examples/robot_multiview.py \
  -i /path/to/your_infer.json \
  --base-path=/path/to/your/base_path \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  -o outputs/robot_multiview_local/
```

---

## 10. 常见问题

1. **报 CUDA extra not installed**
   ```bash
   uv sync --extra=cuda12
   ```

2. **ffmpeg 找不到**
   ```bash
   sudo apt-get install -y ffmpeg
   ```

3. **显存 OOM**
   - 确认：
     - `batch_size=1`
     - `model_parallel.context_parallel_size=1`
   - 仍不够时：
     - 把 `num_frames_per_view` 降到 61（需要同时改转换脚本输出和 dataloader 参数）

4. **多视角不同步/长度不一致**
   - 转换脚本会按三视角最短长度裁剪；如果你需要更严格同步，请在原始数据保证帧数一致。

---

## 11. 给 sandbox agent 的“逐条执行清单”

按顺序复制执行：

```bash
# 1) 环境
cd cosmos-predict2.5
pip install uv
uv sync --extra=cuda12
sudo apt-get install -y ffmpeg
pip install huggingface_hub
huggingface-cli login

# 2) 数据转换
python scripts/convert_local_robot_multiview_dataset.py \
  --input-root fitune_dataset \
  --output-root datasets/robot_multiview_local \
  --prompt-map fitune_dataset/prompts.json \
  --view-map left=hand_0,right=hand_1,low=head \
  --fps 16 \
  --val-ratio 0.05

head -n 3 datasets/robot_multiview_local/train/manifest.jsonl

# 3) 下权重
mkdir -p checkpoints/robot_multiview_agibot
hf download nvidia/Cosmos-Predict2.5-2B \
  --include "f740321e-*/model_ema_bf16.pt" \
  --local-dir checkpoints/robot_multiview_agibot

export CKPT_PATH=$(find checkpoints/robot_multiview_agibot -name model_ema_bf16.pt | head -n 1)
export IMAGINAIRE_OUTPUT_ROOT=/tmp/imaginaire4-output

# 4) LoRA 微调
chmod +x scripts/finetune_robot_multiview_local_lora.sh
bash scripts/finetune_robot_multiview_local_lora.sh

# 5) 导出 pt
CHECKPOINTS_DIR=$IMAGINAIRE_OUTPUT_ROOT/cosmos_diffusion_v2/cosmos_predict_v2p5/multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

# 6) 推理（替换 infer.json/base_path）
python examples/robot_multiview.py \
  -i /path/to/your_infer.json \
  --base-path=/path/to/your/base_path \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  -o outputs/robot_multiview_local/
```
