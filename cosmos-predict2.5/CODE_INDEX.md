# cosmos-predict2.5 代码索引

## 仓库作用
- Cosmos-Predict2.5 是 NVIDIA 开源的物理世界生成与预测模型（Text2World / Image2World / Video2World / 自动驾驶多视角 / 机器人多视角与动作条件），提供推理、后训练（含 LoRA）、多模态数据管线及评测工具。
- 主要组件：核心模型与训练管线（`cosmos_predict2`）、训练/数据处理脚本（`scripts/`）、示例（`examples/`）、文档（`docs/`）、扩展包（`packages/`）。

## 顶层目录与关键文件
- `README.md`：项目介绍、模型家族与快速链接。
- `docs/`：安装、推理、后训练（含 LoRA/多视角/机器人动作）、疑难排查、理论说明。
- `examples/`：推理与后训练示例脚本与 Notebook。
- `scripts/`：数据预处理、训练入口、格式转换、Gradio 启动等实用脚本。
- `cosmos_predict2/`：核心 Python 包，包含训练/推理配置、实验定义、Gradio、内部实现。
- `packages/`：附加子包（`cosmos-cuda`、`cosmos-gradio`、`cosmos-oss`）及其 Docker/文档。
- `assets/`：示例配置与资源。
- `tests/`、`test_data/`：单测与测试数据。
- 其他：`docker/`（容器支持）、`bin/`（可执行脚本）、`uv.lock`/`pyproject.toml`（依赖管理）、`custom_zero_action_loader.py`（自定义零动作加载示例）。

## 核心 Python 包：cosmos_predict2/
- `config.py` / `config_test.py` / `action_conditioned_config.py` / `multiview_config.py` / `robot_multiview_config.py`：不同任务与模式的默认配置入口。
- `inference.py` / `action_conditioned.py` / `multiview.py` / `robot_multiview.py`：推理主入口，封装加载模型、解析 JSON 配置、输出视频的流程。
- `experiments/`：实验配置模板
  - `base/`：通用实验定义（如 `cosmos_nemo_assets.py`、`cosmos_nemo_assets_lora.py`、`groot.py`、`action.py`）。
  - `multiview/waymo.py`：Waymo 多视角实验配置。
- `gradio/`：Gradio UI 的模型配置与 worker（`video2world_worker.py`、`multiview_worker.py`、`gradio_bootstrapper.py`）。

### 训练与基础设施：`cosmos_predict2/_src/imaginaire/`
- 通用训练框架（Hydra 配置、Trainer、Checkpoint、Lazy Config）。
- `datasets/`：数据管线基类、数据源装载、增广器、重放缓存、WebDataset 等。
- `models/` / `modules/` / `functional/`：基础模块（损失、归一化、调度器、视觉/文本组件）与可组合层。
- `trainer.py` / `checkpointer/` / `callbacks/` / `visualize/`：训练循环、权重存取、日志回调与可视化。
- `utils/`：分布式、随机数、Profiling 等通用工具。

### 核心模型实现：`cosmos_predict2/_src/predict2/`
- `configs/`：Text2World、Video2World 等默认配置。
- `datasets/`：视频数据加载（本地/分布式）、增广、分片与 Caption 处理；`local_datasets/dataset_video.py` 处理本地 MP4+prompt。
- `models/` / `modules/` / `networks/`：预测模型、Diffusion/Flow 网络、Transformer 块、UNet 类组件。
- `conditioner.py` / `camera/`：条件信号、相机参数与多模态对齐。
- `text_encoders/` / `tokenizers/`：文本编码器（含 Reason1）、分词与提示处理。
- `inference/` / `interactive/`：推理流水线、交互式接口。
- `action/` / `distill/` / `schedulers/`：动作建模、蒸馏、采样调度。
- `utils/` / `tests/`：辅助函数与单测。

### 自动驾驶多视角：`cosmos_predict2/_src/predict2_multiview/`
- 针对 Waymo 等多摄像头输入的配置、数据加载与模型定义。
- `datasets/`（含 `local.py`、caption 概率控制）、`networks/`、`models/`、`callbacks/`、`scripts/`（多视角训练/推理入口）。

### 文本编码器：`cosmos_predict2/_src/reason1/`
- Reason1/VLM 相关配置、模型、并行策略与分词工具。

## 脚本与工具：`scripts/`
- `train.py`：主训练入口（调用 Hydra/imaginaire 管线，支持单/多卡）。
- `convert_distcp_to_pt.py`：将分布式检查点转换为合并的 PyTorch 权重（推理或 LoRA/后训练后使用）。
- `create_prompts_for_gr1_dataset.py` / `create_prompts_for_nemo_assets.py`：为视频数据生成文本提示文件。
- `convert_waymo.py` / `download_waymo.sh`：下载并转换 Waymo 多视角数据。
- `prepare_*` / `extract_images_from_videos.py` / `prepare_batch_input_json.py`：数据准备与批量推理 JSON 生成。
- `check_environment.py`：环境自检。
- `run_gradio.sh`：启动 Gradio Demo。

## 示例：`examples/`
- `inference.py` / `action_conditioned.py` / `multiview.py` / `robot_multiview.py`：不同模式的推理脚本。
- `posttraining/` / `notebook/`：训练与推理的 Notebook/脚本示例。

## 附加子包：`packages/`
- `cosmos-cuda/`：CUDA 相关扩展与构建脚本。
- `cosmos-gradio/`：独立的 Gradio 前端封装。
- `cosmos-oss/`：开放式工具与文档（含自己的 `docs/`、`scripts/`、`vqa/` 等）。

## 测试与数据
- `tests/`：针对核心数据集与功能的单测。
- `test_data/` / `assets/`：示例输入、配置与演示资源。

## 其他
- `custom_zero_action_loader.py`：示例自定义动作加载函数，生成零动作用于无真实动作的测试。
- `docker/` / `Dockerfile`：容器化支持。
- `justfile` / `pyrefly*.toml`：任务与工具链配置。
