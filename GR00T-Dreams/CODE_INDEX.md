# GR00T-Dreams 代码索引

## 仓库作用
GR00T-Dreams 是 NVIDIA Isaac 团队提供的 DreamGen 蓝图：以 Cosmos 视频世界模型为核心，结合 IDM 逆动力学和 GR00T N1 行为策略，形成“生成海量合成轨迹 → 提取动作 → 下游微调与评测”的完整流水线，用于在新场景/新机型上快速提升机器人泛化能力。

## 代码与脚本索引
### scripts（命令行入口）
- `scripts/gr00t_finetune.py`：GR00T N1 策略微调脚本（加载 LeRobot 数据集、组装 transforms、启动 HuggingFace Trainer，可选 LoRA）。
- `scripts/idm_training.py`：IDM 逆动力学模型训练入口，读取 `IDM_dump/base.yaml` 配置，支持随机初始化/多卡。
- `scripts/eval_policy.py`：基于数据集或远程推理服务计算行动 MSE，兼容本地模型或客户端模式。
- `scripts/inference_service.py`：ZeroMQ 推理服务/客户端示例，封装 `Gr00tPolicy` 提供 `get_action`/`get_modality_config`。
- `scripts/load_dataset.py`：读取并可视化 LeRobot 数据集的模态键、形状与示例帧。

### dreamgenbench（评测）
- `dreamgenbench/eval_sr_qwen_whole.py`：使用 Qwen2.5-VL 评估指令跟随成功率。
- `dreamgenbench/eval_sr_gpt4o_whole.py`：用 GPT-4o 评估成功率，支持零样本标志。
- `dreamgenbench/eval_qwen_pa.py`：Qwen-VL 物理一致性（PA）评分。
- `dreamgenbench/utils.py`：评测通用工具（视频读取、提示组装、结果写入）。

### IDM_dump（合成视频转行动序列）
- `IDM_dump/convert_directory.py`：将 Cosmos 生成的单层视频文件夹整理成每个任务一个子目录的结构（供后续处理）。
- `IDM_dump/preprocess_video.py`：按机型（gr1/franka/so100/robocasa）裁剪/分割/缩放视频，导出标准化视角。
- `IDM_dump/raw_to_lerobot.py`：将处理后的视频与动作转存为 LeRobot V2 兼容结构（data/ 与 videos/ 分块）。
- `IDM_dump/dump_idm_actions.py`：调用已训练 IDM 模型，对生成视频推理动作并写回 parquet。
- `IDM_dump/split_video_instruction.py`：把包含多指令的长视频按任务提示拆分。
- `IDM_dump/base.yaml`：Hydra 配置，定义 IDM backbone / action head 默认参数。

### gr00t/data（数据加载与变换）
- `gr00t/data/dataset.py`：LeRobotSingleDataset 读取器，支持视频+状态+动作+语言模态、统计计算与完整性校验。
- `gr00t/data/schema.py`：Pydantic 架构与旋转/模态枚举，定义 `LeRobotModalityMetadata`、`DatasetMetadata` 等。
- `gr00t/data/embodiment_tags.py`：机型枚举（gr1/so100/franka/robocasa/new_embodiment 等）。
- `gr00t/data/transform/base.py`：模态变换基类与组合器。
- `gr00t/data/transform/video.py`：视频变换（tensor 转换、裁剪、缩放、抖动、回写 numpy）。
- `gr00t/data/transform/state_action.py`：状态/动作归一化、正余弦展开、张量化。
- `gr00t/data/transform/concat.py`：将多模态按顺序拼接成模型所需的统一键。

### gr00t/model（模型与头部）
- `gr00t/model/gr00t_n1.py`：GR00T N1 主模型定义，绑定视觉-语言 Backbone 与动作头的接口。
- `gr00t/model/policy.py`：策略封装与 HuggingFace Hub 下载，`Gr00tPolicy` 负责加载元数据、应用/逆应用 transforms、推理动作。
- `gr00t/model/transforms.py`：模型级预处理（地平线、mask、padding 等）以匹配 GR00T 输入格式。
- `gr00t/model/transforms_idm.py`：IDM 训练/推理使用的特定数据预处理。
- `gr00t/model/idm.py`：IDM 配置与模型主体，验证输入形状，调用 action head 输出 action_pred 与 loss。
- `gr00t/model/action_head/flow_matching_action_head.py`：Diffusion/Flow Matching 动作头（含 SigLIP 视觉编码、DiT）。
- `gr00t/model/action_head/flow_matching_action_head_idm.py`：IDM 场景的 Flow Matching 头部实现。
- `gr00t/model/action_head/cross_attention_dit.py`：Cross-Attention DiT 模块，用于时序动作建模。
- `gr00t/model/action_head/multimodal_projector.py`：多模态特征（视觉/语言/状态）映射到统一隐空间。
- `gr00t/model/action_head/siglip/*`：SigLIP 配置、模型、分词与处理器，提供视觉特征提取。
- `gr00t/model/backbone/eagle_backbone.py`：Eagle 视觉-语言 Backbone 裁剪版（去掉 LM 头、按层选择），支持局部权重加载。
- `gr00t/model/backbone/eagle2_hg_model/*`：内嵌 Eagle 模型的配置、推理封装、对话模板与权重调整。
- `gr00t/model/backbone/identity.py`：占位/恒等 Backbone。

### gr00t/experiment（训练组织）
- `gr00t/experiment/data_config.py`：不同机型的数据配置（模态键、时间窗、变换链），用于 GR00T N1 训练/推理。
- `gr00t/experiment/data_config_idm.py`：IDM 使用的数据配置集合。
- `gr00t/experiment/runner.py`：组装 Trainer、Dataset、模型并运行 train/eval。
- `gr00t/experiment/runner_idm.py`：IDM 训练运行器。
- `gr00t/experiment/trainer.py`：自定义 HuggingFace Trainer（DualBrainTrainer），含采样器、优化器分组、断点恢复。

### gr00t/eval（推理服务与封装）
- `gr00t/eval/service.py`：ZeroMQ 通用推理服务器/客户端，实现序列化、endpoint 管理。
- `gr00t/eval/robot.py`：机器人推理专用 Server/Client，暴露 `get_action` 和 `get_modality_config`。
- `gr00t/eval/wrappers/obs_index_selection_wrapper.py`：根据索引裁剪观测。
- `gr00t/eval/wrappers/video_recording_wrapper.py`：在 rollouts 中录制视频。
- `gr00t/eval/wrappers/multistep_wrapper.py`：多步 rollout 包装器。

### gr00t/utils（工具）
- `gr00t/utils/video.py`：视频帧提取、按时间戳采样。
- `gr00t/utils/peft.py`：LoRA 注入与冻结策略工具。
- `gr00t/utils/experiment.py`：保存/加载实验配置与 tokenizer、embedding。
- `gr00t/utils/eval.py`：MSE 评估、动作 unnormalize 等。
- `gr00t/utils/misc.py`：杂项工具（描述函数、随机种子等）。

### getting_started（上手示例与文档）
- `getting_started/*.ipynb`：数据加载、推理、微调、策略部署等教程。
- `getting_started/LeRobot_compatible_data_schema.md`：LeRobot V2 兼容数据格式与 `meta/modality.json` 说明。
- `getting_started/5_policy_deployment.md`、`4_deeper_understanding.md`：部署与深入讲解。
- `getting_started/examples/eval_gr00t_so100.py`、`tictac_bot.py`：示例推理/高层规划+低层执行案例。
- `getting_started/examples/*modality.json`：不同数据集的模态配置样例。

### 其他
- `reference_architecture/reference_architecture.md`：DreamGen 高层架构与数据流。
- `demo_data/intro_vid.mp4`：演示视频。
- `tests/test_dataset.py`：LeRobotSingleDataset 基础测试。
