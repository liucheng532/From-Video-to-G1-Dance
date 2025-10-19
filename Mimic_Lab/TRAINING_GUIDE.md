# G1 机器人舞蹈数据训练指南

本指南将帮助您使用 Robocup_Lab 在 IsaacLab 中训练 G1 机器人的舞蹈动作。

## 📋 前置条件

- ✅ IsaacLab v2.1.0 已安装（conda 环境）
- ✅ unitree_description 文件已存在
- ✅ 舞蹈数据文件：`dance_motion/douyin1_g1.csv`

## 🚀 训练步骤

### 步骤 1: 激活 IsaacLab 环境并安装 robocup_lab

```bash
# 激活您的 IsaacLab conda 环境（根据您的环境名称修改）
conda activate isaaclab

# 进入 Robocup_Lab 目录
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab

# 安装 robocup_lab 包
python -m pip install -e source/robocup_lab
```

### 步骤 2: 配置 WandB (Weights & Biases)

该项目使用 WandB Registry 来管理参考动作数据。

```bash
# 登录 WandB（如果还没登录）
wandb login

# 设置您的 WandB 组织名称（不是个人用户名）
export WANDB_ENTITY=your-organization-name
```

**重要**: 
- 在 WandB 网站上创建一个新的 Registry Collection
- Collection 名称: "Motions"
- Artifact 类型: "All Types"

### 步骤 3: 将舞蹈 CSV 数据转换为 NPZ 格式

这一步会将您的舞蹈数据转换为训练所需的格式，并自动上传到 WandB Registry。

```bash
# 转换舞蹈数据
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --input_fps 30 \
  --output_name douyin1_dance \
  --robot g1 \
  --headless
```

**参数说明**:
- `--input_file`: 输入的 CSV 文件路径
- `--input_fps`: 输入动作的帧率（默认 30）
- `--output_name`: 输出的动作名称（将用于 WandB Registry）
- `--robot`: 机器人类型（g1 或 booster_t1）
- `--headless`: 无头模式运行（不显示可视化窗口）
- `--output_fps`: 输出帧率（默认 50）
- `--frame_range`: 可选，指定帧范围，如 `--frame_range 0 500`

### 步骤 4: 验证动作数据（可选）

在训练前，您可以在 Isaac Sim 中重播动作来验证数据是否正确：

```bash
python scripts/replay_npz.py \
  --registry_name=${WANDB_ENTITY}-org/wandb-registry-motions/douyin1_dance
```

### 步骤 5: 训练动作跟踪策略

现在开始训练！这会训练一个策略让 G1 机器人学会跟踪您的舞蹈动作。

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name ${WANDB_ENTITY}-org/wandb-registry-motions/douyin1_dance \
  --headless \
  --logger wandb \
  --log_project_name g1_dance_training \
  --run_name douyin1_dance_v1 \
  --num_envs 4096
```

**参数说明**:
- `--task`: 任务类型（G1 机器人在平地上的动作跟踪）
- `--registry_name`: WandB Registry 中的动作路径
- `--headless`: 无头模式（在服务器上训练时使用）
- `--logger`: 使用 WandB 记录训练过程
- `--log_project_name`: WandB 项目名称
- `--run_name`: 本次训练的运行名称
- `--num_envs`: 并行环境数量（根据您的 GPU 内存调整）
- `--max_iterations`: 最大训练迭代次数（可选，默认值在配置中）

### 步骤 6: 评估训练好的策略

训练完成后，您可以测试训练好的策略：

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs=2 \
  --wandb_path=${WANDB_ENTITY}/g1_dance_training/xxxxxxxx
```

**注意**: 
- `wandb_path` 格式为：`{组织名}/{项目名}/{8位运行ID}`
- 运行 ID 可以在 WandB 网站的运行概览中找到

## 🎨 高级选项

### 调整训练参数

如果需要自定义训练参数，可以修改以下文件：
- **环境配置**: `source/robocup_lab/robocup_lab/tasks/tracking/config/g1/flat_env_cfg.py`
- **PPO 超参数**: `source/robocup_lab/robocup_lab/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`

### 批量处理多个舞蹈数据

```bash
# 使用批量脚本
python scripts/batch_csv_to_npz.py \
  --input_dir dance_motion \
  --robot g1
```

### 处理长舞蹈片段

如果舞蹈很长，可以分段训练：

```bash
# 训练前 500 帧
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --input_fps 30 \
  --frame_range 0 500 \
  --output_name douyin1_part1 \
  --robot g1 \
  --headless
```

### 添加过渡帧

在动作开始和结束时添加平滑过渡：

```bash
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --input_fps 30 \
  --output_name douyin1_smooth \
  --robot g1 \
  --prepend_frames 50 \
  --append_frames 50 \
  --headless
```

## 🐛 常见问题

### 1. WandB 相关问题

**问题**: "Make sure to export WANDB_ENTITY to your organization name"
```bash
export WANDB_ENTITY=your-org-name  # 使用组织名，不是个人用户名
```

### 2. 内存不足

如果 GPU 内存不足，减少并行环境数量：
```bash
--num_envs 2048  # 或更少
```

### 3. 临时文件夹问题

如果 `/tmp` 不可访问，修改 `scripts/csv_to_npz.py` 中的临时文件夹路径（第 319 和 326 行）。

### 4. CSV 格式检查

确保您的 CSV 文件格式与 Unitree 数据集一致：
- 每行代表一个时间步
- 包含所有关节角度（按 G1 机器人的关节顺序）
- 数值为弧度制

## 📊 训练监控

训练过程中，您可以在 WandB 网站上实时监控：
- 奖励曲线
- 策略损失
- 价值函数损失
- 动作跟踪误差

## 🎯 预期结果

- **训练时间**: 根据动作复杂度，通常需要 2000-5000 次迭代
- **GPU 使用**: 单 GPU (RTX 3090 或更高) 推荐
- **成功指标**: 平均奖励 > 0.8（满分 1.0）

## 📚 相关资源

- [BeyondMimic 官网](https://beyondmimic.github.io/)
- [论文](https://arxiv.org/abs/2508.08241)
- [视频演示](https://youtu.be/RS_MtKVIAzY)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab)

## 💡 快速开始命令

完整的训练命令（一次性执行）：

```bash
# 1. 激活环境并安装
conda activate isaaclab
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab
python -m pip install -e source/robocup_lab

# 2. 设置 WandB
wandb login
export WANDB_ENTITY=your-org-name

# 3. 转换数据
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --input_fps 30 \
  --output_name douyin1_dance \
  --robot g1 \
  --headless

# 4. 开始训练
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name ${WANDB_ENTITY}-org/wandb-registry-motions/douyin1_dance \
  --headless \
  --logger wandb \
  --log_project_name g1_dance_training \
  --run_name douyin1_dance_v1 \
  --num_envs 4096
```

---

**祝训练顺利！如有问题，请查看项目的 README.md 或提交 Issue。** 🎉

