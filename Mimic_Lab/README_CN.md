# 🎭 G1 机器人舞蹈训练系统

基于 BeyondMimic 的 G1 人形机器人舞蹈动作学习框架，使用 IsaacLab 进行强化学习训练。

## 📊 您的舞蹈数据状态

✅ **数据文件**: `dance_motion/douyin1_g1.csv`  
✅ **数据格式**: 36 个关节角度（符合 G1 机器人规格）  
✅ **总帧数**: 1055 帧  
✅ **舞蹈时长**: 约 35 秒（按 30 FPS 计算）  
✅ **数据状态**: 格式正确，可以直接使用！

## 🚀 三种训练方式

### 方式 1: 一键快速训练（推荐新手）

```bash
# 激活环境
conda activate isaaclab

# 进入目录
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab

# 配置 WandB（首次使用需要）
wandb login
export WANDB_ENTITY=你的组织名

# 一键训练
./quick_train.sh
```

### 方式 2: 自定义参数训练（推荐进阶用户）

编辑 `train_custom.sh` 文件，修改配置区域的参数，然后运行：

```bash
./train_custom.sh
```

可自定义：
- 舞蹈数据文件路径
- 训练帧范围（训练部分片段）
- 并行环境数量（根据 GPU 调整）
- 平滑过渡帧数
- 训练迭代次数

### 方式 3: 手动逐步执行（推荐专业用户）

详细步骤请参考 `快速开始.md` 或 `TRAINING_GUIDE.md`

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `快速开始.md` | 简洁的快速参考指南（中文） |
| `TRAINING_GUIDE.md` | 完整的训练指南（英文） |
| `quick_train.sh` | 一键训练脚本 |
| `train_custom.sh` | 可自定义参数的训练脚本 |
| `dance_motion/douyin1_g1.csv` | 您的舞蹈数据文件 ✅ |

## ⚙️ 系统要求

### 必需
- ✅ IsaacLab v2.1.0 (已有 conda 环境)
- ✅ NVIDIA GPU (推荐 RTX 3090 或更高)
- ✅ Python 3.10
- ✅ CUDA 支持

### 可选
- WandB 账号（用于训练监控和模型管理）

## 🎯 快速开始（3 分钟上手）

```bash
# 1. 激活 IsaacLab 环境
conda activate isaaclab

# 2. 进入项目
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab

# 3. 安装项目
python -m pip install -e source/robocup_lab

# 4. 设置 WandB（首次）
wandb login
export WANDB_ENTITY=你的组织名

# 5. 开始训练！
./quick_train.sh
```

训练完成后，在 WandB 网站查看结果并测试策略。

## 🎛️ 训练参数建议

### GPU 内存配置

| GPU 型号 | 内存 | 推荐 num_envs | 训练速度 |
|---------|------|--------------|---------|
| RTX 4090 | 24GB | 8192 | 很快 ⚡⚡⚡ |
| RTX 3090 | 24GB | 4096-6144 | 快 ⚡⚡ |
| RTX 3080 | 10GB | 2048-4096 | 中等 ⚡ |
| RTX 3070 | 8GB | 1024-2048 | 较慢 |

### 舞蹈复杂度配置

**简单舞蹈**（动作幅度小、速度慢）:
```bash
--max_iterations 2000
--num_envs 4096
```

**中等舞蹈**（您的 douyin1 舞蹈）:
```bash
--max_iterations 3000-4000
--num_envs 4096
```

**复杂舞蹈**（快速、大幅度动作）:
```bash
--max_iterations 5000-8000
--num_envs 6144
```

### 训练片段策略

如果舞蹈很长，建议分段训练：

```bash
# 训练前半段（0-500 帧）
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --frame_range 0 500 \
  --output_name douyin1_part1 \
  --robot g1 --headless

# 训练后半段（500-1055 帧）
python scripts/csv_to_npz.py \
  --input_file dance_motion/douyin1_g1.csv \
  --frame_range 500 1055 \
  --output_name douyin1_part2 \
  --robot g1 --headless
```

## 📈 训练监控

训练过程中关注这些指标（在 WandB 上查看）：

| 指标 | 目标值 | 说明 |
|-----|--------|------|
| `rewards/episode_return` | > 0.8 | 平均回报，越高越好 |
| `rewards/tracking_lin_vel` | > 0.9 | 线速度跟踪精度 |
| `rewards/tracking_ang_vel` | > 0.9 | 角速度跟踪精度 |
| `rewards/joint_pos` | > 0.85 | 关节位置跟踪精度 |

## 🎬 训练完成后

### 1. 在仿真中测试

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs=2 \
  --wandb_path=组织名/g1_dance_training/运行ID
```

### 2. 导出为 ONNX（用于实际机器人部署）

训练好的模型会自动保存在 WandB 上，可以导出为 ONNX 格式部署到真实 G1 机器人。

### 3. 评估性能

观察机器人：
- ✅ 动作流畅度
- ✅ 关节跟踪精度
- ✅ 平衡稳定性
- ✅ 舞蹈完成度

## ❓ 常见问题

### 1. "Command 'python' not found"
**解决**: 确保已激活 conda 环境
```bash
conda activate isaaclab
```

### 2. "WANDB_ENTITY not set"
**解决**: 设置 WandB 组织名（不是个人用户名）
```bash
export WANDB_ENTITY=你的组织名
```

### 3. GPU 内存不足
**解决**: 减少并行环境数量
```bash
--num_envs 2048  # 或更少
```

### 4. 训练速度慢
**解决**: 
- 使用 `--headless` 模式
- 增加 `num_envs`（如果 GPU 内存允许）
- 关闭其他占用 GPU 的程序

### 5. 动作跟踪不准确
**解决**:
- 增加训练迭代次数
- 检查 CSV 数据是否正确（弧度制）
- 尝试添加平滑过渡帧

## 📚 相关资源

- [BeyondMimic 官网](https://beyondmimic.github.io/)
- [论文 (Arxiv)](https://arxiv.org/abs/2508.08241)
- [视频演示](https://youtu.be/RS_MtKVIAzY)
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab)

## 🔧 项目结构

```
Robocup_Lab/
├── dance_motion/              # 舞蹈数据目录
│   └── douyin1_g1.csv        # 您的舞蹈数据 ✅
├── scripts/                   # 训练和工具脚本
│   ├── csv_to_npz.py         # 数据转换脚本
│   ├── replay_npz.py         # 动作回放脚本
│   └── rsl_rl/
│       ├── train.py          # 训练脚本
│       └── play.py           # 测试脚本
├── source/robocup_lab/       # 核心代码
│   └── robocup_lab/
│       ├── tasks/            # 任务定义
│       │   └── tracking/     # 动作跟踪任务
│       └── robots/           # 机器人配置
│           └── g1.py         # G1 机器人配置
├── quick_train.sh            # 一键训练脚本 ⚡
├── train_custom.sh           # 自定义训练脚本 🎛️
├── 快速开始.md               # 中文快速指南 📖
├── TRAINING_GUIDE.md         # 完整英文指南 📚
└── README_CN.md              # 本文件 📄
```

## 💡 使用建议

1. **首次使用**: 先用 `quick_train.sh` 快速体验
2. **调试阶段**: 使用较少的帧数（如 `--frame_range 0 300`）快速验证
3. **正式训练**: 使用完整数据集，增加迭代次数
4. **GPU 优化**: 根据显存大小调整 `num_envs`
5. **监控训练**: 通过 WandB 实时查看训练曲线

## 🎉 开始训练

准备好了吗？执行这个命令开始您的第一次训练：

```bash
conda activate isaaclab
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab
export WANDB_ENTITY=你的组织名
./quick_train.sh
```

祝训练顺利！🚀

---

**技术支持**: 
- 查看详细文档: `TRAINING_GUIDE.md`
- 快速参考: `快速开始.md`
- 项目主页: https://beyondmimic.github.io/







