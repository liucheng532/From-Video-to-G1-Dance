# From-Video-to-G1-Dance

一个完整的机器人舞蹈系统，实现从人类舞蹈视频到Unitree G1机器人舞蹈的全流程自动化处理。

## 🎯 项目概述

本项目实现了一个完整的机器人舞蹈生成管道，包含以下四个主要阶段：

1. **视频处理** (GVHMR) - 从舞蹈视频中提取人体动作数据
2. **动作重定向** (GMR) - 将人体动作数据映射到机器人动作数据
3. **强化学习训练** (Mimic_Lab) - 训练机器人执行舞蹈动作的策略
4. **部署与仿真** (MimicDeploy_G1) - Sim2Sim和Sim2Real部署

## 🏗️ 系统架构

```
舞蹈视频 → GVHMR → 人体动作数据 → GMR → 机器人动作数据 → Mimic_Lab → 训练策略 → MimicDeploy_G1 → 机器人执行
```

## 📁 项目结构

```
robo_dance/
├── GVHMR/                    # 视频处理模块
│   ├── danceVedio/          # 输入舞蹈视频
│   ├── outputs/             # 输出人体动作数据
│   └── tools/               # 处理工具
├── GMR/                     # 动作重定向模块
│   ├── assets/              # 机器人模型文件
│   ├── output/              # 输出机器人动作数据
│   ├── scripts/             # 转换脚本
│   └── general_motion_retargeting/  # 核心重定向算法
├── Mimic_Lab/               # 强化学习训练模块
│   ├── artifacts/           # 训练数据
│   ├── scripts/             # 训练脚本
│   └── source/              # 训练环境源码
├── MimicDeploy_G1/          # 部署模块
│   └── MimicDeploy_G1/      # G1机器人部署代码
└── IsaacLab/                # 仿真环境
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.12+
- Isaac Sim 2022.2+

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/From-Video-to-G1-Dance.git
cd From-Video-to-G1-Dance
```

2. **安装依赖**
```bash
# 安装GVHMR依赖
cd GVHMR
pip install -r requirements.txt

# 安装GMR依赖
cd ../GMR
pip install -e .

# 安装Mimic_Lab依赖
cd ../Mimic_Lab
pip install -e .
```

## 📋 使用流程

### 第一步：视频处理 (GVHMR)

将舞蹈视频转换为人体动作数据：

```bash
cd GVHMR
CUDA_VISIBLE_DEVICES=1 python tools/demo/demo.py --video=danceVedio/love_you_final.mp4 -s
```

**输出**: `outputs/demo/[video_name]/hmr4d_results.pt`

### 第二步：动作重定向 (GMR)

将人体动作数据转换为机器人动作数据：

```bash
cd GMR
python scripts/gvhmr_to_robot.py \
    --gvhmr_pred_file ../GVHMR/outputs/demo/douyin_final/hmr4d_results.pt \
    --robot unitree_g1 \
    --save_path output/douyin_final_g1.pkl
```

**输出**: `output/[motion_name]_g1.pkl`

### 第三步：数据插值处理

对机器人动作数据进行插值处理，添加起始和结束动作：

```bash
cd GMR
python output/gmr_to_pbhc_with_interpolation.py --input love_you_final_g1.pkl
```

**输出**: `output/[motion_name]_g1_interp_S30_E30.pkl`

### 第四步：动作预览

预览生成的机器人动作：

```bash
cd GMR
python scripts/vis_robot_motion.py \
    --robot unitree_g1 \
    --robot_motion_path output/douyin_final_g1.pkl
```

### 第五步：强化学习训练 (Mimic_Lab)

#### 5.1 数据格式转换

将PKL文件转换为训练所需的NPZ格式：

```bash
cd Mimic_Lab
python scripts/csv_to_npz.py \
    --input_file dancedata/love_you_final_g1_interp_S30_E30.csv \
    --input_fps 30 \
    --output_name love_you_final_g1 \
    --output_fps 50 \
    --skip_wandb \
    --headless
```

#### 5.2 动作回放测试

测试转换后的动作数据：

```bash
python scripts/replay_npz.py --local_file dancedata/love_you_final_g1.npz
```

#### 5.3 策略训练

训练机器人执行舞蹈动作的策略：

```bash
conda activate env_isaaclab
cd /home/lyz/Desktop/code/robo_dance/Mimic_Lab
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
    --num_envs 4096 \
    --headless \
    --motion_file artifacts/love_you_final_g1.npz \
    --logger wandb \
    --max_iterations 15000 \
    --run_name love_you_v1
```

#### 5.4 训练结果测试

测试训练好的策略：

```bash
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-G1-v0 \
    --load_run 2025-10-07_02-47-36_love_you_v1 \
    --checkpoint model_9999.pt \
    --motion_file artifacts/love_you_final_g1.npz \
    --num_envs 1
```

### 第六步：部署与仿真 (MimicDeploy_G1)

参考 `MimicDeploy_G1/` 目录中的部署指南进行Sim2Sim和Sim2Real部署。

## 🎮 支持的机器人

- **Unitree G1** - 主要支持的机器人型号
- **Unitree H1** - 支持但未完全测试
- 其他机器人型号可通过配置文件添加

## 📊 数据格式说明

### 输入格式
- **视频文件**: MP4, AVI, MOV等常见格式
- **分辨率**: 建议720p以上
- **时长**: 建议30秒以内

### 中间格式
- **GVHMR输出**: `.pt`文件，包含人体姿态数据
- **GMR输出**: `.pkl`文件，包含机器人关节角度数据
- **训练数据**: `.npz`文件，包含标准化的动作序列

### 输出格式
- **策略模型**: `.pt`文件，包含训练好的神经网络权重
- **部署配置**: JSON/YAML配置文件

## 🔧 配置说明

### 机器人配置
在 `GMR/assets/` 目录中，每个机器人都有对应的URDF文件和配置文件：
- `unitree_g1/` - G1机器人配置
- `unitree_h1/` - H1机器人配置

### 训练配置
在 `Mimic_Lab/` 目录中：
- `scripts/rsl_rl/train.py` - 主训练脚本
- 支持多种任务配置和超参数调整

## 📈 性能指标

- **视频处理速度**: ~1分钟/30秒视频 (RTX 3080)
- **动作重定向**: ~10秒/动作序列
- **训练时间**: ~2-4小时 (4096环境并行)
- **推理速度**: 实时 (60+ FPS)

## 🐛 常见问题

### 1. 内存不足
- 减少并行环境数量 (`--num_envs`)
- 使用更小的批处理大小

### 2. CUDA错误
- 检查CUDA版本兼容性
- 确保GPU内存充足

### 3. 动作质量问题
- 检查输入视频质量
- 调整插值参数
- 验证机器人配置

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [GVHMR](https://github.com/shaoxuan-chen/GVHMR) - 视频姿态估计
- [GMR](https://github.com/DeepMotionEditing/general_motion_retargeting) - 动作重定向
- [Mimic_Lab](https://github.com/DeepMotionEditing/MimicLab) - 强化学习训练
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - 仿真环境

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至: [cliu425@connect.hkust-gz.edu.cn]

---

**注意**: 本项目仅用于研究和教育目的。使用前请确保遵守相关法律法规和机器人安全规范。
