#!/bin/bash
# 自定义舞蹈训练脚本 - 可根据需求修改参数

# ============ 配置区域 - 根据您的需求修改 ============

# 舞蹈数据文件
DANCE_CSV="dance_motion/douyin1_g1.csv"
# 如果有其他舞蹈数据，可以改为：
# DANCE_CSV="dance_motion/另一个舞蹈.csv"

# 动作名称（用于 WandB Registry）
MOTION_NAME="douyin1_dance"

# 输入/输出帧率
INPUT_FPS=30   # CSV 文件的原始帧率
OUTPUT_FPS=50  # 训练使用的帧率

# 帧范围（可选，如果不指定则使用全部帧）
# 取消注释下面的行来使用特定帧范围
# FRAME_RANGE="--frame_range 0 500"  # 只使用第 0-500 帧
FRAME_RANGE=""  # 使用全部帧

# 平滑过渡（在动作开始和结束时添加过渡帧）
PREPEND_FRAMES=0   # 开始前的过渡帧数
APPEND_FRAMES=0    # 结束后的过渡帧数
# 如果需要平滑过渡，可以设置为：
# PREPEND_FRAMES=50
# APPEND_FRAMES=50

# 机器人类型
ROBOT_TYPE="g1"  # 可选: "g1" 或 "booster_t1"

# 训练任务
TASK_NAME="Tracking-Flat-G1-v0"
# 其他可选任务:
# - "Tracking-Flat-G1-v0" (G1 平地)
# - "Tracking-Flat-Booster-T1-v0" (Booster T1 平地)

# WandB 日志配置
PROJECT_NAME="g1_dance_training"  # WandB 项目名称
RUN_NAME="douyin1_v1"             # 本次训练的名称

# 训练参数
NUM_ENVS=4096        # 并行环境数量（根据 GPU 内存调整）
MAX_ITERATIONS=5000  # 最大训练迭代次数（可选）
# GPU 内存参考:
# - RTX 3090 (24GB): 4096-8192
# - RTX 3080 (10GB): 2048-4096
# - 更小的 GPU: 1024-2048

# 是否使用无头模式
HEADLESS="--headless"
# 如果想看可视化，可以注释掉或改为空
# HEADLESS=""

# ============ 脚本执行区域 - 一般不需要修改 ============

set -e

echo "=========================================="
echo "G1 机器人舞蹈训练 - 自定义配置"
echo "=========================================="
echo ""
echo "配置信息："
echo "  舞蹈数据: $DANCE_CSV"
echo "  动作名称: $MOTION_NAME"
echo "  机器人: $ROBOT_TYPE"
echo "  输入帧率: $INPUT_FPS FPS"
echo "  输出帧率: $OUTPUT_FPS FPS"
echo "  并行环境: $NUM_ENVS"
echo "  WandB 项目: $PROJECT_NAME"
echo "  运行名称: $RUN_NAME"
echo ""

# 检查 WandB 配置
if [ -z "$WANDB_ENTITY" ]; then
    echo "❌ 错误: 请先设置 WANDB_ENTITY 环境变量"
    echo "   export WANDB_ENTITY=your-organization-name"
    exit 1
fi

echo "  WandB 组织: $WANDB_ENTITY"
echo ""

# 检查舞蹈数据文件是否存在
if [ ! -f "$DANCE_CSV" ]; then
    echo "❌ 错误: 找不到舞蹈数据文件: $DANCE_CSV"
    exit 1
fi

# 步骤 1: 检查安装
echo "步骤 1/3: 检查 robocup_lab 安装..."
if ! python -c "import robocup_lab" 2>/dev/null; then
    echo "安装 robocup_lab 包..."
    python -m pip install -e source/robocup_lab
    echo "✓ 安装完成"
else
    echo "✓ robocup_lab 已安装"
fi
echo ""

# 步骤 2: 转换舞蹈数据
echo "步骤 2/3: 转换舞蹈数据..."
CMD="python scripts/csv_to_npz.py \
    --input_file $DANCE_CSV \
    --input_fps $INPUT_FPS \
    --output_fps $OUTPUT_FPS \
    --output_name $MOTION_NAME \
    --robot $ROBOT_TYPE \
    $HEADLESS"

# 添加可选参数
if [ ! -z "$FRAME_RANGE" ]; then
    CMD="$CMD $FRAME_RANGE"
fi

if [ $PREPEND_FRAMES -gt 0 ]; then
    CMD="$CMD --prepend_frames $PREPEND_FRAMES"
fi

if [ $APPEND_FRAMES -gt 0 ]; then
    CMD="$CMD --append_frames $APPEND_FRAMES"
fi

echo "执行命令: $CMD"
eval $CMD

echo "✓ 数据转换完成"
echo ""

# 步骤 3: 开始训练
echo "步骤 3/3: 开始训练..."
REGISTRY_PATH="${WANDB_ENTITY}-org/wandb-registry-motions/${MOTION_NAME}"
echo "Registry 路径: $REGISTRY_PATH"
echo ""

TRAIN_CMD="python scripts/rsl_rl/train.py \
    --task=$TASK_NAME \
    --registry_name $REGISTRY_PATH \
    $HEADLESS \
    --logger wandb \
    --log_project_name $PROJECT_NAME \
    --run_name $RUN_NAME \
    --num_envs $NUM_ENVS"

if [ ! -z "$MAX_ITERATIONS" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_iterations $MAX_ITERATIONS"
fi

echo "执行训练命令..."
eval $TRAIN_CMD

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "查看训练结果："
echo "  WandB: https://wandb.ai/${WANDB_ENTITY}/${PROJECT_NAME}"
echo ""
echo "测试策略的命令："
echo "  python scripts/rsl_rl/play.py \\"
echo "    --task=$TASK_NAME \\"
echo "    --num_envs=2 \\"
echo "    --wandb_path=${WANDB_ENTITY}/${PROJECT_NAME}/[运行ID]"
echo ""
echo "提示: 运行 ID 可以在 WandB 网站的运行概览中找到"
echo ""

