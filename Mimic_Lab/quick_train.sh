#!/bin/bash
# G1 机器人舞蹈训练快速启动脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "G1 机器人舞蹈训练脚本"
echo "=========================================="
echo ""

# 配置参数
DANCE_CSV="dance_motion/douyin1_g1.csv"
MOTION_NAME="douyin1_dance"
INPUT_FPS=30
OUTPUT_FPS=50
ROBOT_TYPE="g1"
TASK_NAME="Tracking-Flat-G1-v0"
PROJECT_NAME="g1_dance_training"
RUN_NAME="douyin1_v1"
NUM_ENVS=4096

# 检查 WandB 配置
if [ -z "$WANDB_ENTITY" ]; then
    echo "⚠️  错误: 请先设置 WANDB_ENTITY 环境变量"
    echo "   export WANDB_ENTITY=your-organization-name"
    exit 1
fi

echo "✓ WandB 组织: $WANDB_ENTITY"
echo "✓ 舞蹈数据: $DANCE_CSV"
echo "✓ 机器人类型: $ROBOT_TYPE"
echo ""

# 步骤 1: 安装依赖（如果需要）
echo "步骤 1: 检查安装..."
if ! python -c "import robocup_lab" 2>/dev/null; then
    echo "安装 robocup_lab 包..."
    python -m pip install -e source/robocup_lab
    echo "✓ 安装完成"
else
    echo "✓ robocup_lab 已安装"
fi
echo ""

# 步骤 2: 转换舞蹈数据
echo "步骤 2: 转换舞蹈数据为 NPZ 格式..."
python scripts/csv_to_npz.py \
    --input_file "$DANCE_CSV" \
    --input_fps $INPUT_FPS \
    --output_fps $OUTPUT_FPS \
    --output_name "$MOTION_NAME" \
    --robot "$ROBOT_TYPE" \
    --headless

echo "✓ 数据转换完成并上传到 WandB Registry"
echo ""

# 步骤 3: 开始训练
echo "步骤 3: 开始训练策略..."
echo "Registry 路径: ${WANDB_ENTITY}-org/wandb-registry-motions/${MOTION_NAME}"
echo ""

python scripts/rsl_rl/train.py \
    --task="$TASK_NAME" \
    --registry_name "${WANDB_ENTITY}-org/wandb-registry-motions/${MOTION_NAME}" \
    --headless \
    --logger wandb \
    --log_project_name "$PROJECT_NAME" \
    --run_name "$RUN_NAME" \
    --num_envs $NUM_ENVS

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo ""
echo "查看训练结果："
echo "  WandB: https://wandb.ai/${WANDB_ENTITY}/${PROJECT_NAME}"
echo ""
echo "测试训练好的策略："
echo "  python scripts/rsl_rl/play.py \\"
echo "    --task=$TASK_NAME \\"
echo "    --num_envs=2 \\"
echo "    --wandb_path=${WANDB_ENTITY}/${PROJECT_NAME}/[RUN_ID]"
echo ""

