#!/bin/bash
# 重新运行训练脚本（Registry 创建后使用）

set -e

echo "=========================================="
echo "重新开始训练（Registry 已创建）"
echo "=========================================="
echo ""

# 检查 WANDB_ENTITY
if [ -z "$WANDB_ENTITY" ]; then
    echo "设置 WANDB_ENTITY..."
    export WANDB_ENTITY=1393537481-the-hong-kong-university-of-science-and-techn
fi

echo "WandB 组织: $WANDB_ENTITY"
echo ""

# 配置
DANCE_CSV="dance_motion/douyin1_g1.csv"
MOTION_NAME="douyin1_dance"
INPUT_FPS=30
OUTPUT_FPS=50
ROBOT_TYPE="g1"
TASK_NAME="Tracking-Flat-G1-v0"
PROJECT_NAME="g1_dance_training"
RUN_NAME="douyin1_v1"
NUM_ENVS=4096

echo "步骤 1/2: 转换舞蹈数据..."
python scripts/csv_to_npz.py \
    --input_file "$DANCE_CSV" \
    --input_fps $INPUT_FPS \
    --output_fps $OUTPUT_FPS \
    --output_name "$MOTION_NAME" \
    --robot "$ROBOT_TYPE" \
    --headless

echo ""
echo "✓ 数据转换完成"
echo ""

echo "步骤 2/2: 开始训练..."
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
echo "✅ 训练完成！"
echo "=========================================="

