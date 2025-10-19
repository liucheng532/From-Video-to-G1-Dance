#!/bin/bash
# 本地训练脚本（不使用 WandB Registry）

set -e

echo "=========================================="
echo "本地训练 - G1 机器人舞蹈"
echo "=========================================="
echo ""

# 配置参数
DANCE_CSV="dance_motion/douyin1_g1.csv"
OUTPUT_NPZ="./motions/douyin1_dance.npz"
INPUT_FPS=30
OUTPUT_FPS=50
ROBOT_TYPE="g1"
TASK_NAME="Tracking-Flat-G1-v0"
NUM_ENVS=4096
MAX_ITERATIONS=5000

# 创建输出目录
mkdir -p ./motions

echo "配置信息："
echo "  舞蹈数据: $DANCE_CSV"
echo "  输出文件: $OUTPUT_NPZ"
echo "  机器人: $ROBOT_TYPE"
echo "  并行环境: $NUM_ENVS"
echo "  最大迭代: $MAX_ITERATIONS"
echo ""

# 步骤 1: 转换舞蹈数据到本地文件
echo "步骤 1/2: 转换舞蹈数据..."
python scripts/csv_to_npz.py \
    --input_file "$DANCE_CSV" \
    --input_fps $INPUT_FPS \
    --output_fps $OUTPUT_FPS \
    --output_name douyin1_dance \
    --robot "$ROBOT_TYPE" \
    --headless

echo "✓ 数据转换完成"
echo "  文件保存在: $OUTPUT_NPZ"
echo ""

# 步骤 2: 开始本地训练（不使用 WandB）
echo "步骤 2/2: 开始本地训练..."
echo "  使用本地 NPZ 文件"
echo "  不需要 WandB Registry"
echo ""

# 等待 NPZ 文件生成
sleep 2

# 查找生成的 NPZ 文件
NPZ_FILE=$(find ./wandb -name "motion.npz" -type f | head -n 1)

if [ -z "$NPZ_FILE" ]; then
    echo "❌ 错误: 找不到生成的 motion.npz 文件"
    echo "请检查数据转换步骤是否成功"
    exit 1
fi

echo "✓ 找到动作文件: $NPZ_FILE"
echo ""

python train_local_simple.py \
    --motion_file "$NPZ_FILE" \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --experiment_name g1_dance_local \
    --run_name douyin1 \
    --headless

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "训练结果保存在 logs/rsl_rl/g1_dance_local/ 目录"
echo ""
echo "测试训练结果："
echo "  python scripts/rsl_rl/play.py \\"
echo "    --task=$TASK_NAME \\"
echo "    --num_envs=2 \\"
echo "    --checkpoint=logs/rsl_rl/g1_dance_local/[日期时间]/model_*.pt"
echo ""

