#!/bin/bash

# 自动串行训练两个舞蹈动作
# 作者：自动生成
# 日期：2025-10-09

"""chmod +x /home/lyz/Desktop/code/robo_dance/Robocup_Lab/auto_train_two_dances.sh
方式一：
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab
bash auto_train_two_dances.sh

方式二：后台运行+日志记录（可关闭终端）
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab
nohup bash auto_train_two_dances.sh > training_log_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""



echo "=========================================="
echo "🌙 开始自动训练两个舞蹈动作"
echo "=========================================="
echo ""

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# 进入项目目录
cd /home/lyz/Desktop/code/robo_dance/Robocup_Lab

# 记录开始时间
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "⏰ 总任务开始时间: $START_TIME"
echo ""

# ================== 第一个训练任务：loveu ==================
echo "=========================================="
echo "💖 [1/2] 开始训练 loveu1009_g1.npz"
echo "=========================================="

TRAIN1_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "开始时间: $TRAIN1_START"

python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 4096 \
  --headless \
  --motion_file artifacts/loveu1009_g1.npz \
  --logger wandb \
  --max_iterations 7000 \
  --run_name loveu_1009_v1

TRAIN1_EXIT_CODE=$?
TRAIN1_END=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "💖 loveu 训练完成！"
echo "结束时间: $TRAIN1_END"
echo "退出代码: $TRAIN1_EXIT_CODE"
echo ""

# 检查第一个训练是否成功
if [ $TRAIN1_EXIT_CODE -ne 0 ]; then
    echo "❌ loveu 训练失败！退出代码: $TRAIN1_EXIT_CODE"
    echo "⚠️  不会继续训练 oops"
    exit $TRAIN1_EXIT_CODE
fi

# 短暂休息 10 秒
echo "😴 短暂休息 10 秒..."
sleep 10
echo ""

# ================== 第二个训练任务：oops ==================
echo "=========================================="
echo "🎵 [2/2] 开始训练 oops1009_g1.npz"
echo "=========================================="

TRAIN2_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "开始时间: $TRAIN2_START"

python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs 4096 \
  --headless \
  --motion_file artifacts/oops1009_g1.npz \
  --logger wandb \
  --max_iterations 12000 \
  --run_name oops_1009_v1

TRAIN2_EXIT_CODE=$?
TRAIN2_END=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "🎵 oops 训练完成！"
echo "结束时间: $TRAIN2_END"
echo "退出代码: $TRAIN2_EXIT_CODE"
echo ""

# ================== 总结报告 ==================
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================="
echo "✅ 所有训练任务完成！"
echo "=========================================="
echo ""
echo "📊 训练总结："
echo "  总开始时间: $START_TIME"
echo "  总结束时间: $END_TIME"
echo ""
echo "  loveu 训练:"
echo "    - 开始: $TRAIN1_START"
echo "    - 结束: $TRAIN1_END"
echo "    - 状态: $([ $TRAIN1_EXIT_CODE -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo ""
echo "  oops 训练:"
echo "    - 开始: $TRAIN2_START"
echo "    - 结束: $TRAIN2_END"
echo "    - 状态: $([ $TRAIN2_EXIT_CODE -eq 0 ] && echo '✅ 成功' || echo '❌ 失败')"
echo ""
echo "🎉 所有任务执行完毕，可以安心睡觉了！"
echo "=========================================="

# 退出时使用第二个训练的退出代码
exit $TRAIN2_EXIT_CODE

