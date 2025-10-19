#!/bin/bash

# MotionTracking 文件准备脚本
# 用于从 Beyondmimic 项目复制必要的模型和动作数据文件

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MotionTracking 文件准备脚本${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 目标目录
MODEL_DIR="$SCRIPT_DIR/policy/montion_tracking/model"
MOTION_DIR="$SCRIPT_DIR/policy/montion_tracking/motion"

# 源项目路径（Beyondmimic）
BEYONDMIMIC_ROOT="/home/qp/桌面/LZY/Beyondmimic_Deploy_G1"
BEYONDMIMIC_POLICY_DIR="$BEYONDMIMIC_ROOT/deploy_real/bydmimic/policy"
BEYONDMIMIC_MOTION_DIR="$BEYONDMIMIC_ROOT/deploy_real/bydmimic/motion"

# 检查 Beyondmimic 项目是否存在
if [ ! -d "$BEYONDMIMIC_ROOT" ]; then
    echo -e "${RED}错误: 找不到 Beyondmimic 项目${NC}"
    echo -e "预期路径: $BEYONDMIMIC_ROOT"
    echo -e "请确认路径是否正确，或手动复制文件"
    exit 1
fi

echo -e "${YELLOW}源项目路径: $BEYONDMIMIC_ROOT${NC}"
echo -e "${YELLOW}目标模型目录: $MODEL_DIR${NC}"
echo -e "${YELLOW}目标动作目录: $MOTION_DIR${NC}\n"

# 创建目标目录
mkdir -p "$MODEL_DIR"
mkdir -p "$MOTION_DIR"

# 列出可用的模型文件
echo -e "${GREEN}可用的 ONNX 模型文件:${NC}"
if [ -d "$BEYONDMIMIC_POLICY_DIR" ]; then
    ls -lh "$BEYONDMIMIC_POLICY_DIR"/*.onnx 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo -e "${RED}  未找到 policy 目录${NC}"
fi

echo ""

# 列出可用的动作数据文件
echo -e "${GREEN}可用的动作数据文件:${NC}"
if [ -d "$BEYONDMIMIC_MOTION_DIR" ]; then
    ls -lh "$BEYONDMIMIC_MOTION_DIR"/*.npz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo -e "${RED}  未找到 motion 目录${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${YELLOW}选择要复制的文件:${NC}\n"

# 复制 ONNX 模型
echo -e "${GREEN}[1] ONNX 模型文件${NC}"
echo "输入要复制的模型文件名（例如: oops_3min.onnx）"
echo "或直接按回车跳过"
read -p "> " MODEL_FILE

if [ -n "$MODEL_FILE" ]; then
    SRC_MODEL="$BEYONDMIMIC_POLICY_DIR/$MODEL_FILE"
    if [ -f "$SRC_MODEL" ]; then
        cp "$SRC_MODEL" "$MODEL_DIR/"
        echo -e "${GREEN}✓ 已复制模型: $MODEL_FILE${NC}"
    else
        echo -e "${RED}✗ 文件不存在: $SRC_MODEL${NC}"
    fi
else
    echo -e "${YELLOW}跳过模型文件${NC}"
fi

echo ""

# 复制动作数据
echo -e "${GREEN}[2] 动作数据文件${NC}"
echo "输入要复制的动作文件名（例如: oops1011_final_g1.npz）"
echo "或直接按回车跳过"
read -p "> " MOTION_FILE

if [ -n "$MOTION_FILE" ]; then
    SRC_MOTION="$BEYONDMIMIC_MOTION_DIR/$MOTION_FILE"
    if [ -f "$SRC_MOTION" ]; then
        cp "$SRC_MOTION" "$MOTION_DIR/"
        echo -e "${GREEN}✓ 已复制动作数据: $MOTION_FILE${NC}"
        
        # 分析动作数据
        echo -e "\n${YELLOW}分析动作数据...${NC}"
        python3 <<EOF
import numpy as np
try:
    motion = np.load("$MOTION_DIR/$MOTION_FILE")
    print(f"  包含的键: {list(motion.files)}")
    if 'joint_pos' in motion.files:
        total_frames = motion['joint_pos'].shape[0]
        duration = total_frames * 0.02
        print(f"  总帧数: {total_frames}")
        print(f"  动作时长: {duration:.2f} 秒")
        print(f"  关节维度: {motion['joint_pos'].shape[1]}")
except Exception as e:
    print(f"  分析失败: {e}")
EOF
    else
        echo -e "${RED}✗ 文件不存在: $SRC_MOTION${NC}"
    fi
else
    echo -e "${YELLOW}跳过动作数据文件${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}文件准备完成!${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 列出当前文件
echo -e "${YELLOW}当前 MotionTracking 文件:${NC}\n"

echo -e "${GREEN}模型文件 ($MODEL_DIR):${NC}"
ls -lh "$MODEL_DIR" 2>/dev/null | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' || echo "  (空)"

echo ""

echo -e "${GREEN}动作数据 ($MOTION_DIR):${NC}"
ls -lh "$MOTION_DIR" 2>/dev/null | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}' || echo "  (空)"

echo -e "\n${YELLOW}下一步:${NC}"
echo "1. 编辑配置文件: policy/montion_tracking/config/Motion.yaml"
echo "2. 更新 policy_path 和 motion_path 字段"
echo "3. 运行测试: python deploy_mujoco/deploy_mujoco.py"

echo -e "\n${GREEN}完成!${NC}"

