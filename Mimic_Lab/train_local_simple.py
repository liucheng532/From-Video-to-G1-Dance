#!/usr/bin/env python
"""本地训练脚本 - 直接使用本地 NPZ 文件，不需要 WandB Registry"""

import argparse
import sys
import os
from datetime import datetime

# 添加 IsaacLab 路径
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="本地训练 - 使用本地 NPZ 文件")
parser.add_argument("--motion_file", type=str, required=True, help="本地 NPZ 动作文件路径")
parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数量")
parser.add_argument("--max_iterations", type=int, default=5000, help="最大训练迭代次数")
parser.add_argument("--experiment_name", type=str, default="g1_dance_local", help="实验名称")
parser.add_argument("--run_name", type=str, default="local_train", help="运行名称")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""主要导入"""
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from robocup_lab.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

# 导入环境和配置
import robocup_lab.tasks
from robocup_lab.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from robocup_lab.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    
    # 检查动作文件是否存在
    if not os.path.exists(args_cli.motion_file):
        raise FileNotFoundError(f"动作文件不存在: {args_cli.motion_file}")
    
    print(f"[INFO] 使用本地动作文件: {args_cli.motion_file}")
    
    # 创建环境配置
    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 直接设置本地动作文件路径（绕过 WandB Registry）
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    
    # 创建训练配置
    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg.run_name = args_cli.run_name
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    
    # 设置日志目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 日志目录: {log_root_path}")
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # 创建环境
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg)
    
    # 包装环境
    env = RslRlVecEnvWrapper(env)
    
    # 创建训练器
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 保存配置
    print(f"[INFO] 保存配置到: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # 开始训练
    print("[INFO] 开始训练...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # 关闭环境
    env.close()
    
    print(f"[INFO] 训练完成！模型保存在: {log_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()







