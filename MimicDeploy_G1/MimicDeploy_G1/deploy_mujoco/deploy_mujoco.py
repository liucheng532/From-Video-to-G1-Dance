import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.realtime_keyboard_controller import RealtimeKeyboardController, CommandType
from omegaconf import DictConfig
import hydra    



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

@hydra.main(config_path="config", config_name="mujoco")
def main(cfg: DictConfig):
    xml_path = os.path.join(PROJECT_ROOT, cfg.xml_path)
    simulation_dt = cfg.simulation_dt
    control_decimation = cfg.control_decimation
    tau_limit = np.array(cfg.tau_limit)
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    print(f"num_joints: {num_joints}")
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    # 获取 torso_link 的 body_id（用于 MotionTracking）
    torso_body_name = "torso_link"
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
    if torso_body_id == -1:
        print(f"Warning: Body '{torso_body_name}' not found, MotionTracking may not work correctly")
        torso_body_id = 0  # fallback to world
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)
    
    # 使用实时键盘控制器
    controller = RealtimeKeyboardController()
    
    # 初始化机器人姿态 - 发送位置重置命令
    print("\n正在初始化机器人姿态...")
    state_cmd.skill_cmd = FSMCommand.POS_RESET
    print("提示: 机器人将自动执行位置重置，稳定后即可开始控制")
    print("建议: 先按 '3' 进入运动模式，然后按住 WASD/QE 键控制移动\n")
    
    Running = True
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            sim_start_time = time.time()
            while viewer.is_running() and Running and controller.is_running():
                try:
                    # 更新控制器，处理命令队列
                    controller.update()
                    
                    # 获取技能命令
                    skill_cmd = controller.get_skill_command()
                    if skill_cmd is not None:
                        if skill_cmd == CommandType.EXIT:
                            Running = False
                        elif skill_cmd == CommandType.PASSIVE:
                            state_cmd.skill_cmd = FSMCommand.PASSIVE
                        elif skill_cmd == CommandType.POS_RESET:
                            state_cmd.skill_cmd = FSMCommand.POS_RESET
                        elif skill_cmd == CommandType.STAND_UP:
                            state_cmd.skill_cmd = FSMCommand.STAND_UP
                        elif skill_cmd == CommandType.LOCO:
                            state_cmd.skill_cmd = FSMCommand.LOCO
                        elif skill_cmd == CommandType.SKILL_1:
                            state_cmd.skill_cmd = FSMCommand.SKILL_1
                        elif skill_cmd == CommandType.SKILL_2:
                            state_cmd.skill_cmd = FSMCommand.SKILL_2
                        elif skill_cmd == CommandType.MOTION_TRACKING:
                            state_cmd.skill_cmd = FSMCommand.MOTION_TRACKING
                        # 注意：SKILL_3, SKILL_4, SKILL_5 已删除，对应键盘6,7,8键暂时无效
                    
                    # 获取速度命令
                    vel_x, vel_y, vel_yaw = controller.get_velocity_command()
                    state_cmd.vel_cmd[0] = vel_x   # 前后速度
                    state_cmd.vel_cmd[1] = vel_y   # 左右速度
                    state_cmd.vel_cmd[2] = vel_yaw # 旋转速度
                    
                    step_start = time.time()
                    
                    tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                    tau = np.clip(tau, -tau_limit, tau_limit)
                    d.ctrl[:] = tau
                    mujoco.mj_step(m, d)
                    FSM_controller.sim_counter += 1
                    if FSM_controller.sim_counter % control_decimation == 0:
                        
                        qj = d.qpos[7:]
                        dqj = d.qvel[6:]
                        quat = d.qpos[3:7]
                        
                        omega = d.qvel[3:6] 
                        gravity_orientation = get_gravity_orientation(quat)
                        
                        # 获取 torso_link 的位置和姿态（用于 MotionTracking）
                        torso_pos = d.xpos[torso_body_id].copy()
                        torso_quat = d.xquat[torso_body_id].copy()
                        
                        state_cmd.q = qj.copy()
                        state_cmd.dq = dqj.copy()
                        state_cmd.gravity_ori = gravity_orientation.copy()
                        state_cmd.ang_vel = omega.copy()
                        state_cmd.base_pos = torso_pos  # 使用 torso_link 的位置
                        state_cmd.base_quat = torso_quat  # 使用 torso_link 的四元数
                        
                        FSM_controller.run()
                        policy_output_action = policy_output.actions.copy()
                        kps = policy_output.kps.copy()
                        kds = policy_output.kds.copy()
                    viewer.sync()
                    time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                except ValueError as e:
                    print(f"仿真错误: {e}")
                except Exception as e:
                    print(f"未预期的错误: {e}")
                    break
    except KeyboardInterrupt:
        print("\n\n收到中断信号 (Ctrl+C)，正在退出...")
    finally:
        # 清理控制器资源
        print("正在清理资源...")
        controller.quit()
        print("程序已安全退出")

if __name__ == "__main__":
    main()
