import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time
import os
import yaml

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation_real, transform_imu_data, transform_pelvis_to_torso_complete
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()
        self.num_joints = config.num_joints
        self.control_dt = config.control_dt
        
        
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        
        # inital connection
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.wait_for_low_state()
        
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        
        self.policy_output_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kps = np.zeros(self.num_joints, dtype=np.float32)
        self.kds = np.zeros(self.num_joints, dtype=np.float32)
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0,0,-1], dtype=np.float32)
        
        self.state_cmd = StateAndCmd(self.num_joints)                   # 定义了机器人的state
        self.policy_output = PolicyOutput(self.num_joints)              # 定义了action, kp, kd
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.running = True
        self.counter_over_time = 0
        
        
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        
    def run(self):
        try:
            # if(self.counter_over_time >= config.error_over_time):
            #     raise ValueError("counter_over_time >= error_over_time")
            
            loop_start_time = time.time()
            
            ## 1. 检测遥控器按键
            if self.remote_controller.is_button_pressed(KeyMap.F1):             # F1阻尼保护模式
                self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            if self.remote_controller.is_button_pressed(KeyMap.start):
                self.state_cmd.skill_cmd = FSMCommand.POS_RESET

            if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.L1):      # 摔倒爬起, L1+X
                self.state_cmd.skill_cmd = FSMCommand.STAND_UP

            if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.LOCO
            if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_1
            if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.SKILL_2
            # L1+A 暂时不映射到任何模式，可以后续添加
            # if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.L1):
            #     self.state_cmd.skill_cmd = FSMCommand.SKILL_5
            if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                self.state_cmd.skill_cmd = FSMCommand.MOTION_TRACKING # R1 + B
            # if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.L1):
            #     self.state_cmd.skill_cmd = FSMCommand.SKILL_4
            
            self.state_cmd.vel_cmd[0] =  self.remote_controller.ly              # 速度指令
            self.state_cmd.vel_cmd[1] =  self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] =  self.remote_controller.rx * -1
            
            ## 2. 获取底层状态
            for i in range(self.num_joints):
                self.qj[i] = self.low_state.motor_state[i].q            # 关节位置
                self.dqj[i] = self.low_state.motor_state[i].dq          # 关节速度

            # imu_state quaternion: w, x, y, z
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
            
            gravity_orientation = get_gravity_orientation_real(quat)
            
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = ang_vel.copy()
            
            # 对于真机，计算 torso 的姿态（用于 MotionTracking）
            # 获取腰部三个关节的位置（waist_yaw=12, waist_roll=13, waist_pitch=14）
            waist_yaw = self.qj[12]
            waist_roll = self.qj[13]
            waist_pitch = self.qj[14]
            
            # 将 pelvis 的 IMU 四元数转换为 torso 的四元数
            torso_quat = transform_pelvis_to_torso_complete(waist_yaw, waist_roll, waist_pitch, quat)
            
            self.state_cmd.base_pos = np.zeros(3, dtype=np.float32)  # 真机无全局位置
            self.state_cmd.base_quat = torso_quat  # 使用 torso 的四元数
            
            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()
        
            
            # 设定电机指令
            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
                
            # send the command
            # create_damping_cmd(controller.low_cmd) # only for debug
            self.send_cmd(self.low_cmd)         # 发送指令
            
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            if(delta_time < self.control_dt):
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
                # 性能监控：如果是MotionTracking模式，显示性能统计
                if hasattr(self.FSM_controller.cur_policy, 'get_performance_stats'):
                    stats = self.FSM_controller.cur_policy.get_performance_stats()
                    if stats and stats['samples'] > 0:
                        # 每100步显示一次性能统计
                        if self.counter_over_time % 100 == 0:
                            print(f"\nMotionTracking性能: 平均{stats['avg_time_ms']:.2f}ms, "
                                f"最大{stats['max_time_ms']:.2f}ms, 当前{stats['current_time_ms']:.2f}ms, "
                                f"紧急停止: {stats['emergency_stop']}")
            pass
        except ValueError as e:
            print(str(e))
            pass
        
        pass
        
        
if __name__ == "__main__":
    config = Config()
    # Initialize DDS communication
    ChannelFactoryInitialize(0, config.net)
    
    controller = Controller(config)
    
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):           # 按下select键退出
                break
        except KeyboardInterrupt:
            break
    
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    