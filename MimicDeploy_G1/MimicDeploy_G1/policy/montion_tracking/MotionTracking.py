from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnxruntime
import mujoco
import os


class MotionTracking(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_MotionTracking
        self.name_str = "skill_motion_tracking"
        self.timestep = 0
        
        # 安全保护机制
        self.safety_enabled = True
        self.max_compute_time = 0.015  # 15ms最大计算时间
        self.emergency_stop = False
        self.last_valid_action = None
        self.compute_time_history = []
        self.max_history_size = 10
        
        # 定义关节顺序映射（MuJoCo XML顺序 vs 训练时的序列顺序）
        self.joint_xml = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
            "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        self.joint_seq = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint',
            'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
            'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
            'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
            'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
            'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]
        
        # 预计算关节索引映射，避免重复查找
        self._joint_xml_to_seq = {joint: i for i, joint in enumerate(self.joint_xml)}
        self._joint_seq_to_xml = {joint: i for i, joint in enumerate(self.joint_seq)}
        
        # 预分配数组，避免重复创建
        self._temp_quat = np.zeros(4, dtype=np.float32)
        self._temp_matrix = np.zeros(9, dtype=np.float32)
        self._temp_obs = np.zeros(154, dtype=np.float32)
        self._temp_motioninput = np.zeros(58, dtype=np.float32)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "Motion.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["policy_path"])
            self.motion_path = os.path.join(current_dir, "motion", config["motion_path"])
            self.kps = np.array(config["stiffness"], dtype=np.float32)
            self.kds = np.array(config["damping"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.default_angles_seq = np.array(config["default_angles_seq"], dtype=np.float32)
            self.action_scale_seq = np.array(config["action_scale_seq"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.motion_length = config.get("motion_length", 10.0)
            
            # 加载动作数据文件 (.npz)
            print(f"Loading motion data from {self.motion_path}...")
            motion = np.load(self.motion_path)
            self.motionpos = motion["body_pos_w"]        # 参考动作的body位置
            self.motionquat = motion["body_quat_w"]      # 参考动作的body四元数
            self.motioninputpos = motion["joint_pos"]    # 参考动作的关节位置
            self.motioninputvel = motion["joint_vel"]    # 参考动作的关节速度
            
            # 获取动作总帧数
            self.total_frames = self.motioninputpos.shape[0]
            print(f"Motion data loaded: {self.total_frames} frames")
            
            # 初始化观测和动作
            self.obs = np.zeros(self.num_obs, dtype=np.float32)
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)
            
            # 加载 ONNX 模型
            print(f"Loading policy from {self.onnx_path}...")
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            
            # 预热模型
            for _ in range(10):
                obs_tensor = self.obs[np.newaxis, :]
                self.ort_session.run(
                    ['actions'], 
                    {'obs': obs_tensor, 'time_step': np.array([[0.0]], dtype=np.float32)}
                )
                    
            print("MotionTracking policy initialized!")
    
    def enter(self):
        """进入 MotionTracking 模式时的初始化"""
        self.timestep = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        # 只刷新yaw轴的世界坐标系，避免pitch和roll的误差累积
        # 获取motion数据期望的初始姿态
        motion_initial_quat = self.motionquat[0, 9, :]  # motion数据第一帧的torso四元数
        
        # 获取当前机器人的实际姿态
        current_quat = self.state_cmd.base_quat.reshape(-1)
        
        # 提取yaw角度（只处理水平旋转）
        from scipy.spatial.transform import Rotation as R
        
        # 将四元数转换为旋转矩阵，提取yaw角度
        current_r = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        motion_r = R.from_quat([motion_initial_quat[1], motion_initial_quat[2], motion_initial_quat[3], motion_initial_quat[0]])
        
        current_yaw = current_r.as_euler('xyz')[2]  # 当前yaw角度
        motion_yaw = motion_r.as_euler('xyz')[2]    # motion期望的yaw角度
        
        # 计算yaw角度差
        yaw_diff = current_yaw - motion_yaw
        
        # 创建只包含yaw变换的四元数
        yaw_transform_r = R.from_euler('z', yaw_diff)
        self.world_transform_quat_scipy = yaw_transform_r.as_quat()  # [x,y,z,w]格式
        self.world_transform_quat = np.array([
            self.world_transform_quat_scipy[3],  # w
            self.world_transform_quat_scipy[0],  # x
            self.world_transform_quat_scipy[1],  # y
            self.world_transform_quat_scipy[2]   # z
        ], dtype=np.float32)  # 转换为[w,x,y,z]格式
        
        print(f"\n开始执行 MotionTracking 动作序列 (共 {self.total_frames} 帧)")
        print(f"已刷新世界坐标系，机器人将从当前位置开始执行动作")
        
    def subtract_frame_transforms(self, pos_a, quat_a, pos_b, quat_b):
        """
        与IsaacLab中subtract_frame_transforms完全相同的实现（一维版本）
        计算从坐标系A到坐标系B的相对变换
        
        参数:
            pos_a: 坐标系A的位置 (3,)
            quat_a: 坐标系A的四元数 (4,) [w, x, y, z]格式
            pos_b: 坐标系B的位置 (3,)
            quat_b: 坐标系B的四元数 (4,) [w, x, y, z]格式
            
        返回:
            rel_pos: B相对于A的位置 (3,)
            rel_quat: B相对于A的旋转四元数 (4,) [w, x, y, z]格式
        """
        # 计算相对位置: pos_B_to_A = R_A^T * (pos_B - pos_A)
        rotm_a = np.zeros(9)
        mujoco.mju_quat2Mat(rotm_a, quat_a)
        rotm_a = rotm_a.reshape(3, 3)
        
        rel_pos = rotm_a.T @ (pos_b - pos_a)
        
        # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
        rel_quat = self.quaternion_multiply(self.quaternion_conjugate(quat_a), quat_b)
        
        # 确保四元数归一化（与IsaacLab保持一致）
        rel_quat = rel_quat / np.linalg.norm(rel_quat)
        
        return rel_pos, rel_quat
    
    def quaternion_conjugate(self, q):
        """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quaternion_multiply(self, q1, q2):
        """四元数乘法: q1 ⊗ q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
        
    def run(self):
        """每个控制步执行一次"""
        import time
        start_time = time.time()
        
        # 检查是否超出动作帧数
        if self.timestep >= self.total_frames:
            print("\n动作序列执行完毕")
            return
        
        # 安全保护：如果处于紧急停止状态，使用上一个有效动作
        if self.emergency_stop and self.last_valid_action is not None:
            print("WARNING: MotionTracking in emergency stop mode, using last valid action")
            self.policy_output.actions = self.last_valid_action.copy()
            self.policy_output.kps = self.kps
            self.policy_output.kds = self.kds
            return
        
        # 获取当前机器人状态
        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1)
        qj = self.state_cmd.q.reshape(-1)  # 29-DOF 关节位置
        dqj = self.state_cmd.dq.reshape(-1)  # 29-DOF 关节速度
        ang_vel = self.state_cmd.ang_vel.reshape(-1)  # 角速度
        
        # 获取 base 的位置和姿态（从 state_cmd 中获取，由 deploy_mujoco 传入）
        position = self.state_cmd.base_pos.reshape(-1)  # base 位置
        quaternion = self.state_cmd.base_quat.reshape(-1)  # base 四元数 [w, x, y, z]
        
        # 获取当前帧的参考动作数据
        motioninput = np.concatenate((
            self.motioninputpos[self.timestep, :],  # 29维关节位置
            self.motioninputvel[self.timestep, :]   # 29维关节速度
        ), axis=0)  # 总共 58 维
        
        motionposcurrent = self.motionpos[self.timestep, 9, :]  # 参考动作的 torso_link 位置
        motionquatcurrent = self.motionquat[self.timestep, 9, :]  # 参考动作的 torso_link 四元数
        
        # 使用刷新后的世界坐标系：将motion数据转换到当前机器人的坐标系
        # 应用世界坐标变换到motion数据的姿态
        transformed_motion_quat = self.quaternion_multiply(
            self.world_transform_quat, 
            motionquatcurrent
        )
        
        # 计算相对变换（使用刷新后的坐标系）
        _, anchor_quat = self.subtract_frame_transforms(position, quaternion, motionposcurrent, transformed_motion_quat)
        
        # 将相对四元数转换为旋转矩阵，取前两列并展平为 6 维
        anchor_ori = np.zeros(9)
        mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
        anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
        anchor_ori = anchor_ori.reshape(-1,)
        
        # 构建观测向量 (总共 154 维)
        offset = 0
        self.obs[offset:offset+58] = motioninput  # 参考动作的关节位置和速度
        offset += 58
        self.obs[offset:offset+6] = anchor_ori  # 相对旋转矩阵（6维）
        offset += 6
        self.obs[offset:offset+3] = ang_vel  # 角速度
        offset += 3
        
        # 将关节从 XML 顺序转换为 Seq 顺序（使用预计算的映射，提高效率）
        qj_obs_seq = np.array([qj[self._joint_xml_to_seq[joint]] for joint in self.joint_seq])
        self.obs[offset:offset+29] = qj_obs_seq - self.default_angles_seq
        offset += 29
        
        dqj_obs_seq = np.array([dqj[self._joint_xml_to_seq[joint]] for joint in self.joint_seq])
        self.obs[offset:offset+29] = dqj_obs_seq
        offset += 29
        
        self.obs[offset:offset+29] = self.action_buffer  # 上一步的动作
        
        # 调用 policy 获取动作
        obs_tensor = self.obs[np.newaxis, :]
        action = self.ort_session.run(
            ['actions'], 
            {'obs': obs_tensor, 'time_step': np.array([[float(self.timestep)]], dtype=np.float32)}
        )[0]
        
        action = np.asarray(action).reshape(-1)
        self.action = action.copy()
        self.action_buffer = action.copy()
        
        # 将动作转换为目标关节位置（Seq 顺序）
        target_dof_pos_seq = self.default_angles_seq + self.action * self.action_scale_seq
        
        # 将 Seq 顺序转换回 XML 顺序（MuJoCo 需要的顺序，使用预计算的映射）
        target_dof_pos = np.array([
            target_dof_pos_seq[self._joint_seq_to_xml[joint]] for joint in self.joint_xml
        ])
        
        # 输出到 policy_output
        self.policy_output.actions = target_dof_pos
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds
        
        # 保存有效动作用于安全保护
        self.last_valid_action = target_dof_pos.copy()
        
        # 性能监控和安全检查
        end_time = time.time()
        compute_time = end_time - start_time
        
        # 记录计算时间历史
        self.compute_time_history.append(compute_time)
        if len(self.compute_time_history) > self.max_history_size:
            self.compute_time_history.pop(0)
        
        # 安全检查：如果计算时间过长，触发紧急停止
        if self.safety_enabled and compute_time > self.max_compute_time:
            print(f"WARNING: MotionTracking compute time {compute_time*1000:.2f}ms exceeds limit {self.max_compute_time*1000:.2f}ms")
            self.emergency_stop = True
            # 使用上一个有效动作
            if self.last_valid_action is not None:
                self.policy_output.actions = self.last_valid_action.copy()
        
        # 如果连续多次计算时间正常，解除紧急停止
        if self.emergency_stop and len(self.compute_time_history) >= 3:
            recent_times = self.compute_time_history[-3:]
            if all(t < self.max_compute_time for t in recent_times):
                print("MotionTracking emergency stop cleared")
                self.emergency_stop = False
        
        # 更新时间步
        self.timestep += 1
        
        # 显示进度
        motion_time = self.timestep * 0.02
        total_time = self.total_frames * 0.02
        print(progress_bar(motion_time, total_time), end="", flush=True)
    
    def exit(self):
        """退出 MotionTracking 模式"""
        self.timestep = 0
        self.emergency_stop = False
        self.compute_time_history.clear()
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.compute_time_history:
            return None
        
        times = np.array(self.compute_time_history) * 1000  # 转换为毫秒
        return {
            'avg_time_ms': np.mean(times),
            'max_time_ms': np.max(times),
            'min_time_ms': np.min(times),
            'current_time_ms': times[-1] if times.size > 0 else 0,
            'emergency_stop': self.emergency_stop,
            'samples': len(times)
        }
    def checkChange(self):
        """检查状态切换"""
        # 如果动作执行完毕，自动返回 LOCO 模式
        if self.timestep >= self.total_frames:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_MotionTracking
