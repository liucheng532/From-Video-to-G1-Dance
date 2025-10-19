from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os


class HOST(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.STANDMODE
        self.name_str = "skill_stand"
        self.counter_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.unactuated_time = 30
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "host.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.jit_model = os.path.join(current_dir, "model", config["jit_model_path"])
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles =  np.array(config["default_angles"], dtype=np.float32)
            self.dof23_index =  np.array(config["dof23_index"], dtype=np.int32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.joint_limit_min = np.array(config["joint_limit_min"], dtype=np.float32)
            self.joint_limit_max = np.array(config["joint_limit_max"], dtype=np.float32)
            self.soft_dof_pos_limit = 1.1
            for i in range(len(self.joint_limit_min)):
                # soft limits
                if i != 5 and i != 11 and i !=4 and i != 10:
                    m = (self.joint_limit_min[i] + self.joint_limit_max[i]) / 2
                    r = self.joint_limit_max[i] - self.joint_limit_min[i]
                    self.joint_limit_min[i] = m - 0.5 * r * self.soft_dof_pos_limit
                    self.joint_limit_max[i] = m + 0.5 * r * self.soft_dof_pos_limit

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]                    # 76
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.project_gravity_scale = config["project_gravity_scale"]
            self.clip_actions = config["clip_actions"]
            self.clip_obs = config["clip_obs"]
            self.history_length = config["history_length"]

            self.control_dt = config["control_dt"]
            self.total_obs_length = self.num_obs * (self.history_length + 1)        # 456
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs_tensor = torch.zeros(self.total_obs_length, dtype=torch.float32, device=self.device)
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            
            # load policy
            self.policy = torch.jit.load(self.jit_model).to(self.device)
            for _ in range(50):
                self.policy(self.obs_tensor).detach().cpu().numpy().squeeze()
                    
            print("HoST policy initializing ...")
    
    def enter(self):
        # self.policy_output.kps = self.kps
        # self.policy_output.kds = self.kds
        print("HoST policy enter ...")
        self.counter_step = 0
        
        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_tensor = torch.zeros(self.total_obs_length, dtype=torch.float32, device=self.device)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        pass
        
        
    def run(self):
        # 1. 获取当前state
        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1) * self.project_gravity_scale
        qj = self.state_cmd.q.reshape(-1)
        dqj = self.state_cmd.dq.reshape(-1)
        ang_vel = self.state_cmd.ang_vel.reshape(-1)
        # 2. 计算observations
        qj_23dof = qj[self.dof23_index].copy()          # 从29dof中选出23dof
        dqj_23dof = dqj[self.dof23_index].copy()
        qj_23dof = qj_23dof * self.dof_pos_scale
        dqj_23dof = dqj_23dof * self.dof_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale
        action_rescale = self.action_scale * np.ones(1, dtype=np.float32)
        self.real_episode_length_buf += 1

        mimic_obs_buf = np.concatenate((
                                ang_vel,
                                gravity_orientation,
                                qj_23dof,
                                dqj_23dof,
                                self.action,
                                action_rescale + (np.random.uniform(0, 1, size=1) - 0.5) * 0.05
                                ),
                                axis=-1, dtype=np.float32)
        
        current_obs = torch.from_numpy(mimic_obs_buf).to(self.device)
        current_obs *= self.real_episode_length_buf > self.unactuated_time
        current_obs = torch.clip(current_obs, -self.clip_obs, self.clip_obs)

        self.obs_tensor = torch.cat((self.obs_tensor[self.num_obs : self.total_obs_length], current_obs), dim=-1)
        
        # 3. 推理得出 actions
        action = self.policy(self.obs_tensor.detach()).detach().cpu().numpy().squeeze()
        action = np.clip(action, -self.clip_actions, self.clip_actions)
        action *= self.real_episode_length_buf.cpu().numpy() > self.unactuated_time
        self.action = action.copy()  

        action_scaled = self.action * self.action_scale
        target_dof_pos = np.concatenate((action_scaled[:13], np.zeros(2), action_scaled[13:18],  \
                                                np.zeros(2), action_scaled[18:23], np.zeros(2))) + qj       # 29
        target_dof_pos[13] = 0.
        target_dof_pos[14] = 0.  
        target_dof_pos[np.r_[20:22, 27:29]] = 0.

        if (target_dof_pos - self.joint_limit_min < 0).any():
            out_range_idx = np.where(target_dof_pos - self.joint_limit_min < 0)
            target_dof_pos[out_range_idx] = self.joint_limit_min[out_range_idx] + 0.05
            print(f"[INFO] joint limit out of min. index: {np.where(target_dof_pos - self.joint_limit_min < 0)}")
        if (target_dof_pos - self.joint_limit_max > 0).any():
            out_range_idx = np.where(target_dof_pos - self.joint_limit_max > 0)
            target_dof_pos[out_range_idx] = self.joint_limit_max[out_range_idx] - 0.05
            print(f"[INFO] joint limit out of max. index: {np.where(target_dof_pos - self.joint_limit_max > 0)}")

        if self.real_episode_length_buf <= 1:
            target_dof_pos = self.default_angles.copy()
        else:
            self.policy_output.actions = target_dof_pos
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds
        

    def exit(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.real_episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.counter_step = 0
        print()

    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.STANDMODE