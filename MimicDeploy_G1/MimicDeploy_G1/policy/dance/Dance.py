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


class Dance(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_Dance
        self.name_str = "skill_dance"
        self.motion_phase = 0
        self.counter_step = 0
        self.ref_motion_phase = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "Dance.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles =  np.array(config["default_angles"], dtype=np.float32)
            self.dof23_index =  np.array(config["dof23_index"], dtype=np.int32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.history_length = config["history_length"]
            self.motion_length = config["motion_length"]
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            self.obs_history = np.zeros((self.history_length, self.num_obs), dtype=np.float32)
            
            self.ang_vel_buf = np.zeros(3 * self.history_length, dtype=np.float32)
            self.proj_g_buf = np.zeros(3 * self.history_length, dtype=np.float32)
            self.dof_pos_buf = np.zeros(23 * self.history_length, dtype=np.float32)
            self.dof_vel_buf = np.zeros(23 * self.history_length, dtype=np.float32)
            self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
            self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
            
            # load policy
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            for _ in range(50):
                obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).cpu().numpy()
                obs_tensor = obs_tensor.astype(np.float32)
                self.ort_session.run(None, {self.input_name: obs_tensor})[0]
                    
            print("Dance policy initializing ...")
    
    def enter(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase = 0.
        self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
        self.motion_time = 0
        self.counter_step = 0
        
        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs)
        self.action = np.zeros(self.num_actions)
        self.obs_history = np.zeros((self.history_length, self.num_obs), dtype=np.float32)
        
        self.ang_vel_buf = np.zeros(3 * self.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * self.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
        pass
        
        
    def run(self):
        
        gravity_orientation = self.state_cmd.gravity_ori.reshape(-1)
        qj = self.state_cmd.q.reshape(-1)
        dqj = self.state_cmd.dq.reshape(-1)
        ang_vel = self.state_cmd.ang_vel.reshape(-1)
        
        qj_23dof = qj[self.dof23_index].copy()
        dqj_23dof = dqj[self.dof23_index].copy()
        default_angles_23dof = self.default_angles[self.dof23_index].copy()
        qj_23dof = (qj_23dof - default_angles_23dof) * self.dof_pos_scale
        dqj_23dof = dqj_23dof * self.dof_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale
        
        self.ang_vel_buf = np.concatenate((ang_vel, self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        self.proj_g_buf = np.concatenate((gravity_orientation, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((qj_23dof, self.dof_pos_buf[:-23] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dqj_23dof, self.dof_vel_buf[:-23] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-23] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate((np.array([min(self.ref_motion_phase,1.0)]), self.ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)
        
        mimic_history_obs_buf = np.concatenate((self.action_buf, 
                                                self.ang_vel_buf, 
                                                self.dof_pos_buf, 
                                                self.dof_vel_buf, 
                                                self.proj_g_buf, 
                                                self.ref_motion_phase_buf
                                                ), 
                                                axis=-1, dtype=np.float32)
        
        mimic_obs_buf = np.concatenate((self.action,
                                        ang_vel,
                                        qj_23dof,
                                        dqj_23dof,
                                        mimic_history_obs_buf,
                                        gravity_orientation,
                                        np.array([min(self.ref_motion_phase,1.0)])
                                        ),
                                        axis=-1, dtype=np.float32)
        
        mimic_obs_tensor = torch.from_numpy(mimic_obs_buf).unsqueeze(0).cpu().numpy()
        self.action = np.squeeze(self.ort_session.run(None, {self.input_name: mimic_obs_tensor})[0])
        target_dof_pos = np.zeros(29)
        target_dof_pos[:15] = self.action[:15] * self.action_scale + self.default_angles[:15]
        target_dof_pos[15:19] = self.action[15:19] * self.action_scale + self.default_angles[15:19]
        target_dof_pos[22:26] = self.action[19:] * self.action_scale + self.default_angles[22:26]
        
        target_dof_pos[19:22] = self.default_angles[19:22]
        target_dof_pos[26:29] = self.default_angles[26:29]
        
        self.policy_output.actions = target_dof_pos
        self.policy_output.kps = self.kps
        self.policy_output.kds = self.kds
        
        # update motion phase
        self.counter_step += 1
        motion_time = self.counter_step * 0.02
        self.ref_motion_phase = motion_time / self.motion_length
        motion_time = min(motion_time, self.motion_length)
        print(progress_bar(motion_time, self.motion_length), end="", flush=True)
    
    def exit(self):
        self.action = np.zeros(23, dtype=np.float32)
        self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase = 0.
        self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)
        self.motion_time = 0
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
            return FSMStateName.SKILL_Dance