from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints):
        # robot state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.dq = np.zeros(num_joints, dtype=np.float32)
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        self.gravity_ori = np.array([0., 0., 1.])
        self.ang_vel = np.zeros(3)
        # base/torso state (for motion tracking)
        self.base_pos = np.zeros(3, dtype=np.float32)      # base position in world frame
        self.base_quat = np.array([1., 0., 0., 0.], dtype=np.float32)  # base quaternion [w, x, y, z]
        # joy cmd
        self.vel_cmd = np.zeros(3)
        self.skill_cmd = FSMCommand.INVALID
        # skill change cmd
        # self.skill_set = FSMCommand.SKILL_1

class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)
        