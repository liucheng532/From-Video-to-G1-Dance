from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.dance.Dance import Dance
from policy.host.host import HOST
from policy.skill_cooldown.SkillCooldown import SkillCooldown
from policy.skill_cast.SkillCast import SkillCast
from policy.montion_tracking.MotionTracking import MotionTracking
from FSM.FSMState import *
import time
from common.ctrlcomp import *
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        self.sim_counter = 0
        self.FSMmode = FSMMode.NORMAL
        
        # 只保留需要的策略模式
        self.passive_mode = PassiveMode(state_cmd, policy_output)       # 阻尼保护模式
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)         
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.dance_policy = Dance(state_cmd, policy_output)
        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.skill_cast_policy = SkillCast(state_cmd, policy_output)
        self.host_policy = HOST(state_cmd, policy_output)
        self.motion_tracking_policy = MotionTracking(state_cmd, policy_output)
        
        # 创建策略映射表，避免长if-elif链，提高查找效率
        self.policy_map = {
            FSMStateName.PASSIVE: self.passive_mode,
            FSMStateName.FIXEDPOSE: self.fixed_pose_1,
            FSMStateName.LOCOMODE: self.loco_policy,
            FSMStateName.SKILL_Dance: self.dance_policy,
            FSMStateName.SKILL_COOLDOWN: self.skill_cooldown_policy,
            FSMStateName.SKILL_CAST: self.skill_cast_policy,
            FSMStateName.STANDMODE: self.host_policy,
            FSMStateName.SKILL_MotionTracking: self.motion_tracking_policy,
        }
        
        print("initalized all policies!!!")
        
        self.cur_policy = self.passive_mode             # 当前policy
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        start_time = time.time()
        if(self.FSMmode == FSMMode.NORMAL): 
            self.cur_policy.run()
            nextPolicyName = self.cur_policy.checkChange()
            
            if(nextPolicyName != self.cur_policy.name):
                # change policy
                self.FSMmode = FSMMode.CHANGE
                self.cur_policy.exit()
                self.get_next_policy(nextPolicyName)
                print("Switched to ", self.cur_policy.name_str)
        
        elif(self.FSMmode == FSMMode.CHANGE):
            self.cur_policy.enter()
            self.sim_counter = 0
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()
            
        # self.absoluteWait(self.cur_policy.control_horzion,self.start_time)
        end_time = time.time()
        # print("time cusume: ", end_time - start_time)

    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        # 使用映射表，O(1)查找，提高效率
        if policy_name in self.policy_map:
            self.cur_policy = self.policy_map[policy_name]
        else:
            print(f"Unknown policy: {policy_name}, staying in current policy")
            
        
        