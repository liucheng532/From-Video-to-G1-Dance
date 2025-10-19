from isaaclab.utils import configclass

from robocup_lab.robots.booster_t1 import BOOSTER_T1_CFG

from robocup_lab.tasks.ball_dribble.ball_env_cfg import BallDribbleBaseEnvCfg


@configclass
class BoosterT1BallDribbleEnvCfg(BallDribbleBaseEnvCfg):
    """Single-stage ball-dribble environment for Booster T1."""

    base_link_name = "Trunk"
    foot_link_regex = ".*_foot_link"
    torque_joint_patterns = [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
    illegal_contact_body_patterns = [
        "Trunk",
        "Waist",
        "H.*",
        "AL.*",
        "AR.*",
        "left_hand_link",
        "right_hand_link",
    ]

    def __post_init__(self):
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        super().__post_init__()

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.rewards.joint_torques_l2.weight = -3.0e-7
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.action_rate_l2.weight = -0.075

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)

        if type(self) is BoosterT1BallDribbleEnvCfg:
            self.disable_zero_weight_rewards()


@configclass
class BoosterT1BallDribbleBodyTrackEnvCfg(BoosterT1BallDribbleEnvCfg):
    """Ball dribble env for Booster T1 with body velocity tracking enabled."""

    track_body_velocity_weight = 3.0

    def __post_init__(self):
        super().__post_init__()
        if type(self) is BoosterT1BallDribbleBodyTrackEnvCfg:
            self.disable_zero_weight_rewards()
