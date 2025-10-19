from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

from robocup_lab.tasks.ball_dribble.ball_env_cfg import BallDribbleBaseEnvCfg


@configclass
class UnitreeG1BallDribbleEnvCfg(BallDribbleBaseEnvCfg):
    """Single-stage ball-dribble environment for Unitree G1."""

    base_link_name = "torso_link"
    foot_link_regex = ".*_ankle_roll_link"
    observed_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]
    torque_joint_patterns = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
    illegal_contact_body_patterns = [
        "torso_link",
        ".*hand.*",
        ".*wrist.*",
        ".*forearm.*",
    ]

    def __post_init__(self):
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        super().__post_init__()

        # Match previous locomotion regularization while keeping the new dribble rewards.
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.action_rate_l2.weight = -0.005

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)

        if type(self) is UnitreeG1BallDribbleEnvCfg:
            self.disable_zero_weight_rewards()


@configclass
class UnitreeG1BallDribbleBodyTrackEnvCfg(UnitreeG1BallDribbleEnvCfg):
    """Ball dribble env that also rewards body velocity tracking."""

    track_body_velocity_weight = 2.0

    def __post_init__(self):
        super().__post_init__()
        if type(self) is UnitreeG1BallDribbleBodyTrackEnvCfg:
            self.disable_zero_weight_rewards()
