"""Standalone configuration for the ball-dribble task."""

import copy
from dataclasses import MISSING
from typing import Dict, Optional, Sequence, Tuple

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robocup_lab.tasks.ball_dribble.mdp as dribble_mdp
import robocup_lab.tasks.locomotion.velocity.mdp as locomotion_mdp
from robocup_lab.assets import SOCCER_BALL_CFG


@configclass
class BallDribbleSceneCfg(InteractiveSceneCfg):
    """Flat terrain with a robot, a ball and contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    ball: Optional[RigidObjectCfg] = None
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class BallDribbleActionsCfg:
    """Action specifications for the policy."""

    joint_pos = locomotion_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class BallDribbleCommandsCfg:
    """Velocity command sampler for the base."""

    base_velocity = dribble_mdp.BallDribbleVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        heading_control_stiffness=3.0,
        debug_vis=True,
        planar_speed_range=(0.4, 1.3),
        heading_offset_range=(-0.7, 0.7),
        min_planar_speed=0.12,
        min_turn_speed=0.3,
    )


@configclass
class BallDribbleObservationsCfg:
    """Observation terms shared by policy and critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=locomotion_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=locomotion_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=locomotion_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=locomotion_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=locomotion_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=locomotion_mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        ball_position = ObsTerm(
            func=dribble_mdp.ball_pos_in_base,
            params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")},
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        ball_velocity = ObsTerm(
            func=dribble_mdp.ball_vel_in_base,
            params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        ball_spin = ObsTerm(
            func=dribble_mdp.ball_ang_vel,
            params={"ball_cfg": SceneEntityCfg("ball")},
            clip=(-20.0, 20.0),
            scale=0.25,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=locomotion_mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=locomotion_mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=locomotion_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=locomotion_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=locomotion_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        ball_position = ObsTerm(
            func=dribble_mdp.ball_pos_in_base,
            params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")},
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        ball_velocity = ObsTerm(
            func=dribble_mdp.ball_vel_in_base,
            params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        ball_spin = ObsTerm(
            func=dribble_mdp.ball_ang_vel,
            params={"ball_cfg": SceneEntityCfg("ball")},
            clip=(-20.0, 20.0),
            scale=0.25,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BallDribbleRewardsCfg:
    """Reward terms tailored for ball dribbling."""

    is_terminated = RewTerm(func=locomotion_mdp.is_terminated, weight=-200.0)

    flat_orientation_l2 = RewTerm(func=locomotion_mdp.flat_orientation_l2, weight=-0.2)
    base_height_l2 = RewTerm(
        func=locomotion_mdp.base_height_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "target_height": 0.0},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    joint_torques_l2 = RewTerm(
        func=locomotion_mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc_l2 = RewTerm(
        func=locomotion_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(func=locomotion_mdp.action_rate_l2, weight=-0.01)

    ball_directional_speed = RewTerm(
        func=dribble_mdp.ball_directional_speed,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "min_speed": 1.0e-3,
        },
    )
    ball_velocity_tracking = RewTerm(
        func=dribble_mdp.ball_velocity_tracking_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
            "axis_weights": (1.0, 1.0, 0.2),
        },
    )
    ball_height_penalty = RewTerm(
        func=dribble_mdp.ball_height_penalty,
        weight=-1.0,
        params={"ball_cfg": SceneEntityCfg("ball"), "target_height": 0.18, "tolerance": 0.05},
    )


@configclass
class BallDribbleEventCfg:
    """Domain-randomization and reset events."""

    randomize_rigid_body_material = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_mass_base = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=locomotion_mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )
    randomize_actuator_gains = EventTerm(
        func=locomotion_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_reset_joints = EventTerm(
        func=locomotion_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    randomize_reset_base = EventTerm(
        func=locomotion_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    randomize_apply_external_force_torque = EventTerm(
        func=locomotion_mdp.apply_external_force_torque,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "force_range": (-10.0, 10.0), "torque_range": (-10.0, 10.0)},
    )
    randomize_push_robot = EventTerm(
        func=locomotion_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(12.0, 18.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    reset_ball_state = EventTerm(
        func=dribble_mdp.reset_ball_near_base,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "offset_range": {"x": (0.3, 0.5), "y": (-0.1, 0.1), "z": (0.02, 0.1)},
            "lin_vel_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (0.0, 0.0)},
            "ang_vel_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.0, 1.0)},
        },
    )
    randomize_ball_material = EventTerm(
        func=dribble_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.35, 0.8),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.2, 0.5),
            "num_buckets": 16,
        },
    )
    randomize_ball_mass = EventTerm(
        func=dribble_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )


@configclass
class BallDribbleTerminationsCfg:
    """Termination conditions for the ball-dribble task."""

    time_out = DoneTerm(func=locomotion_mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=locomotion_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    illegal_contact = DoneTerm(
        func=locomotion_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    ball_far_from_robot = DoneTerm(
        func=dribble_mdp.ball_far_from_robot,
        params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot"), "threshold": 1.0},
    )
    ball_outside_front_sector = DoneTerm(
        func=dribble_mdp.ball_outside_front_sector,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "aperture_deg": 120.0,
            "grace_period_s": 0.3,
            "min_speed": 1.0e-3,
        },
    )
    robot_fallen = DoneTerm(
        func=dribble_mdp.robot_fallen,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "min_height": 0.35,
            "min_up_dot": 0.6,
        },
    )


@configclass
class BallDribbleBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration shared by all ball-dribble robots."""

    scene: BallDribbleSceneCfg = BallDribbleSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: BallDribbleActionsCfg = BallDribbleActionsCfg()
    observations: BallDribbleObservationsCfg = BallDribbleObservationsCfg()
    commands: BallDribbleCommandsCfg = BallDribbleCommandsCfg()
    rewards: BallDribbleRewardsCfg = BallDribbleRewardsCfg()
    terminations: BallDribbleTerminationsCfg = BallDribbleTerminationsCfg()
    events: BallDribbleEventCfg = BallDribbleEventCfg()

    # Robot-specific overrides
    base_link_name: str = ""
    foot_link_regex: str = ""
    observed_joint_names: Optional[Sequence[str]] = None
    torque_joint_patterns: Optional[Sequence[str]] = None

    # Ball behaviour parameters shared across robots
    ball_prim_path: str = "{ENV_REGEX_NS}/Ball"
    ball_spawn_position: Tuple[float, float, float] = (0.45, 0.0, 0.18)
    ball_reset_offset_range: Dict[str, Tuple[float, float]] = {"x": (0.3, 0.5), "y": (-0.1, 0.1), "z": (0.02, 0.1)}
    ball_reset_lin_vel_range: Dict[str, Tuple[float, float]] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (0.0, 0.0)}
    ball_reset_ang_vel_range: Dict[str, Tuple[float, float]] = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.0, 1.0)}
    ball_max_distance: float = 1.0
    ball_front_sector_aperture_deg: float = 120.0
    ball_front_sector_min_speed: float = 1.0e-3
    ball_front_sector_grace_s: float = 0.3
    randomize_push_interval_s: Tuple[float, float] = (12.0, 18.0)
    track_body_velocity_weight: float = 0.0
    robot_min_height: float = 0.35
    robot_min_up_dot: float = 0.6
    illegal_contact_body_patterns: Optional[Sequence[str]] = None

    def __post_init__(self):
        super().__post_init__()

        if SOCCER_BALL_CFG is None:
            raise ImportError("SOCCER_BALL_CFG is unavailable. Install Isaac Lab assets before using the ball task.")

        # Simulation settings
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Attach the ball asset and initial state
        self.scene.ball = SOCCER_BALL_CFG.replace(prim_path=self.ball_prim_path)
        self.scene.ball.spawn.prim_path = self.ball_prim_path
        self.scene.ball.init_state.pos = self.ball_spawn_position

        ball_entity = SceneEntityCfg("ball")
        robot_entity = SceneEntityCfg("robot")

        # Observation wiring for robot-specific joints
        if self.observed_joint_names is not None:
            joint_cfg = SceneEntityCfg("robot", joint_names=list(self.observed_joint_names), preserve_order=True)
            self.observations.policy.joint_pos.params["asset_cfg"] = copy.deepcopy(joint_cfg)
            self.observations.policy.joint_vel.params["asset_cfg"] = copy.deepcopy(joint_cfg)
            self.observations.critic.joint_pos.params["asset_cfg"] = copy.deepcopy(joint_cfg)
            self.observations.critic.joint_vel.params["asset_cfg"] = copy.deepcopy(joint_cfg)
            self.actions.joint_pos.joint_names = list(self.observed_joint_names)

        # Update reward parameters that depend on the robot model
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        if self.torque_joint_patterns is not None:
            torque_cfg = SceneEntityCfg("robot", joint_names=list(self.torque_joint_patterns))
            self.rewards.joint_torques_l2.params["asset_cfg"] = copy.deepcopy(torque_cfg)
            self.rewards.joint_acc_l2.params["asset_cfg"] = copy.deepcopy(torque_cfg)

        # Domain randomization hooks that rely on the base link
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_push_robot.interval_range_s = self.randomize_push_interval_s

        # Event parameters for the ball need to be deep-copied when using dataclasses
        self.events.reset_ball_state.params.update(
            {
                "offset_range": copy.deepcopy(self.ball_reset_offset_range),
                "lin_vel_range": copy.deepcopy(self.ball_reset_lin_vel_range),
                "ang_vel_range": copy.deepcopy(self.ball_reset_ang_vel_range),
            }
        )

        # Termination thresholds for the ball
        self.terminations.ball_far_from_robot.params["threshold"] = self.ball_max_distance
        self.terminations.ball_outside_front_sector.params.update(
            {
                "aperture_deg": self.ball_front_sector_aperture_deg,
                "grace_period_s": self.ball_front_sector_grace_s,
                "min_speed": self.ball_front_sector_min_speed,
            }
        )
        self.terminations.robot_fallen.params.update(
            {"min_height": self.robot_min_height, "min_up_dot": self.robot_min_up_dot}
        )

        # Tie contact-based terms to the targeted feet if provided
        contact_patterns = self.illegal_contact_body_patterns or [self.base_link_name]
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = list(contact_patterns)
        # Rewards relying on contact sensors can share the same pattern
        # Users can override or nullify these terms in subclasses if needed.

        # Ball-focused rewards should reference the active entities explicitly
        self.rewards.ball_directional_speed.params.update(
            {"ball_cfg": copy.deepcopy(ball_entity), "robot_cfg": copy.deepcopy(robot_entity)}
        )
        self.rewards.ball_velocity_tracking.params.update(
            {"ball_cfg": copy.deepcopy(ball_entity), "robot_cfg": copy.deepcopy(robot_entity)}
        )
        self.rewards.ball_height_penalty.params["ball_cfg"] = copy.deepcopy(ball_entity)

        # Restore body-velocity tracking if requested by subclasses.
        self.rewards.track_lin_vel_xy_exp.weight = self.track_body_velocity_weight


    def disable_zero_weight_rewards(self):
        """Remove reward terms whose weights are zero to save computation."""

        for attr in dir(self.rewards):
            if attr.startswith("__"):
                continue
            term = getattr(self.rewards, attr)
            if callable(term) or term is None:
                continue
            if getattr(term, "weight", None) == 0:
                setattr(self.rewards, attr, None)


__all__ = [
    "BallDribbleSceneCfg",
    "BallDribbleActionsCfg",
    "BallDribbleCommandsCfg",
    "BallDribbleObservationsCfg",
    "BallDribbleRewardsCfg",
    "BallDribbleEventCfg",
    "BallDribbleTerminationsCfg",
    "BallDribbleBaseEnvCfg",
]
