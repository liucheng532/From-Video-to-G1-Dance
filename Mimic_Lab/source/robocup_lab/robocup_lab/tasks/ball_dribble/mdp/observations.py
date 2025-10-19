from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ball_pos_in_base(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the ball position expressed in the robot base frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    delta = ball.data.root_pos_w - robot.data.root_link_pos_w
    pos_b = math_utils.quat_apply(math_utils.quat_conjugate(robot.data.root_link_quat_w), delta)
    return pos_b


def ball_vel_in_base(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the relative linear velocity of the ball in the robot base frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    lin_vel_rel = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    vel_b = math_utils.quat_apply(math_utils.quat_conjugate(robot.data.root_link_quat_w), lin_vel_rel)
    return vel_b


def ball_ang_vel(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return the angular velocity of the ball in world coordinates."""

    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_ang_vel_w


def zero_vector_observation(env: ManagerBasedEnv, length: int) -> torch.Tensor:
    """Utility returning a zero observation vector of the requested length."""

    device = env.device if hasattr(env, "device") else getattr(env.scene, "device", "cpu")
    return torch.zeros(env.num_envs, length, device=device)
