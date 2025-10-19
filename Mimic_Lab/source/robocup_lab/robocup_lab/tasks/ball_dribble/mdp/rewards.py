from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_tracking_exp(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float],
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Reward keeping the ball close to a desired offset in the robot base frame."""

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    delta = ball.data.root_pos_w - robot.data.root_link_pos_w
    delta_b = math_utils.quat_apply(math_utils.quat_conjugate(robot.data.root_link_quat_w), delta)
    target = delta_b.new_tensor(target_offset)
    weight_tensor = delta_b.new_tensor(axis_weights)
    error = torch.sum(weight_tensor * torch.square(delta_b - target), dim=1)
    reward = torch.exp(-error / std**2)
    return reward


def ball_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 0.2),
) -> torch.Tensor:
    """Reward matching the ball velocity with the commanded base velocity."""

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_vel_b = math_utils.quat_apply(
        math_utils.quat_conjugate(robot.data.root_link_quat_w), ball.data.root_lin_vel_w
    )
    cmd = env.command_manager.get_command(command_name)
    weight_tensor = ball_vel_b.new_tensor(axis_weights)
    diff_xy = torch.sum(weight_tensor[:2] * torch.square(ball_vel_b[:, :2] - cmd[:, :2]), dim=1)
    diff_z = weight_tensor[2] * torch.square(ball_vel_b[:, 2])
    diff = diff_xy + diff_z
    reward = torch.exp(-diff / std**2)
    return reward


def ball_height_penalty(
    env: ManagerBasedRLEnv,
    target_height: float,
    tolerance: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize the ball height when it deviates significantly from the ground contact band."""

    ball: RigidObject = env.scene[ball_cfg.name]
    height_error = torch.abs(ball.data.root_pos_w[:, 2] - target_height)
    penalty = torch.clamp(height_error - tolerance, min=0.0)
    return penalty


def ball_directional_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_speed: float = 1.0e-3,
) -> torch.Tensor:
    """Reward proportional to ball speed aligned with commanded heading.

    r = ||v_ball|| * cos(theta) where theta is angle between ball velocity and commanded planar direction.
    """

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_vel_b = math_utils.quat_apply(
        math_utils.quat_conjugate(robot.data.root_link_quat_w), ball.data.root_lin_vel_w
    )
    ball_xy = ball_vel_b[:, :2]
    speed = torch.linalg.norm(ball_xy, dim=1)

    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)

    reward = torch.zeros_like(speed)
    valid_mask = (speed > min_speed) & (command_norm > min_speed)
    if torch.any(valid_mask):
        normalized_ball = ball_xy[valid_mask] / speed[valid_mask].unsqueeze(1)
        normalized_cmd = command_xy[valid_mask] / command_norm[valid_mask].unsqueeze(1)
        cos_theta = torch.sum(normalized_ball * normalized_cmd, dim=1)
        reward[valid_mask] = speed[valid_mask] * cos_theta

    return reward
