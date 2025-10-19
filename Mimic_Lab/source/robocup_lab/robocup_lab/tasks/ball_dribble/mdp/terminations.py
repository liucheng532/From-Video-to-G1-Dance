from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_far_from_robot(
    env: "ManagerBasedRLEnv",
    threshold: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the distance between robot base and ball exceeds ``threshold``."""

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    distance = torch.linalg.norm(ball.data.root_pos_w - robot.data.root_link_pos_w, dim=1)
    return distance > threshold


def ball_outside_front_sector(
    env: "ManagerBasedRLEnv",
    aperture_deg: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_period_s: float = 0.2,
    min_speed: float = 1.0e-3,
) -> torch.Tensor:
    """Terminate when the ball velocity deviates from the forward cone longer than ``grace_period_s``."""

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    vel_b = math_utils.quat_apply(
        math_utils.quat_conjugate(robot.data.root_link_quat_w), ball.data.root_lin_vel_w
    )

    planar_vel = vel_b[:, :2]
    planar_speed = torch.linalg.norm(planar_vel, dim=1)
    cos_angle = torch.where(
        planar_speed > min_speed,
        planar_vel[:, 0] / planar_speed,
        torch.ones_like(planar_speed),
    )

    half_aperture_rad = math.radians(aperture_deg * 0.5)
    cos_limit = math.cos(half_aperture_rad)
    outside_mask = cos_angle < cos_limit

    timer = getattr(env, "_ball_front_sector_out_timer", None)
    if timer is None or timer.shape[0] != env.scene.num_envs:
        timer = torch.zeros(env.scene.num_envs, device=vel_b.device, dtype=vel_b.dtype)
        env._ball_front_sector_out_timer = timer

    if hasattr(env, "episode_length_buf"):
        fresh = env.episode_length_buf == 0
        if torch.any(fresh):
            timer[fresh] = 0.0

    mask_float = outside_mask.to(timer.dtype)
    timer.mul_(mask_float)

    dt = float(getattr(env, "step_dt", 0.0))
    if dt <= 0.0:
        timer.add_(mask_float)
    else:
        timer.add_(mask_float * dt)

    if hasattr(env, "reset_buf"):
        reset_ids = env.reset_buf > 0
        if torch.any(reset_ids):
            timer[reset_ids] = 0.0

    return timer > grace_period_s



def robot_fallen(
    env: "ManagerBasedRLEnv",
    min_height: float,
    min_up_dot: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot base slumps below the commanded height or tilts too much."""

    robot: Articulation = env.scene[robot_cfg.name]

    height_fail = robot.data.root_link_pos_w[:, 2] < min_height

    up_axis = robot.data.root_link_pos_w.new_zeros((robot.data.root_link_pos_w.shape[0], 3))
    up_axis[:, 2] = 1.0
    base_up_world = math_utils.quat_apply(robot.data.root_link_quat_w, up_axis)
    tilt_fail = base_up_world[:, 2] < min_up_dot

    return torch.logical_or(height_fail, tilt_fail)


__all__ = ["ball_far_from_robot", "ball_outside_front_sector", "robot_fallen"]
