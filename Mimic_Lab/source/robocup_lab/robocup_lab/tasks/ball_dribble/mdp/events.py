from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_ball_near_base(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    offset_range: dict[str, tuple[float, float]],
    lin_vel_range: dict[str, tuple[float, float]] | None = None,
    ang_vel_range: dict[str, tuple[float, float]] | None = None,
):
    """Reset a rigid ball near the robot base with optional random offsets."""

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    offsets = torch.zeros((len(env_ids), 3), device=env.device, dtype=robot.data.root_link_pos_w.dtype)
    for axis, idx in zip(["x", "y", "z"], range(3)):
        low, high = offset_range.get(axis, (0.0, 0.0))
        offsets[:, idx] = torch.empty(len(env_ids), device=env.device).uniform_(low, high)

    yaw_only = yaw_quat(robot.data.root_link_quat_w[env_ids])
    offsets_w = math_utils.quat_apply(yaw_only, offsets)

    root_states = ball.data.root_state_w.clone()
    root_states[env_ids, :3] = robot.data.root_link_pos_w[env_ids] + offsets_w
    root_states[env_ids, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device, dtype=root_states.dtype
    )

    if lin_vel_range is None:
        root_states[env_ids, 7:10] = 0.0
    else:
        sampled = torch.zeros((len(env_ids), 3), device=env.device, dtype=root_states.dtype)
        for axis, idx in zip(["x", "y", "z"], range(3)):
            low, high = lin_vel_range.get(axis, (0.0, 0.0))
            sampled[:, idx] = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
        root_states[env_ids, 7:10] = sampled

    if ang_vel_range is None:
        root_states[env_ids, 10:13] = 0.0
    else:
        sampled_ang = torch.zeros((len(env_ids), 3), device=env.device, dtype=root_states.dtype)
        for axis, idx in zip(["x", "y", "z"], range(3)):
            low, high = ang_vel_range.get(axis, (0.0, 0.0))
            sampled_ang[:, idx] = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
        root_states[env_ids, 10:13] = sampled_ang

    ball.write_root_state_to_sim(root_states[env_ids], env_ids)
