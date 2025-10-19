"""Custom command generators for the ball-dribble task."""

from __future__ import annotations

from typing import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

import robocup_lab.tasks.locomotion.velocity.mdp as locomotion_mdp


class BallDribbleVelocityCommand(locomotion_mdp.UniformVelocityCommand):
    """Sample planar commands that keep the ball in front while allowing gentle heading changes."""

    cfg: BallDribbleVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._as_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        # Reset buffers for the selected environments.
        self.vel_command_b[env_ids_tensor] = 0.0
        self.is_heading_env[env_ids_tensor] = False
        self.is_standing_env[env_ids_tensor] = False
        self.heading_target[env_ids_tensor] = self.robot.data.heading_w[env_ids_tensor]

        # Sample which environments should remain stationary.
        if self.cfg.rel_standing_envs > 0.0:
            standing_mask = (
                torch.rand(env_ids_tensor.numel(), device=self.device) <= self.cfg.rel_standing_envs
            )
            self.is_standing_env[env_ids_tensor] = standing_mask
        else:
            standing_mask = torch.zeros(env_ids_tensor.numel(), dtype=torch.bool, device=self.device)

        moving_env_ids = env_ids_tensor[~standing_mask]
        if moving_env_ids.numel() == 0:
            return

        speeds = torch.empty(moving_env_ids.numel(), device=self.device)
        speeds.uniform_(*self.cfg.planar_speed_range)

        heading_offsets = torch.empty(moving_env_ids.numel(), device=self.device)
        heading_offsets.uniform_(*self.cfg.heading_offset_range)

        vel_x = speeds * torch.cos(heading_offsets)
        vel_y = speeds * torch.sin(heading_offsets)
        planar_speed = torch.linalg.norm(torch.stack((vel_x, vel_y), dim=1), dim=1)

        # Zero out tiny commands so they don't trigger spurious heading targets.
        valid_motion = planar_speed >= self.cfg.min_planar_speed
        if not torch.all(valid_motion):
            zero = torch.zeros_like(vel_x)
            vel_x = torch.where(valid_motion, vel_x, zero)
            vel_y = torch.where(valid_motion, vel_y, zero)
            planar_speed = torch.where(valid_motion, planar_speed, torch.zeros_like(planar_speed))
            heading_offsets = torch.where(valid_motion, heading_offsets, torch.zeros_like(heading_offsets))

        # Populate the command buffers.
        self.vel_command_b[moving_env_ids, 0] = torch.clamp(
            vel_x, self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1]
        )
        self.vel_command_b[moving_env_ids, 1] = torch.clamp(
            vel_y, self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1]
        )
        self.vel_command_b[moving_env_ids, 2] = 0.0

        # Determine desired heading so the robot turns toward the motion direction.
        current_heading = self.robot.data.heading_w[moving_env_ids]
        desired_heading = math_utils.wrap_to_pi(current_heading + heading_offsets)
        allow_turn = planar_speed >= self.cfg.min_turn_speed

        self.heading_target[moving_env_ids] = torch.where(allow_turn, desired_heading, current_heading)
        self.is_heading_env[moving_env_ids] = allow_turn

    def _as_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class BallDribbleVelocityCommandCfg(locomotion_mdp.UniformVelocityCommandCfg):
    """Configuration for ball-dribble specific velocity sampling."""

    class_type: type = BallDribbleVelocityCommand

    heading_command: bool = True
    rel_heading_envs: float = 1.0

    planar_speed_range: tuple[float, float] = (0.4, 1.4)
    heading_offset_range: tuple[float, float] = (-0.75, 0.75)
    min_planar_speed: float = 0.1
    min_turn_speed: float = 0.25

    ranges: locomotion_mdp.UniformVelocityCommandCfg.Ranges = (
        locomotion_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.2, 1.2),
            heading=(-3.1416, 3.1416),
        )
    )
