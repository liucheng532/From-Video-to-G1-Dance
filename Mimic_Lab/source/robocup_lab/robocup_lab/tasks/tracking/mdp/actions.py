from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class RandomizedJointPositionAction(JointPositionAction):
    """Joint position action with per-episode residual offsets and step-wise torque noise."""

    cfg: "RandomizedJointPositionActionCfg"

    def __init__(self, cfg: "RandomizedJointPositionActionCfg", env):
        super().__init__(cfg, env)

        # buffers for residual actuator offsets (RAO) and random force injection (RFI) scaling
        self._rao_offsets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._rfi_scale = torch.ones(self.num_envs, self.action_dim, device=self.device)

        # cached limits for faster noise sampling
        self._rao_limit = float(cfg.rao_limit)
        self._rfi_limit = float(cfg.rfi_limit)
        self._use_rao = bool(cfg.use_rao)
        self._use_rfi = self._rfi_limit > 0.0

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        if self._use_rao:
            self._processed_actions += self._rao_offsets

        if self._use_rfi:
            noise = (torch.rand_like(self._processed_actions) * 2.0 - 1.0) * self._rfi_limit * self._rfi_scale
            self._processed_actions += noise

    # ---------------------------------------------------------------------
    # Randomization helpers
    # ---------------------------------------------------------------------
    def _resolve_env_ids(self, env_ids):
        if env_ids is None or env_ids == slice(None):
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def randomize_rao_offsets(self, env_ids, limit_range: tuple[float, float] | None = None):
        if not self._use_rao:
            return
        lower, upper = limit_range if limit_range is not None else (-self._rao_limit, self._rao_limit)
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._rao_offsets[env_ids] = torch.empty(
            (env_ids.numel(), self.action_dim), device=self.device
        ).uniform_(lower, upper)

    def set_rao_offsets(self, env_ids, offsets: torch.Tensor):
        if not self._use_rao:
            return
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._rao_offsets[env_ids] = offsets.to(self.device)

    def randomize_rfi_scale(self, env_ids, scale_range: tuple[float, float] | None = None):
        if not self._use_rfi:
            return
        lower, upper = scale_range if scale_range is not None else self.cfg.rfi_scale_range
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._rfi_scale[env_ids] = torch.empty(
            (env_ids.numel(), self.action_dim), device=self.device
        ).uniform_(lower, upper)


@configclass
class RandomizedJointPositionActionCfg(actions_cfg.JointPositionActionCfg):
    """Configuration for :class:`RandomizedJointPositionAction`."""

    class_type: type = RandomizedJointPositionAction

    use_rao: bool = True
    rao_limit: float = 0.05

    rfi_limit: float = 0.05
    rfi_scale_range: tuple[float, float] = (0.5, 1.5)
