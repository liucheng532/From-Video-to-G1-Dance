import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from robocup_lab.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _extract_policy_normalizer(policy):
    """Return the observation normalizer attached to the policy, if any."""
    if policy is None:
        return None
    if hasattr(policy, "actor_obs_normalizer"):
        return policy.actor_obs_normalizer
    if hasattr(policy, "student_obs_normalizer"):
        return policy.student_obs_normalizer
    return None


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self._last_checkpoint_path: str | None = None

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        self._last_checkpoint_path = path
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            normalizer = _extract_policy_normalizer(getattr(self.alg, "policy", None))
            export_policy_as_onnx(self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self._last_checkpoint_path: str | None = None

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        self._last_checkpoint_path = path
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            normalizer = _extract_policy_normalizer(getattr(self.alg, "policy", None))
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
