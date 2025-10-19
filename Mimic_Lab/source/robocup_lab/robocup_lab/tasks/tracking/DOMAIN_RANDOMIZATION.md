# Tracking Task Domain Randomization

The RoboCup Lab tracking environments now mirror the perturbations used by HumanoidVerse3.

## Physics
- Rigid-body friction per environment sampled from `[0.2, 1.2]` (64 buckets).
- All tracked link masses and inertias uniformly scaled within `[0.9, 1.1]`.
- Torso centre of mass jittered by `±0.05 m` in `x/y` and `±0.01 m` in `z`.
- Joint home offsets perturbed `±0.01 rad` at startup.

## Actuation & Control
- PD gains scaled in `[0.9, 1.1]` each reset.
- Residual actuator offsets sampled uniformly in `±0.05` for every joint.
- Random force injection adds per-step uniform noise bounded by `0.05 × U[0.5, 1.5]`.
- Implicit actuators apply a control delay of 0–2 physics steps per environment.

## Disturbances & Resets
- Push disturbances every 5–10 seconds with planar impulses up to `0.1 m/s`.
- Root reset noise: position `±0.05 m`, attitude `±10°`, velocities `±0.01`.
- Joint reset noise: position `±0.1 rad`, velocity `±0.15 rad/s`.

These values are configured in `tracking_env_cfg.py` and the robot-specific configs so every tracking variant inherits the same domain randomization envelope as HumanoidVerse3.
