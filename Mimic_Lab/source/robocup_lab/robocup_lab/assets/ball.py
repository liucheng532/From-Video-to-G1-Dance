# Copyright (c) 2024-2025 RoboCup Lab Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration snippets for generic ball-shaped rigid objects."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


#: Nominal soccer ball specification used across locomotion-with-ball experiments.
SOCCER_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.11,
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.35,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.3, 0.0, 0.16),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    collision_group=-1,
    debug_vis=False,
)


__all__ = ["SOCCER_BALL_CFG"]
