from isaaclab.utils import configclass

from robocup_lab.robots.booster_t1 import BOOSTER_T1_ACTION_SCALE, BOOSTER_T1_CFG
from robocup_lab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class BoosterT1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot and actions
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = BOOSTER_T1_ACTION_SCALE

        # Motion tracking command config
        # Anchor body name is resolved robustly in MotionCommand, but we set a reasonable default.
        self.commands.motion.anchor_body_name = "Waist"
        # Select all bodies by regex to avoid hand-maintaining names across robot variants.
        self.commands.motion.body_names = [".*"]

        anchor_body = self.commands.motion.anchor_body_name
        body_names_cfg = self.commands.motion.body_names

        if body_names_cfg == [".*"]:
            self.events.base_com.params["asset_cfg"].body_names = [anchor_body]
            self.events.randomize_rigid_body_mass_links.params["asset_cfg"].body_names = ".*"
            self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = ".*"
        else:
            self.events.base_com.params["asset_cfg"].body_names = [anchor_body]
            self.events.randomize_rigid_body_mass_links.params["asset_cfg"].body_names = body_names_cfg
            self.events.randomize_rigid_body_inertia.params["asset_cfg"].body_names = body_names_cfg
