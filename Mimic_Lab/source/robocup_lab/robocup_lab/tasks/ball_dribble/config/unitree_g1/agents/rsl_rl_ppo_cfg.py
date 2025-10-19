from isaaclab.utils import configclass

from robocup_lab.tasks.locomotion.velocity.config.unitree_g1.agents.rsl_rl_ppo_cfg import (
    UnitreeG1FlatPPORunnerCfg,
)


@configclass
class UnitreeG1BallDribblePPORunnerCfg(UnitreeG1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "unitree_g1_ball_dribble"
        self.save_interval = 50

