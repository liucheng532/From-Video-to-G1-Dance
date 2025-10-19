from isaaclab.utils import configclass

from robocup_lab.tasks.locomotion.velocity.config.booster_t1.agents.rsl_rl_ppo_cfg import (
    BoosterT1FlatPPORunnerCfg,
)


@configclass
class BoosterT1BallDribblePPORunnerCfg(BoosterT1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "booster_t1_ball_dribble"
        self.save_interval = 50
