import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-Ball-Dribble-Booster-T1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_env_cfg:BoosterT1BallDribbleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterT1BallDribblePPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:BoosterT1BallDribbleTrainerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Ball-Dribble-BodyTrack-Booster-T1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_env_cfg:BoosterT1BallDribbleBodyTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoosterT1BallDribblePPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:BoosterT1BallDribbleTrainerCfg",
    },
)
