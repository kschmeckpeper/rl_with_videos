from gym.envs.registration import register


register(
        id='AcrobotContinuous-v1',
        entry_point='custom_envs.envs.acrobot_continuous:AcrobotContinuousEnv',
        max_episode_steps=500,
    )