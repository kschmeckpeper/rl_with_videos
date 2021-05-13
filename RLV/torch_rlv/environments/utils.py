from .acrobot_env import get_acrobot
import gym


def get_environment(name):
    if name == "acrobot":
        return get_acrobot()
    else:
        print('acrobot not available, created InvertedPendulumBulletEnv-v0')
        return gym.make('InvertedPendulumBulletEnv-v0')

