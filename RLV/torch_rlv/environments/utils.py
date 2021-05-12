import acrobot_env


def get_environment(name):
    if name == "acrobot":
        return acrobot_env.get_acrobot()
