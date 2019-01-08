import vizdoom

class DoomEnv:
    """
    Doom environment
    """

    def __init__(self, conf):
        pass

    def new_episode(self):
        pass

    def is_new_episode(self):
        # Vizdoom give this info
        pass

    def init_env(self):
        pass

    def close_env(self):
        pass

    def step(self, action):
        pass

    def get_random_action(self):
        pass


class MultiDoomEnv:
    """
    Multi Doom Environments (Wrapper for multiple DoomEnv
    """

    def __init__(self, conf):
        pass

    def step(self, actions):
        pass

    def get_random_actions(self):
        pass