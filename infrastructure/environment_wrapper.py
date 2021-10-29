import gym

"""Wrapper Class for an Environment. Stores the same environment to be accessed globally."""


class EnvironmentWrapper:
    def __init__(self, env_name=None):
        self.env_name = env_name
        if self.env_name is not None:
            self.env = gym.make(env_name)
        else:
            self.env = None

    def get(self):
        return self.get_env()

    def get_env(self):
        return self.env

    def get_env_name(self):
        return self.env_name

    def set_env(self, env_name):
        self.env_name = env_name
        if self.env_name is not None:
            self.env = gym.make(env_name)
        else:
            self.env = None
