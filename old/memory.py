import numpy as np


class Memory:
    def __init__(self, env, test_mode, size=1000) -> None:
        self.observation_samples = self.run_episodes(size, env) if test_mode else None 

    @property
    def get(self):
        return self.observation_samples

    def run_episodes(self, episodes, env):
        observation_samples = []
        for episode in range(episodes):
            observation = env.reset()
            observation_samples.append(observation)
            done = False
            while not done:
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                observation_samples.append(observation)
        return np.array(observation_samples)
