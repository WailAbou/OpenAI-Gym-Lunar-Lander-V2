import numpy as np


class Memory:
    def __init__(self, env, train_mode, size=1000) -> None:
        self.state_samples = self.run_episodes(size, env) if train_mode else None

    @property
    def get(self):
        return self.state_samples

    def run_episodes(self, episodes, env):
        state_samples = []
        for episode in range(episodes):
            state = env.reset()
            state_samples.append(state)
            done = False
            while not done:
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                state_samples.append(state)
        return np.array(state_samples)
