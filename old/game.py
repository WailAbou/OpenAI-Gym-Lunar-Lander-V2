from memory import Memory
from model import Model
from tqdm import tqdm
import numpy as np


class Game:
    def __init__(self, env, train_mode=True) -> None:
        self.env = env
        self.train_mode = train_mode
        self.total_rewards = None
        self.memory = Memory(self.env, self.train_mode, self.train_mode)
        self.model = Model(self.env, self.memory.get, self.train_mode)

    def start(self, episodes, gamma=0.99):
        self.total_rewards = np.empty(episodes)
        for episode in tqdm(range(episodes)):
            epsilon = 1.0 / np.sqrt(episode + 1)
            total_reward, steps = self.play_one(self.env, epsilon, gamma)
            self.total_rewards[episode] = total_reward
            if episode % 100 == 0 and episode != 0: print(f"episode: {episode}", f"steps: {steps}", f"total reward: {total_reward}", f"epsilon: {epsilon}")
        self.env.close()

    def play_one(self, env, epsilon, gamma):
        state = env.reset()
        done = False
        total_reward = 0
        iters = 0
        while not done:
            action = self.model.sample_action(state, epsilon)
            previous_state = state
            state, reward, done, info = env.step(action)
            total_reward += self.td_learning(state, previous_state, action, reward, gamma)
            iters += 1
        if not self.train_mode: print(f'Total reward: {total_reward}')
        return total_reward, iters

    def td_learning(self, state, previous_state, action, reward, gamma):
        next = self.model.predict(state)
        G = reward + gamma * np.max(next)
        self.model.update(previous_state, action, G)
        return reward

    def save(self):
        self.model.save()
        textfile = open("../saves_old/totalrewards.txt", 'w')
        [textfile.write(str(total_reward) + "\n") for total_reward in self.total_rewards]
        textfile.close()

    def load(self):
        self.model.load()
