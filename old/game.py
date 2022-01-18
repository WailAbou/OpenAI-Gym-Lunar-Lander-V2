from memory import Memory
from model import Model
from tqdm import tqdm
import numpy as np

class Game:
    def __init__(self, env, test_mode=False) -> None:
        self.env = env
        self.test_mode = test_mode
        self.total_rewards = None
        self.memory = Memory(self.env, self.test_mode)
        self.model = Model(self.env, self.memory.get)

    def start(self, episodes, gamma=0.99):
        self.total_rewards = np.empty(episodes)
        for episode in tqdm(range(episodes)):
            epsilon = 1.0 / np.sqrt(episode + 1)
            total_reward, steps = self.play_one(self.env, epsilon, gamma)
            self.total_rewards[episode] = total_reward
            if episode % 100 == 0 and episode != 0: print(f"episode: {episode}", f"steps: {steps}", f"total reward: {total_reward}", f"epsilon: {epsilon}")
        self.env.close()

    def play_one(self, env, epsilon, gamma):
        observation = env.reset()
        done = False
        total_reward = 0
        iters = 0
        while not done:
            if self.test_mode:
                self.env.render()
            action = self.model.sample_action(observation, epsilon)
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            total_reward += self.td_learning(observation, prev_observation, action, reward, gamma)
            iters += 1
        return total_reward, iters

    def td_learning(self, observation, prev_observation, action, reward, gamma):
        next = self.model.predict(observation)
        G = reward + gamma * np.max(next)
        if not self.test_mode: self.model.update(prev_observation, action, G)
        return reward

    def save(self):
        self.model.save()
        textfile = open("../saves/totalrewards.txt", 'w')
        [textfile.write(str(total_reward) + "\n") for total_reward in self.total_rewards]
        textfile.close()

    def load(self):
        self.model.load()
