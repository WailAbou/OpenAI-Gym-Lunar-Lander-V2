from gym import make as gym_make
from agent import Agent
import matplotlib.pyplot as plt
from transition import Transition
from tqdm import trange


class Game:
    def __init__(self, agent: Agent, load: bool) -> None:
        self.env = gym_make('LunarLander-v2')
        self.total_rewards = []
        self.agent = agent
        if load:
            self.agent.policy_network.load()
            self.agent.target_network.load()

    def play(self, epsiodes: int, max_steps: int, train_freq: int, copy_freq: int) -> None:
        for episode in trange(epsiodes):
            total_reward = self.play_one(episode, max_steps, train_freq, copy_freq)
            self.total_rewards.append(total_reward)
            self.agent.policy.decay(episode)
            # self.display_stats()
        self.env.close()

    def play_one(self, episode: int, max_steps: int, train_freq: int, copy_freq: int) -> float:
        state = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            if (episode + 1) % train_freq == 0: self.agent.train()
            if (episode + 1) % copy_freq == 0: self.agent.copy_model()
            
            action = self.agent.policy.select_action(state, self.env.action_space, self.agent.policy_network)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            transition = Transition(state, action, reward, next_state, done)
            self.agent.memory.record(transition)
            
            state = next_state
            if done: break
        return total_reward

    def save(self):
        self.agent.policy_network.save()
        self.agent.target_network.save()

    def plot(self) -> None:
        plt.plot(self.total_rewards)
        plt.gca().set(xlabel="Episode number", ylabel="Total reward x time")
        plt.show()
