import numpy as np
import random


class EpsilonGreedyPolicy:
    epsilon: float = 1

    def select_action(self, state, action_space, model):
        if random.random() > self.epsilon:
            q_values = model.q_values(state)[0]
            best_action = np.argmax(q_values)
            return best_action
        return action_space.sample()
    
    def decay(self, episode):
        self.epsilon = 1.0 / np.sqrt(episode + 1)
