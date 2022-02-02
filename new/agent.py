from function_approximator import FunctionApproximator
from memory import Memory
import numpy as np
from epsilon_greedy_policy import EpsilonGreedyPolicy


class Agent:
    def __init__(self, learning_rate, memory_size, batch_size, discount, tau) -> None:
        self.policy = EpsilonGreedyPolicy()
        self.policy_network = FunctionApproximator(learning_rate)
        self.target_network = FunctionApproximator(learning_rate)
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

    def train(self):
        data, targets = [], []
        batch = self.memory.sample(self.batch_size)
        for transition in batch:
            state, next_state = transition.state, transition.next_state

            q_values = [0, 0, 0, 0] if transition.done else self.policy_network.q_values(next_state)
            best_action = np.argmax(q_values)
            new_value = transition.reward + self.discount * self.target_network.q_values(next_state)[best_action]

            target = self.policy_network.q_values(state)
            target[transition.action] = new_value

            data.append(state)
            targets.append(target)

        self.policy_network.train(data, targets)

    def copy_model(self):
        policy_weights = self.policy_network.get_weights()
        target_weights = self.target_network.get_weights()
        merged_weights = self.tau * policy_weights + (1 - self.tau) * target_weights
        self.target_network.set_weights(merged_weights)
