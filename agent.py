import numpy as np

class QLearningAgent:
    """Tabular Q-learning agent implemented from scratch."""
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995, seed=42):
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, terminated, truncated):
        done = terminated or truncated
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
