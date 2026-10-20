import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def moving_average(values, window=500):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")

def save_training_plot(rewards, output_path="outputs/training_rewards.png", window=500):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    ma = moving_average(rewards, window)
    if len(ma):
        offset = window - 1 if len(rewards) >= window else 0
        plt.plot(np.arange(offset, offset + len(ma)), ma, linewidth=2, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("FrozenLake Q-Learning Training Progress")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def save_gif(frames, output_path="outputs/trained_agent.gif", fps=2):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1 / fps, loop=0)

def evaluate(env, q_table, episodes=100, seed=1000):
    successes = 0
    lengths = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        for step in range(100):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        successes += int(total_reward > 0)
        lengths.append(step + 1)
    return successes / episodes, float(np.mean(lengths))
