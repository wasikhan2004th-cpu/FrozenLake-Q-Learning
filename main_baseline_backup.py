import os
import numpy as np
import gymnasium as gym
from agent import QLearningAgent
from utils import save_training_plot, save_gif, evaluate

ENV_ID = "FrozenLake-v1"
EPISODES = 50000
MAX_STEPS = 100
SEED = 42

def train():
    os.makedirs("outputs", exist_ok=True)
    env = gym.make(ENV_ID, is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states, n_actions, learning_rate=0.1, gamma=0.99,
                           epsilon=1.0, epsilon_min=0.02,
                           epsilon_decay=0.9995, seed=SEED)
    rewards = []
    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state, terminated, truncated)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rewards.append(total_reward)
        if (episode + 1) % 5000 == 0:
            recent = np.mean(rewards[-1000:])
            print(f"Episode {episode + 1:>5}/{EPISODES} | recent 1000-episode success rate: {recent:.3f} | epsilon: {agent.epsilon:.3f}")
    env.close()

    np.save("outputs/q_table.npy", agent.q_table)
    save_training_plot(rewards)
    eval_env = gym.make(ENV_ID, is_slippery=True)
    success_rate, avg_length = evaluate(eval_env, agent.q_table, episodes=100)
    eval_env.close()
    print(f"Evaluation success rate over 100 episodes: {success_rate:.1%}")
    print(f"Average episode length: {avg_length:.2f} steps")

    # Search for a successful greedy episode to use in the assignment GIF.
    render_env = gym.make(ENV_ID, is_slippery=True, render_mode="rgb_array")
    successful_frames = None
    best_frames = None
    best_reward = -1.0
    for attempt in range(100):
        state, _ = render_env.reset(seed=2026 + attempt)
        frames = [render_env.render()]
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(agent.q_table[state]))
            state, reward, terminated, truncated, _ = render_env.step(action)
            total_reward += reward
            frames.append(render_env.render())
            if terminated or truncated:
                break
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
        if total_reward > 0:
            successful_frames = frames
            break
    render_env.close()
    if successful_frames is not None:
        save_gif(successful_frames)
        print("GIF saved from a successful episode.")
    else:
        save_gif(best_frames)
        print("No successful episode found in 100 GIF attempts; saved the best attempt. Consider training longer or tuning hyperparameters.")
    print("Saved: outputs/q_table.npy, outputs/training_rewards.png, outputs/trained_agent.gif")

if __name__ == "__main__":
    train()
