import os
import numpy as np
import gymnasium as gym

from agent import QLearningAgent
from utils import save_training_plot, save_gif, evaluate


# Experiment B configuration
ENV_ID = "FrozenLake-v1"
EPISODES = 50000
MAX_STEPS = 100
SEED = 42

# Save all Experiment B results separately
OUTPUT_DIR = "outputs/experiment_c"


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = gym.make(ENV_ID, is_slippery=True)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.2,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.999,
        seed=SEED
    )

    rewards = []

    print("Experiment B: epsilon_decay = 0.999")
    print("Training started...")

    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            agent.learn(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (episode + 1) % 5000 == 0:
            recent_rate = np.mean(rewards[-1000:])

            print(
                f"Episode {episode + 1:5}/{EPISODES} | "
                f"Recent success rate: {recent_rate:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    # Save the trained Q-table and training graph
    np.save(
        f"{OUTPUT_DIR}/q_table.npy",
        agent.q_table
    )

    save_training_plot(
        rewards,
        output_path=f"{OUTPUT_DIR}/training_rewards.png"
    )

    # Evaluate the trained agent
    eval_env = gym.make(ENV_ID, is_slippery=True)

    success_rate, avg_length = evaluate(
        eval_env,
        agent.q_table,
        episodes=100,
        seed=1000
    )

    eval_env.close()

    print("\nExperiment B Results")
    print("--------------------")
    print(f"Evaluation success rate: {success_rate:.1%}")
    print(f"Average episode length: {avg_length:.2f} steps")

    # Find a successful episode for the GIF
    render_env = gym.make(
        ENV_ID,
        is_slippery=True,
        render_mode="rgb_array"
    )

    successful_frames = None
    best_frames = None
    best_reward = -1

    for attempt in range(100):
        state, _ = render_env.reset(seed=2026 + attempt)

        frames = [render_env.render()]
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = int(np.argmax(agent.q_table[state]))

            state, reward, terminated, truncated, _ = (
                render_env.step(action)
            )

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

    frames_to_save = (
        successful_frames
        if successful_frames is not None
        else best_frames
    )

    save_gif(
        frames_to_save,
        output_path=f"{OUTPUT_DIR}/trained_agent.gif"
    )

    if successful_frames is not None:
        print("Successful episode GIF saved.")
    else:
        print("No successful GIF episode found in 100 attempts.")
        print("Saved the best attempt instead.")

    print(f"\nAll Experiment B files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()