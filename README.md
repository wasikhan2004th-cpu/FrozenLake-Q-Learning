# Custom Reinforcement Learning Agent — FrozenLake-v1

## Overview
This project implements tabular Q-learning from scratch with Farama Gymnasium's `FrozenLake-v1` environment. The map is slippery, so actions can lead to a different direction than intended.

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

The script trains the agent and saves:
- `outputs/q_table.npy` — learned Q-table
- `outputs/training_rewards.png` — episode reward and moving average
- `outputs/trained_agent.gif` — one greedy-policy episode
- Console evaluation success rate and average episode length

## Algorithm
For state `s`, action `a`, reward `r`, and next state `s'`:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

When an episode ends, the future-value term is zero. Epsilon-greedy action selection balances exploration and exploitation.

## Hyperparameters
| Parameter | Value |
|---|---:|
| Episodes | 50,000 |
| Max steps/episode | 100 |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.02 |
| Epsilon decay | 0.9995 |
| Environment | FrozenLake-v1, is_slippery=True |

## Important
Run the project in your own Python environment and use the actual printed results/plots in your report. Do not claim a success rate before running the experiment. Since the environment is stochastic, results can vary slightly by library version.
