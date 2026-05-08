"""
Q-Learning on FrozenLake (deterministic).

A simple tabular RL example: the agent learns a Q-table mapping
(state, action) pairs to expected rewards, then follows the greedy policy.

KEY RL CONCEPTS:
- Agent: the learner/decision-maker (our algorithm)
- Environment: the world the agent interacts with (FrozenLake grid)
- State: where the agent currently is (grid position 0-15)
- Action: what the agent can do (left, down, right, up)
- Reward: feedback signal (+1 for reaching goal, 0 otherwise)
- Policy: the strategy — which action to take in each state
- Q-value: expected total future reward for taking action A in state S
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def create_environment():
    # FrozenLake is a 4x4 grid:
    #   S = Start, F = Frozen (safe), H = Hole (game over), G = Goal
    #   The agent must navigate from S to G without falling in holes.
    #   is_slippery=False makes it deterministic (actions always go where intended)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    return env


def train(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Train the agent using Q-Learning.

    Parameters:
    - alpha (learning rate): how much new info overrides old Q-values (0.1 = update 10% toward new estimate)
    - gamma (discount factor): how much the agent cares about future vs immediate rewards (0.99 = very forward-looking)
    - epsilon: probability of taking a random action (exploration vs exploitation)
    """
    n_states = env.observation_space.n   # 16 states (4x4 grid)
    n_actions = env.action_space.n       # 4 actions (left, down, right, up)

    # Q-table: a 16x4 matrix, initially all zeros.
    # Each cell Q[s][a] represents "how good is it to take action a in state s?"
    q_table = np.zeros((n_states, n_actions))

    epsilon = epsilon_start
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # EPSILON-GREEDY: the core exploration/exploitation tradeoff.
            # With probability epsilon: take a random action (EXPLORE — discover new paths)
            # With probability 1-epsilon: take the best known action (EXPLOIT — use what we learned)
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # random action
            else:
                action = np.argmax(q_table[state])  # best action according to Q-table

            # Take the action, observe what happens
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # THE Q-LEARNING UPDATE (Bellman equation):
            #   Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]
            #
            # In plain English:
            #   new_estimate = reward_we_got + discounted_best_future_value
            #   error = new_estimate - old_estimate
            #   Q(s,a) = Q(s,a) + learning_rate * error
            #
            # Over many episodes, Q-values converge to the true expected rewards.
            best_next = np.max(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])

            state = next_state
            total_reward += reward

        # Decay epsilon: explore less over time as we learn more
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def evaluate(env, q_table, episodes=100):
    """Test the learned policy (no exploration, pure exploitation)."""
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Always pick the best action — no randomness
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward == 1.0:
            wins += 1
    return wins / episodes


def print_policy(q_table):
    """Visualize the learned policy as arrows on the grid."""
    action_symbols = ["←", "↓", "→", "↑"]
    print("\nLearned Policy (4x4 grid):")
    print("-" * 20)
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            if state == 15:
                row_str += " G "
            elif state in [5, 7, 11, 12]:
                row_str += " H "
            else:
                best_action = np.argmax(q_table[state])
                row_str += f" {action_symbols[best_action]} "
        print(row_str)
    print("-" * 20)
    print("G = Goal, H = Hole")


def plot_rewards(rewards, window=100):
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (window={window})")
    plt.title("Q-Learning on FrozenLake — Training Progress")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("code/q_learning_rewards.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    env = create_environment()

    print("Training Q-Learning agent on FrozenLake (4x4, deterministic)...")
    q_table, rewards = train(env, episodes=2000)

    win_rate = evaluate(env, q_table)
    print(f"\nEvaluation win rate: {win_rate * 100:.1f}%")

    print_policy(q_table)
    plot_rewards(rewards)

    env.close()