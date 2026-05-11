"""
Deep Q-Network (DQN) on LunarLander-v3.

Learns to land a spacecraft using a neural network to approximate
the Q-function, with experience replay and a target network.

WHY DQN INSTEAD OF Q-TABLE?
- Q-learning uses a table: one row per state, one column per action.
- LunarLander has continuous states (position, velocity, angle, etc.)
  so there are infinite possible states — a table won't work.
- DQN replaces the table with a neural network that takes a state as input
  and outputs Q-values for all actions. The network generalizes across
  similar states it hasn't seen before.

KEY DQN TRICKS (these make training stable):
1. Replay Buffer: store experiences, train on random batches (breaks correlation)
2. Target Network: a frozen copy of the network used to compute targets (prevents moving target problem)
"""

import os
import random
from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

CHECKPOINTS_DIR = "results/checkpoints_lunar_lander"


class QNetwork(nn.Module):
    """
    Neural network that approximates Q(state, action).

    Input: state vector (8 values for LunarLander: x, y, velocity_x, velocity_y, angle, angular_vel, leg_left, leg_right)
    Output: Q-value for each action (4 actions: do nothing, fire left, fire main, fire right)

    Architecture: simple feedforward net with 2 hidden layers.
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),   # 8 inputs -> 64 neurons
            nn.ReLU(),                            # activation function (introduces non-linearity)
            nn.Linear(hidden_size, hidden_size),  # 64 -> 64 neurons
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),  # 64 -> 4 outputs (one Q-value per action)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Stores past experiences so we can learn from them later in random order.

    WHY? If we only learn from consecutive experiences, the network sees
    correlated data (similar states in a row) which makes training unstable.
    Random sampling breaks this correlation — like shuffling a dataset.
    """
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Each experience is a tuple: (where I was, what I did, what I got, where I ended up, is it over?)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Grab a random batch for training
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=5e-4,             # learning rate for the neural network optimizer
        gamma=0.99,          # discount factor: how much we value future rewards
        epsilon_start=1.0,   # start fully random (100% exploration)
        epsilon_end=0.01,    # end with 1% randomness (mostly exploitation)
        epsilon_decay=0.995, # multiply epsilon by this each episode
        batch_size=64,       # how many experiences to learn from at once
        target_update_freq=10,  # update target network every N episodes
    ):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # POLICY NETWORK: the network we actively train — makes decisions
        self.policy_net = QNetwork(state_size, action_size).to(self.device)

        # TARGET NETWORK: a frozen copy used to calculate "what should Q be?"
        # We update it periodically. This prevents the "chasing a moving target" problem —
        # if we used the same network for both predicting and targeting, it's like
        # trying to measure a ruler that keeps changing length.
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.steps_done = 0

    def select_action(self, state):
        """Epsilon-greedy: random action with probability epsilon, else best known action."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # No gradient needed — we're just picking an action, not training
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def learn(self):
        """Sample a batch from replay buffer and do one gradient update."""
        if len(self.buffer) < self.batch_size:
            return None  # not enough experiences yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert numpy arrays to tensors for PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # CURRENT Q-VALUES: what our policy network thinks Q(s,a) is
        # .gather picks the Q-value for the action we actually took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TARGET Q-VALUES: what Q(s,a) should be according to Bellman equation
        # target = reward + gamma * max(Q_target(next_state)) if not done
        # We use the TARGET network here (not policy net) for stability
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)  # (1-dones) zeros out future reward if episode ended

        # LOSS: how far off are our predictions? Train to minimize this gap.
        loss = nn.MSELoss()(current_q, target_q)

        # BACKPROPAGATION: adjust network weights to reduce the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy policy network weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(episodes=1000, checkpoint_every=200):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]  # 8 continuous values
    action_size = env.action_space.n             # 4 discrete actions

    agent = DQNAgent(state_size, action_size)
    rewards_history = []
    best_avg_reward = -float("inf")

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Play one full episode
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience and learn from a random batch
            agent.buffer.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        # After each episode: decay exploration and periodically sync target net
        agent.decay_epsilon()

        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target()

        rewards_history.append(total_reward)

        # Save periodic checkpoints to visualize learning progression
        if (episode + 1) % checkpoint_every == 0:
            path = os.path.join(CHECKPOINTS_DIR, f"episode_{episode + 1}.pth")
            torch.save(agent.policy_net.state_dict(), path)
            print(f"  [Checkpoint saved: {path}]")

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINTS_DIR, "best.pth"))

    torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINTS_DIR, "final.pth"))

    env.close()
    return agent, rewards_history


def evaluate(agent, episodes=10, render=True):
    """Watch the trained agent play (no learning, no exploration)."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()
    print(f"\nMean reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards


def demo(checkpoint_path=None, episodes=5):
    """
    Watch the trained agent play with rendering.
    Loops episodes until you hit Ctrl+C or it finishes the requested number.

    Args:
        checkpoint_path: path to a .pth file (defaults to best.pth)
        episodes: how many episodes to play (set high and Ctrl+C to stop whenever)
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best.pth")

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}")
        print("Train the model first, or pass a valid checkpoint path.")
        return

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    # Load the trained model
    agent = DQNAgent(state_size, action_size)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Playing {episodes} episodes. Press Ctrl+C to stop early.\n")

    env = gym.make("LunarLander-v3", render_mode="human")

    try:
        ep = 0
        while True:
            ep += 1
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            # LunarLander: reward > 200 is considered solved, negative means crash
            print(f"  Episode {ep}: reward = {total_reward:.2f} | steps = {steps}")
            if total_reward < 0:
                print("\nAgent crashed. Stopping.")
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    env.close()
    print("Done.")


def plot_rewards(rewards, window=50):
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"Smoothed (window={window})")
    plt.axhline(y=200, color="r", linestyle="--", alpha=0.5, label="Solved threshold (200)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on LunarLander-v3 — Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/dqn_lunar_lander_rewards.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN on LunarLander-v3")
    parser.add_argument("--mode", choices=["train", "demo"], default="train",
                        help="train: train from scratch | demo: watch a trained model play")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth file (for demo mode)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes (training or demo)")
    args = parser.parse_args()

    if args.mode == "demo":
        demo(checkpoint_path=args.checkpoint, episodes=args.episodes)
    else:
        print("Training DQN agent on LunarLander-v3...")
        print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')}")
        print("-" * 50)

        agent, rewards = train(episodes=args.episodes)
        plot_rewards(rewards)

        print("\nEvaluating trained agent...")
        agent.epsilon = 0.0
        evaluate(agent, episodes=10, render=False)