"""
Deep Q-Network (DQN) on CarRacing-v3.

Learns to drive a car around a track using a convolutional neural network
to approximate the Q-function, with experience replay and a target network.

KEY DIFFERENCES FROM LUNARLANDER:
- Observation is an image (96x96x3 RGB) instead of a vector of 8 numbers.
  We preprocess it: grayscale, crop, resize, and stack 4 frames for motion info.
- Action space is continuous (steering, gas, brake) but we DISCRETIZE it into
  5 simple actions so DQN can still work (DQN only handles discrete actions).
- Uses a CNN (convolutional neural network) instead of a feedforward net,
  because the input is now an image — CNNs are designed to extract spatial features.

WHY STACK FRAMES?
A single frame doesn't tell you velocity or direction of movement.
By stacking 4 consecutive frames, the network can infer motion — like
how a flipbook shows movement through sequential images.
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

CHECKPOINTS_DIR = "code/checkpoints_car_racing"

# Discretized action space: (steering, gas, brake)
# CarRacing's continuous actions: steering [-1, 1], gas [0, 1], brake [0, 1]
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),    # 0: do nothing
    np.array([-1.0, 0.0, 0.0]),   # 1: turn left
    np.array([1.0, 0.0, 0.0]),    # 2: turn right
    np.array([0.0, 1.0, 0.0]),    # 3: gas
    np.array([0.0, 0.0, 0.8]),    # 4: brake
    np.array([-1.0, 1.0, 0.0]),   # 5: turn left + gas
    np.array([1.0, 1.0, 0.0]),    # 6: turn right + gas
    np.array([0.0, 0.3, 0.0]),    # 7: gentle gas (useful for curves)
]


def preprocess_frame(frame):
    """
    Convert 96x96x3 RGB image to 84x84 grayscale.

    WHY?
    - Grayscale: color isn't important for driving, reduces data 3x
    - Crop bottom: removes the score bar which is irrelevant
    - Normalize to [0, 1]: helps neural network training converge faster
    """
    # Crop the bottom 12 pixels (score bar)
    frame = frame[:84, 6:90]
    # Convert to grayscale: weighted sum matching human perception
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    # Normalize to [0, 1]
    gray = gray / 255.0
    return gray.astype(np.float32)


class FrameStack:
    """
    Stacks N consecutive frames together to give the network temporal information.

    Without stacking, the network sees a single snapshot — it can't tell
    if the car is moving forward, backward, or standing still.
    With 4 stacked frames, it can infer velocity and direction.
    """
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self._get_state()

    def _get_state(self):
        # Stack frames along a new axis: shape becomes (4, 84, 84)
        return np.array(self.frames)


class QNetworkCNN(nn.Module):
    """
    Convolutional neural network that approximates Q(state, action).

    Input: 4 stacked grayscale frames (4, 84, 84)
    Output: Q-value for each discrete action (8 actions)

    Architecture: 3 conv layers (extract spatial features) + 2 fully connected layers.
    This is similar to the original DeepMind Atari DQN architecture.
    """
    def __init__(self, n_frames, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),  # (4, 84, 84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),        # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),        # (64, 9, 9) -> (64, 7, 7)
            nn.ReLU(),
        )
        # Calculate flattened size after convolutions
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten: (batch, 64, 7, 7) -> (batch, 3136)
        return self.fc(x)


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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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
        n_frames=4,
        action_size=len(DISCRETE_ACTIONS),
        lr=1e-4,             # lower learning rate for CNN (more parameters to tune carefully)
        gamma=0.99,          # discount factor: how much we value future rewards
        epsilon_start=1.0,   # start fully random (100% exploration)
        epsilon_end=0.05,    # end with 5% randomness
        epsilon_decay=0.999, # slower decay — CarRacing needs more exploration
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
        self.policy_net = QNetworkCNN(n_frames, action_size).to(self.device)

        # TARGET NETWORK: a frozen copy used to calculate "what should Q be?"
        self.target_net = QNetworkCNN(n_frames, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        """Epsilon-greedy: random action with probability epsilon, else best known action."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def learn(self):
        """Sample a batch from replay buffer and do one gradient update."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # CURRENT Q-VALUES: what our policy network thinks Q(s,a) is
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TARGET Q-VALUES: reward + gamma * max(Q_target(next_state))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients with image inputs
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
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

    env = gym.make("CarRacing-v3", continuous=False if False else True)
    # We handle discretization ourselves since we want custom action combos
    agent = DQNAgent()
    frame_stack = FrameStack(n_frames=4)

    rewards_history = []
    best_avg_reward = -float("inf")

    for episode in range(episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0
        done = False
        negative_reward_count = 0

        while not done:
            action_idx = agent.select_action(state)
            action = DISCRETE_ACTIONS[action_idx]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Early stopping if the car is stuck (getting negative rewards for too long)
            if reward < 0:
                negative_reward_count += 1
            else:
                negative_reward_count = 0
            if negative_reward_count > 100:
                done = True

            next_state = frame_stack.step(next_obs)

            agent.buffer.push(state, action_idx, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target()

        rewards_history.append(total_reward)

        # Save periodic checkpoints
        if (episode + 1) % checkpoint_every == 0:
            path = os.path.join(CHECKPOINTS_DIR, f"episode_{episode + 1}.pth")
            torch.save(agent.policy_net.state_dict(), path)
            print(f"  [Checkpoint saved: {path}]")

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
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
    """Watch the trained agent drive (no learning, no exploration)."""
    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    frame_stack = FrameStack(n_frames=4)
    total_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            action = DISCRETE_ACTIONS[action_idx]
            obs, reward, terminated, truncated, _ = env.step(action)
            state = frame_stack.step(obs)
            done = terminated or truncated
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()
    print(f"\nMean reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    return total_rewards


def demo(checkpoint_path=None, episodes=5):
    """
    Watch the trained agent drive with rendering.
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

    # Load the trained model
    agent = DQNAgent()
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0.0  # no exploration — pure exploitation

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Playing {episodes} episodes. Press Ctrl+C to stop early.\n")

    env = gym.make("CarRacing-v3", render_mode="human")
    frame_stack = FrameStack(n_frames=4)

    try:
        ep = 0
        while True:
            ep += 1
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action_idx = agent.select_action(state)
                action = DISCRETE_ACTIONS[action_idx]
                obs, reward, terminated, truncated, _ = env.step(action)
                state = frame_stack.step(obs)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            # CarRacing: negative total reward means the car went off-track / failed
            print(f"  Episode {ep}: reward = {total_reward:.2f} | steps = {steps}")
            if total_reward < 0:
                print("\nAgent failed. Stopping.")
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    env.close()
    print("Done.")


def plot_rewards(rewards, window=20):
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"Smoothed (window={window})")
    plt.axhline(y=900, color="r", linestyle="--", alpha=0.5, label="Solved threshold (900)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CarRacing-v3 — Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("code/dqn_car_racing_rewards.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN on CarRacing-v3")
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
        print("Training DQN agent on CarRacing-v3...")
        print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')}")
        print("-" * 50)

        agent, rewards = train(episodes=args.episodes)
        plot_rewards(rewards)

        print("\nEvaluating trained agent...")
        agent.epsilon = 0.0
        evaluate(agent, episodes=10, render=False)