"""
Deep Q-Network (DQN) on CarRacing-v3.

Learns to drive a car around a track using a convolutional neural network
to approximate the Q-function, with experience replay and a target network.

KEY DIFFERENCES FROM LUNARLANDER:
- Observation is an image (96x96x3 RGB) instead of a vector of 8 numbers.
  We preprocess it: grayscale, crop, resize, and stack 4 frames for motion info.
- Action space is continuous (steering, gas, brake) but we DISCRETIZE it into
  simple actions so DQN can still work (DQN only handles discrete actions).
- Uses a CNN (convolutional neural network) instead of a feedforward net,
  because the input is now an image — CNNs are designed to extract spatial features.

WHY STACK FRAMES?
A single frame doesn't tell you velocity or direction of movement.
By stacking 4 consecutive frames, the network can infer motion — like
how a flipbook shows movement through sequential images.

TUNING NOTES (what makes this work vs. a naive DQN):
1. Buffer warmup: collect 5000 random experiences before learning starts
2. Learn every 4 steps, not every step — reduces correlation between updates
3. Soft target updates (slowly blend target network) instead of hard copy every N episodes
4. Action repeat: hold each action for 4 frames — speeds up training and gives
   more meaningful state transitions (single frames change too little)
5. Huber loss instead of MSE — less sensitive to outlier rewards
"""

import os
import json
import random
from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

CHECKPOINTS_DIR = "results/checkpoints_car_racing"

# Discretized action space: (steering, gas, brake)
# CarRacing's continuous actions: steering [-1, 1], gas [0, 1], brake [0, 1]
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),    # 0: do nothing
    np.array([-0.6, 0.0, 0.0]),   # 1: turn left
    np.array([0.6, 0.0, 0.0]),    # 2: turn right
    np.array([0.0, 0.8, 0.0]),    # 3: gas
    np.array([0.0, 0.0, 0.6]),    # 4: brake
    np.array([-0.4, 0.5, 0.0]),   # 5: turn left + gas
    np.array([0.4, 0.5, 0.0]),    # 6: turn right + gas
    np.array([0.0, 0.3, 0.0]),    # 7: gentle gas (useful for curves)
    np.array([-0.3, 0.3, 0.0]),   # 8: gentle left + gentle gas
    np.array([0.3, 0.3, 0.0]),    # 9: gentle right + gentle gas
    np.array([-0.6, 0.0, 0.3]),   # 10: turn left + brake
    np.array([0.6, 0.0, 0.3]),    # 11: turn right + brake
]


def preprocess_frame(frame):
    """
    Convert 96x96x3 RGB image to 84x84 grayscale.

    WHY?
    - Grayscale: color isn't important for driving, reduces data 3x
    - Crop bottom: removes the score bar which is irrelevant
    - Normalize to [0, 1]: helps neural network training converge faster
    """
    frame = frame[:84, 6:90]
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
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
        return np.array(self.frames)


class QNetworkCNN(nn.Module):
    """
    Convolutional neural network that approximates Q(state, action).

    Input: 4 stacked grayscale frames (4, 84, 84)
    Output: Q-value for each discrete action

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
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
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
        lr=5e-5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        batch_size=128,
        tau=0.005,           # soft target update rate (blend target net slowly)
        learn_every=4,       # only do a gradient step every N env steps
        buffer_warmup=5000,  # collect this many random experiences before learning
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
        self.tau = tau
        self.learn_every = learn_every
        self.buffer_warmup = buffer_warmup
        self.steps_done = 0

        self.policy_net = QNetworkCNN(n_frames, action_size).to(self.device)
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
        self.steps_done += 1

        if len(self.buffer) < self.buffer_warmup:
            return None
        if self.steps_done % self.learn_every != 0:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Huber loss: like MSE but less sensitive to outlier rewards
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Soft target update: slowly blend target net toward policy net
        # This is smoother than copying weights every N episodes
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def _save_training_state(rewards_history, best_avg_reward):
    """Save rewards history and best reward so training can be resumed without loss."""
    state = {
        "rewards_history": rewards_history,
        "best_avg_reward": best_avg_reward,
        "total_episodes": len(rewards_history),
    }
    path = os.path.join(CHECKPOINTS_DIR, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f)


def _load_training_state():
    """Load previous training state if it exists."""
    path = os.path.join(CHECKPOINTS_DIR, "training_state.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def train(episodes=1500, checkpoint_every=200, resume_from=None):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    env = gym.make("CarRacing-v3")
    agent = DQNAgent()
    frame_stack = FrameStack(n_frames=4)

    start_episode = 0
    rewards_history = []
    best_avg_reward = -float("inf")
    action_repeat = 4  # hold each action for this many frames

    # Resume from checkpoint: load weights, history, and best reward
    if resume_from is not None:
        agent.policy_net.load_state_dict(torch.load(resume_from, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.epsilon = 0.1  # mostly exploit, but still explore a bit to improve

        saved_state = _load_training_state()
        if saved_state is not None:
            rewards_history = saved_state["rewards_history"]
            best_avg_reward = saved_state["best_avg_reward"]
            start_episode = saved_state["total_episodes"]

        print(f"Resumed from: {resume_from}")
        print(f"Continuing from episode: {start_episode}")
        print(f"Epsilon set to: {agent.epsilon}")
        print(f"Best avg reward so far: {best_avg_reward:.2f}")

    for episode in range(start_episode, start_episode + episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        total_reward = 0
        done = False
        negative_reward_count = 0

        while not done:
            action_idx = agent.select_action(state)
            action = DISCRETE_ACTIONS[action_idx]

            # Action repeat: take the same action for multiple frames
            # Single frames change too little — this gives more meaningful transitions
            repeat_reward = 0
            for _ in range(action_repeat):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                repeat_reward += reward
                done = terminated or truncated
                if done:
                    break

            if repeat_reward < 0:
                negative_reward_count += 1
            else:
                negative_reward_count = 0
            if negative_reward_count > 50:
                done = True

            next_state = frame_stack.step(next_obs)

            agent.buffer.push(state, action_idx, repeat_reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += repeat_reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (episode + 1) % checkpoint_every == 0:
            path = os.path.join(CHECKPOINTS_DIR, f"episode_{episode + 1}.pth")
            torch.save(agent.policy_net.state_dict(), path)
            print(f"  [Checkpoint saved: {path}]")

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.buffer):6d}"
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINTS_DIR, "best.pth"))

    torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINTS_DIR, "final.pth"))
    _save_training_state(rewards_history, best_avg_reward)

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
    Loops until the agent fails or you hit Ctrl+C.

    Args:
        checkpoint_path: path to a .pth file (defaults to best.pth)
        episodes: ignored, loops until failure or Ctrl+C
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

    agent = DQNAgent()
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("Playing until failure. Press Ctrl+C to stop early.\n")

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
    filename = f"results/dqn_car_racing_rewards_{len(rewards)}ep.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN on CarRacing-v3")
    parser.add_argument("--mode", choices=["train", "demo"], default="train",
                        help="train: train from scratch | demo: watch a trained model play")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth file (for demo or resume)")
    parser.add_argument("--episodes", type=int, default=1500,
                        help="Number of episodes (training or demo)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --checkpoint (default: best.pth)")
    args = parser.parse_args()

    if args.mode == "demo":
        demo(checkpoint_path=args.checkpoint, episodes=args.episodes)
    else:
        print("Training DQN agent on CarRacing-v3...")
        print(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')}")
        print("-" * 50)

        resume_path = None
        if args.resume:
            resume_path = args.checkpoint or os.path.join(CHECKPOINTS_DIR, "best.pth")

        agent, rewards = train(episodes=args.episodes, resume_from=resume_path)
        plot_rewards(rewards)

        print("\nEvaluating trained agent...")
        agent.epsilon = 0.0
        evaluate(agent, episodes=10, render=False)
