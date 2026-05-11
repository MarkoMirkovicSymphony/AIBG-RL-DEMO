"""
Replay saved DQN checkpoints to visualize agent progression.

Loads models from results/checkpoints_lunar_lander/ and plays episodes with rendering
so you can see how the agent improves over training.
"""

import os
import glob

import numpy as np
import gymnasium as gym
import torch

from dqn_lunar_lander import QNetwork, CHECKPOINTS_DIR


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_agent(checkpoint_path, state_size=8, action_size=4):
    device = get_device()
    net = QNetwork(state_size, action_size).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    net.eval()
    return net, device


def play_episode(net, device, render=True):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = net(state_tensor).argmax(dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    return total_reward


def replay_all_checkpoints(episodes_per_checkpoint=3):
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "episode_*.pth")))

    if not checkpoint_files:
        print(f"No checkpoints found in {CHECKPOINTS_DIR}/")
        print("Run dqn_lunar_lander.py first to train and save checkpoints.")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")
    print("=" * 50)

    for path in checkpoint_files:
        name = os.path.basename(path).replace(".pth", "").replace("_", " ").title()
        print(f"\n--- {name} ---")

        net, device = load_agent(path)
        rewards = []

        for ep in range(episodes_per_checkpoint):
            reward = play_episode(net, device, render=True)
            rewards.append(reward)
            print(f"  Run {ep + 1}: reward = {reward:.2f}")

        print(f"  Average: {np.mean(rewards):.2f}")

    # Also play the best model
    best_path = os.path.join(CHECKPOINTS_DIR, "best.pth")
    if os.path.exists(best_path):
        print(f"\n--- Best Model ---")
        net, device = load_agent(best_path)
        rewards = []
        for ep in range(episodes_per_checkpoint):
            reward = play_episode(net, device, render=True)
            rewards.append(reward)
            print(f"  Run {ep + 1}: reward = {reward:.2f}")
        print(f"  Average: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    replay_all_checkpoints(episodes_per_checkpoint=3)