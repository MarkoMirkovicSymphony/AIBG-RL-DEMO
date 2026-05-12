# Reinforcement Learning Demo

From tabular Q-learning to deep RL with convolutional neural networks. Three progressively harder environments, each introducing new concepts needed to solve them.

## Project Structure

```
AIBG-RL-DEMO/
├── code/
│   ├── q_learning.py              # Tabular Q-learning on FrozenLake
│   ├── dqn_lunar_lander.py        # DQN on LunarLander (vector observations)
│   ├── dqn_car_racing.py          # DQN + CNN on CarRacing (image observations)
│   └── replay_checkpoints.py      # Visualize agent improvement across checkpoints
├── results/
│   ├── checkpoints_q_learning/    # Q-table snapshots (.npy)
│   ├── checkpoints_lunar_lander/  # DQN weight files (.pth)
│   ├── checkpoints_car_racing/    # CNN-DQN weight files (.pth)
│   ├── q_learning_rewards.png
│   ├── dqn_lunar_lander_rewards.png
│   └── dqn_car_racing_rewards.png
└── pyproject.toml
```

## Setup

```bash
# Requires Python 3.11+
uv sync
```

## The Three Environments

### 1. FrozenLake — Tabular Q-Learning

**Problem**: Navigate a 4x4 grid from start to goal without falling into holes. The ice is slippery — actions don't always go where you intend.

**State space**: 16 discrete positions  
**Action space**: 4 directions (left, down, right, up)  
**Solution**: A 16x4 Q-table — small enough to store every state-action value directly.

**Core algorithm**:
```
Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
```

**Key hyperparameters**:

| Parameter               | Value  |
|-------------------------|--------|
| Learning rate (alpha)   | 0.8    |
| Discount factor (gamma) | 0.99   |
| Epsilon decay           | 0.9995 |
| Training episodes       | 20,000 |

**Usage**:
```bash
uv run code/q_learning.py --mode train --episodes 20000 --slippery
uv run code/q_learning.py --mode demo --checkpoint results/checkpoints_q_learning/final.npy
```

---

### 2. LunarLander — Deep Q-Network (DQN)

**Problem**: Land a spacecraft between the flags. State is now a vector of 8 continuous values (position, velocity, angle, leg contact) — infinite possible states means a Q-table won't work.

**State space**: 8-dimensional continuous vector  
**Action space**: 4 discrete actions (no-op, fire left, fire main, fire right)  
**Solution**: A neural network approximates Q(state, action) for any state.

**Network architecture**:
```
Input (8) → Linear(64) + ReLU → Linear(64) + ReLU → Linear(4) → Q-values
```

**Why DQN works where Q-tables don't**:
- Neural network generalizes across similar states
- Experience replay buffer (100k) breaks correlation between consecutive samples
- Target network (updated every 10 episodes) stabilizes learning targets

**Key hyperparameters**:

| Parameter               | Value             |
|-------------------------|-------------------|
| Learning rate           | 5e-4              |
| Discount factor (gamma) | 0.99              |
| Epsilon decay           | 0.995             |
| Batch size              | 64                |
| Replay buffer           | 100,000           |
| Target update frequency | Every 10 episodes |
| Training episodes       | 1,000             |
| Solved threshold        | 200               |

**Usage**:
```bash
uv run code/dqn_lunar_lander.py --mode train --episodes 1000
uv run code/dqn_lunar_lander.py --mode demo --checkpoint results/checkpoints_lunar_lander/best.pth
```

---

### 3. CarRacing — DQN with CNN

**Problem**: Drive a car around a randomly generated track. Input is now a 96x96 RGB image — raw pixels, no feature engineering. The action space is also continuous (steering, gas, brake).

**State space**: Images (96x96x3 RGB), preprocessed to 4 stacked grayscale frames (4, 84, 84)  
**Action space**: Continuous, discretized into 12 meaningful driving actions  
**Solution**: A convolutional neural network processes the images, and additional tricks stabilize training.

**Preprocessing pipeline**:
```
96x96 RGB → crop to 84x84 → grayscale → normalize [0,1] → stack 4 frames
```

Frame stacking gives the network temporal information — a single frame can't tell you which direction the car is moving.

**Discretized actions** (12 total):

| Index | Action             | Values (steering, gas, brake) |
|-------|--------------------|-------------------------------|
| 0     | Do nothing         | [0, 0, 0]                     |
| 1     | Turn left          | [-0.6, 0, 0]                  |
| 2     | Turn right         | [0.6, 0, 0]                   |
| 3     | Gas                | [0, 0.8, 0]                   |
| 4     | Brake              | [0, 0, 0.6]                   |
| 5     | Left + gas         | [-0.4, 0.5, 0]                |
| 6     | Right + gas        | [0.4, 0.5, 0]                 |
| 7     | Gentle gas         | [0, 0.3, 0]                   |
| 8     | Gentle left + gas  | [-0.3, 0.3, 0]                |
| 9     | Gentle right + gas | [0.3, 0.3, 0]                 |
| 10    | Left + brake       | [-0.6, 0, 0.3]                |
| 11    | Right + brake      | [0.6, 0, 0.3]                 |

**CNN architecture** (based on DeepMind Atari DQN):
```
Input (4, 84, 84)
  → Conv2d(4→32, 8x8, stride=4) → ReLU    → (32, 20, 20)
  → Conv2d(32→64, 4x4, stride=2) → ReLU   → (64, 9, 9)
  → Conv2d(64→64, 3x3, stride=1) → ReLU   → (64, 7, 7)
  → Flatten                                 → 3136
  → Linear(3136→512) → ReLU
  → Linear(512→12)                          → Q-values
```

**Training stabilization tricks**:

| Trick                            | Why                                              |
|----------------------------------|--------------------------------------------------|
| Buffer warmup (5000 steps)       | Diverse experiences before first gradient update |
| Learn every 4 steps              | Reduces correlation between updates              |
| Soft target updates (tau=0.005)  | Smoother than hard copy every N episodes         |
| Action repeat (4 frames)         | Makes state transitions more meaningful          |
| Huber loss                       | Less sensitive to outlier rewards than MSE       |
| Gradient clipping (norm=10)      | Prevents exploding gradients                     |
| Early stop on 50+ negative steps | Avoids wasting time on failed trajectories       |

**Key hyperparameters**:

| Parameter               | Value       |
|-------------------------|-------------|
| Learning rate           | 5e-5        |
| Discount factor (gamma) | 0.99        |
| Epsilon decay           | 0.999       |
| Batch size              | 128         |
| Replay buffer           | 100,000     |
| Soft update tau         | 0.005       |
| Buffer warmup           | 5,000 steps |
| Training episodes       | 1,500       |
| Solved threshold        | 900         |

**Usage**:
```bash
uv run code/dqn_car_racing.py --mode train --episodes 1500
uv run code/dqn_car_racing.py --mode train --resume --checkpoint results/checkpoints_car_racing/final.pth --episodes 3000
uv run code/dqn_car_racing.py --mode demo --checkpoint results/checkpoints_car_racing/best.pth
```

---

## Algorithm Progression

|                     | Q-Learning    | DQN (LunarLander)   | DQN + CNN (CarRacing) |
|---------------------|---------------|---------------------|-----------------------|
| **State**           | Discrete (16) | Continuous (8 dims) | Images (84x84x4)      |
| **Actions**         | Discrete (4)  | Discrete (4)        | Discretized (12)      |
| **Q-function**      | Table (16x4)  | Feedforward NN      | CNN                   |
| **Learning rate**   | 0.8           | 5e-4                | 5e-5                  |
| **Replay buffer**   | None          | 100k                | 100k + warmup         |
| **Target network**  | None          | Hard update / 10 ep | Soft update (tau)     |
| **Loss**            | N/A           | MSE                 | Huber                 |
| **Checkpoint size** | 640 B         | ~22 KB              | ~5.5 MB               |

Each step adds complexity because the environment demands it:
- Continuous states → need function approximation (neural network)
- Correlated data → need experience replay
- Moving targets → need target network
- Image inputs → need CNN
- No velocity info in pixels → need frame stacking
- Continuous actions → need discretization

## Replay Checkpoints

Visualize how the LunarLander agent improves over training by playing each saved checkpoint:

```bash
uv run code/replay_checkpoints.py
```

This loads checkpoints sequentially and renders 3 episodes per checkpoint, showing the progression from random behavior to skilled landing.

## Device Support

All DQN scripts auto-detect the best available device:
1. Apple Silicon (MPS)
2. NVIDIA GPU (CUDA)
3. CPU (fallback)