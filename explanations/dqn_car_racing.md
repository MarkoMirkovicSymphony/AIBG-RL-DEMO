# DQN on CarRacing

## Scaling Up — From Numbers to Pixels

- "LunarLander gave us 8 numbers as input. CarRacing gives us a 96x96 RGB image. The agent has to learn to drive by looking at the screen — just like a human would."
- "This introduces two new challenges: how do we process image input efficiently, and how do we handle a continuous action space with DQN (which only supports discrete actions)?"
- "This is where things start to resemble the original DeepMind Atari work that kicked off the deep RL revolution."

## Preprocessing — Making the Input Manageable

- "We do four things to the raw 96x96 RGB frame:"
  - "Grayscale — color isn't important for driving, and this reduces data 3x."
  - "Crop — remove the bottom score bar, it's irrelevant noise."
  - "Resize to 84x84 — standard size, slightly smaller for faster computation."
  - "Normalize to [0, 1] — helps the network train faster and more stably."
- "After preprocessing, one frame is an 84x84 grayscale image."

## Frame Stacking — Teaching the Network About Motion

- "A single frame is a snapshot. You can't tell from one photo whether the car is moving fast or slow, turning left or right."
- "We stack 4 consecutive frames together. Now the network sees a mini movie — it can infer velocity, acceleration, and direction from how things change between frames."
- "Think of it like a flipbook: one page shows a picture, four pages show movement."
- "The input to our network is (4, 84, 84) — 4 channels of 84x84 images."

## CNN Architecture — Spatial Feature Extraction

- "We can't use a feedforward net on images — it would need 4 * 84 * 84 = 28,224 input neurons and would lose all spatial structure."
- "Instead we use a Convolutional Neural Network (CNN). Convolutional layers slide small filters across the image to detect features: edges, curves, road boundaries."
- "Architecture: 3 conv layers progressively shrink the spatial dimensions while increasing feature depth: (4, 84, 84) -> (32, 20, 20) -> (64, 9, 9) -> (64, 7, 7). Then 2 fully connected layers map to Q-values."
- "This is essentially the same architecture DeepMind used for Atari games in 2015."

## Action Discretization — Making DQN Work

- "CarRacing has continuous actions: steering [-1, 1], gas [0, 1], brake [0, 1]. DQN only handles discrete choices."
- "Solution: define 12 meaningful action combinations — do nothing, turn left, turn right, gas, brake, turn+gas, gentle gas, gentle turn+gas, turn+brake."
- "This is a trade-off: we lose fine-grained control but gain the ability to use DQN. For more precise control, you'd use algorithms like PPO or SAC that handle continuous actions natively."

## Training Tricks That Make It Work

- "Naive DQN fails on CarRacing. These engineering decisions are what make training converge:"
  - "**Buffer warmup (5,000 steps)**: collect random experiences before learning. Avoids training on an almost-empty buffer with no diversity."
  - "**Learn every 4 steps**: reduces correlation between consecutive updates. The environment changes slowly frame-to-frame."
  - "**Action repeat (4 frames)**: hold each action for 4 frames. Single frames change too little to create meaningful state transitions."
  - "**Soft target updates (tau=0.005)**: instead of copying weights every N episodes, slowly blend the target network toward the policy network each step. Smoother, more stable."
  - "**Huber loss instead of MSE**: less sensitive to large reward outliers (like a big crash penalty), prevents gradient explosions."
  - "**Gradient clipping (max_norm=10)**: safety net against unstable gradients."

## The Negative Reward Cutoff

- "The environment doesn't have a natural 'you failed' signal like FrozenLake. The car can drive off-track indefinitely, accumulating negative reward."
- "We add an early stopping rule: if the agent gets 50 consecutive negative-reward steps, we end the episode. This prevents wasting time on hopeless episodes and speeds up training significantly."

## Training Resume Capability

- "CarRacing takes a long time to train (1,500 episodes). We save training state — rewards history, best average reward, episode count — alongside model checkpoints."
- "This means you can stop training, come back later, and resume exactly where you left off with --resume flag."
- "Practical consideration for anyone doing RL research: always save enough state to resume."

## Results

- "The solved threshold is 900 average reward."
- "Early episodes: the car barely moves or drives straight off track (reward near 0 or negative)."
- "Mid-training: the car learns to follow the road but struggles with sharp turns."
- "Late training: smooth driving, proper braking into corners, consistent lap completion."

## Key Takeaways to Emphasize

1. **Same algorithm, harder problem** — it's still DQN with replay and target networks, but image input requires CNN + preprocessing + frame stacking.
2. **Engineering matters as much as the algorithm** — buffer warmup, action repeat, soft updates, early stopping. Without these, the same DQN algorithm fails completely.
3. **Action discretization is a practical compromise** — real-world RL often requires these kinds of approximations.
4. **Progression of complexity** — FrozenLake (table, 16 states) -> LunarLander (small net, 8-dim vector) -> CarRacing (CNN, raw pixels). Same core idea at every level.