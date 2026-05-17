# Q-Learning on FrozenLake — Speaker Notes

## The Problem — FrozenLake

- "We're working with FrozenLake — a 4x4 grid world. The agent starts at the top-left and needs to reach the goal at the bottom-right, without falling into holes."
- "What makes this interesting is the slippery ice. When you tell the agent to go right, there's only a 1/3 chance it actually goes right — it might slip up or down. This means even a perfect policy can't win every time."
- "This is why we need RL here — you can't just hardcode a path when the environment is uncertain."

## What the Agent Knows — The Q-Table

- "The agent's entire knowledge is stored in a simple 16x4 table. 16 states (one per grid cell), 4 actions (left, down, right, up)."
- "Each cell answers one question: 'How good is it to take this action from this state?' — that's the Q-value."
- "At the start, the table is all zeros — the agent knows absolutely nothing. Everything it learns comes from trial and error."

## Exploration vs. Exploitation (Epsilon-Greedy)

- "This is the fundamental tradeoff in RL. Should the agent try something new, or go with what it already knows works?"
- "We control this with epsilon. Early on, epsilon is 1.0 — the agent acts completely randomly. This is exploration — it's discovering the environment."
- "Over time, epsilon decays toward 0.01. The agent starts trusting its Q-values and picks the best known action. This is exploitation."
- "The decay rate matters — decay too fast and the agent gets stuck with a bad policy. Too slow and training takes forever."

## The Learning Rule — Bellman Update

- "After every single step, the agent updates its Q-table. The formula looks like this:"
  - `Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]`
- "In plain English: the agent asks 'Was this action better or worse than I expected?' and nudges the Q-value accordingly."
- "Alpha (0.8) is the learning rate — how aggressively we update. Gamma (0.99) is the discount factor — how much the agent cares about future rewards vs. immediate ones."
- "Over thousands of episodes, these small nudges converge to the true expected rewards."

## Training Progress

- "We train for 20,000 episodes. Early on, the agent wins rarely — it's mostly exploring randomly."
- "Around episode 5,000-8,000, the reward curve climbs steeply — the agent has discovered successful paths and is refining them."
- "By the end, the smoothed reward plateaus — the agent has converged on its best policy."
- "We save checkpoints every 2,000 episodes so we can inspect how the Q-table evolves over time."

## Results

- "After training, we evaluate with pure exploitation — no randomness, always pick the best action."
- "On the slippery version, the agent achieves roughly 70-80% win rate. That's near-optimal given that the ice randomness makes 100% impossible."
- "We can visualize the learned policy as arrows on the grid — you'll see it generally points toward the goal while avoiding holes."

## Key Takeaways to Emphasize

1. **No model needed** — the agent never sees the grid layout or transition probabilities. It learns purely from experience.
2. **The Q-table is the knowledge** — thousands of episodes compressed into a simple lookup table.
3. **Stochasticity is the challenge** — deterministic FrozenLake is trivial; the slippery version is where RL shines.
4. **Tabular Q-learning doesn't scale** — this works for 16 states, but for larger problems (Atari, robotics) we need function approximation (Deep Q-Networks). That's the bridge to deep RL.