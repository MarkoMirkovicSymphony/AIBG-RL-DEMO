# Introduction to Reinforcement Learning

## What is Reinforcement Learning?

- "Reinforcement learning is how an agent learns to make decisions by trial and error. There's no labeled dataset — no one tells it the right answer. It tries things, gets feedback, and improves."
- "Think of it like teaching a dog a trick: you don't explain the trick, you just reward good behavior and the dog figures it out."
- "This is fundamentally different from supervised learning (where you have correct answers) and unsupervised learning (where you find patterns). RL is about sequential decision-making under uncertainty."

## The Core Components

- "Every RL problem has the same five pieces:"
  - "**Agent** — the learner. Our algorithm that makes decisions."
  - "**Environment** — the world the agent lives in. A grid, a game, a physics simulation."
  - "**State** — where the agent currently is. Could be a grid position, a vector of numbers, or even an image."
  - "**Action** — what the agent can do. Move left, fire a thruster, turn a steering wheel."
  - "**Reward** — the feedback signal. A number that tells the agent how good or bad that step was."
- "The agent observes a state, picks an action, receives a reward, lands in a new state. Repeat forever."

## The Goal — Maximize Cumulative Reward

- "The agent doesn't just want the best immediate reward — it wants the best total reward over time."
- "This is why we have the discount factor (gamma). A reward now is worth more than a reward later. With gamma = 0.99, a reward 100 steps in the future is worth about 37% of an immediate reward."
- "This creates interesting tradeoffs: sometimes the agent must accept short-term pain (using fuel, taking a longer path) for long-term gain (safe landing, reaching the goal)."

## Policy — The Agent's Strategy

- "A policy is simply: given a state, what action do I take?"
- "A random policy picks actions at random — terrible, but it's where every agent starts."
- "The optimal policy picks the action that leads to the highest expected cumulative reward from every state."
- "The whole point of training is to discover this optimal policy through experience."

## Q-Values — The Heart of Q-Learning

- "A Q-value answers one question: 'If I'm in state S and I take action A, then act optimally from there, what total reward do I expect?'"
- "If you know the true Q-values for every state-action pair, the optimal policy is trivial: always pick the action with the highest Q-value."
- "Q-learning is the algorithm that discovers these values through experience. It never needs a model of how the environment works — it learns purely from (state, action, reward, next_state) tuples."

## The Bellman Equation — How Q-Values Update

- "The update rule: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s')) - Q(s,a)]"
- "In plain English: after taking action A in state S, the agent asks 'was this better or worse than I expected?' and nudges the Q-value accordingly."
- "The key insight: the value of a state depends on the value of the next state. Good states lead to good states. This recursive relationship is the Bellman equation — the mathematical foundation of all value-based RL."
- "Over thousands of updates, these nudges converge to the true expected rewards."

## Exploration vs. Exploitation

- "This is THE fundamental tradeoff in RL."
- "Exploitation: do what you currently think is best. Go to the restaurant you know is good."
- "Exploration: try something new. Go to a restaurant you've never been to — it might be better, might be worse."
- "If you only exploit, you might miss a better strategy you never tried. If you only explore, you never capitalize on what you've learned."
- "Epsilon-greedy is the simplest solution: with probability epsilon take a random action (explore), otherwise take the best known action (exploit). Start epsilon high, decay it over time."

## From Tables to Neural Networks

- "When the state space is small and discrete (like a 4x4 grid), you can store Q-values in a literal table. This is tabular Q-learning."
- "When states are continuous (position = 3.7, velocity = -0.2...) a table is impossible — infinite rows."
- "Solution: replace the table with a neural network. Input: state. Output: Q-values for all actions. The network generalizes — it can estimate Q-values for states it has never seen."
- "This is Deep Q-Learning (DQN). Same Bellman update, but applied through gradient descent on a neural network."

## What Makes DQN Stable

- "Naively combining neural networks with Q-learning is unstable. Two tricks fix this:"
  - "**Experience Replay**: store past experiences in a buffer, train on random batches. Breaks temporal correlation — consecutive experiences are too similar to learn from directly."
  - "**Target Network**: use a frozen copy of the network to compute training targets. Without this, you're chasing a moving target — the thing you're trying to predict changes every time you update."
- "These two ideas, from the 2015 DeepMind paper, are what made deep RL actually work."

## Today's Demo Progression

- "We'll see the same core idea applied at three levels of complexity:"
  1. "**FrozenLake (Q-table)** — 16 states, 4 actions, a literal table. See the algorithm in its purest form."
  2. "**LunarLander (DQN with feedforward net)** — 8 continuous state values, small neural network. The jump from table to function approximation."
  3. "**CarRacing (DQN with CNN)** — raw pixel input, convolutional network, frame stacking. The full deep RL pipeline."
- "Same Bellman equation at every level. Same exploration/exploitation tradeoff. The only difference is how we represent and generalize Q-values."