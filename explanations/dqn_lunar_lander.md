# DQN on LunarLander — Speaker Notes

## Why DQN? The Limitation of Q-Tables

- "In FrozenLake we had 16 discrete states — a table works fine. But LunarLander has continuous states: position, velocity, angle, angular velocity, leg contact. There are infinite possible states."
- "You can't build a row for every possible (x=0.3142, velocity=-0.872...) combination. We need something that generalizes."
- "The solution: replace the table with a neural network. Give it a state vector, it outputs Q-values for all actions. It learns to generalize across similar states it has never seen before."

## The Environment — LunarLander-v3

- "The agent controls a spacecraft with 4 actions: do nothing, fire left thruster, fire main engine, fire right thruster."
- "The state is 8 numbers: x position, y position, x velocity, y velocity, angle, angular velocity, left leg contact, right leg contact."
- "Reward: +200 for a safe landing, negative for crashing or using too much fuel. The agent must balance precision with fuel economy."

## The Network Architecture

- "A simple feedforward net: 8 inputs (the state vector) -> 64 neurons -> 64 neurons -> 4 outputs (Q-value per action)."
- "ReLU activations between layers introduce non-linearity — without them the network could only learn linear relationships."
- "This is intentionally simple. LunarLander doesn't need a massive network — it's about the algorithm, not the architecture."

## Trick #1 — Experience Replay

- "If the agent learns from consecutive experiences, it sees highly correlated data: similar states, similar actions, similar outcomes in a row. This makes training unstable."
- "The replay buffer stores past experiences (up to 100,000). When learning, we sample a random batch of 64. This breaks temporal correlation — like shuffling a training dataset."
- "It also improves data efficiency: each experience can be used for learning multiple times, not just once."

## Trick #2 — Target Network

- "Here's the problem: we're using the network to both predict Q-values AND compute the target we're training toward. If we update the network, the target moves too."
- "It's like trying to measure something with a ruler that keeps changing length."
- "Solution: keep a frozen copy (the target network). Use it to compute targets. Every 10 episodes, sync it with the policy network."
- "This gives us stable targets to train against, making convergence much more reliable."

## The Training Loop

- "Each episode: the agent plays a full landing attempt. For each step, it picks an action (epsilon-greedy), observes the result, stores the experience, and does one gradient update on a random batch."
- "Epsilon starts at 1.0 (pure random) and decays by 0.995x per episode toward 0.01."
- "We train for 1,000 episodes. The agent typically starts solving it (reward > 200) around episode 400-600."

## The Learning Step (Bellman + Backprop)

- "The loss function is simple: MSE between what we predicted — Q(s, a) — and what the Bellman equation says it should be — reward + gamma * max(Q_target(next_state))."
- "We use Adam optimizer with a learning rate of 5e-4. Backpropagation adjusts all network weights to reduce this loss."
- "This is the same Bellman idea from Q-learning, but now we're fitting a neural network instead of updating a table cell."

## Results

- "The reward curve starts deeply negative (crashing constantly), climbs through zero (surviving but not landing), and eventually plateaus above 200 (consistent safe landings)."
- "We save the best model based on rolling average reward — this avoids keeping a model from a lucky episode."
- "The solved threshold is 200 average reward over 100 episodes."

## Key Takeaways to Emphasize

1. **DQN = Q-Learning + Neural Network** — same algorithm, but the network generalizes across continuous states.
2. **Two tricks make it work** — without replay and target network, training is hopelessly unstable.
3. **The architecture is simple** — a 2-layer feedforward net is enough. The magic is in the training algorithm.
4. **Bridge from tabular to deep RL** — this is the same Bellman update from FrozenLake, just applied through gradient descent instead of direct table edits.