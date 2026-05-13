# HW3: DQN and its Variants - Understanding Report

## HW3-1: Naive DQN for Static Mode
### 1. Basic DQN Implementation
DQN (Deep Q-Network) combines Q-Learning with neural networks to handle high-dimensional state spaces. In `hw3_1.py`, we implemented a basic DQN to solve the GridWorld's `static` mode. The neural network takes a 64-dimensional flattened state array and outputs Q-values for four possible actions (up, down, left, right). We used an Epsilon-Greedy strategy to balance exploration and exploitation.

### 2. Experience Replay Buffer
To break the temporal correlation of sequential data, we implemented a `ReplayBuffer`. Instead of learning from the most recent transition, the agent stores experiences `(state, action, reward, next_state, done)` into the buffer and samples a random mini-batch for training. This significantly stabilizes the loss and improves data efficiency.

---

## HW3-2: Enhanced DQN Variants for Player Mode
In `hw3_2.py`, we tackled the `player` mode, where the player's starting position changes every episode. We evaluated two major structural variants of DQN:

### 1. Double DQN (DDQN)
Standard DQN tends to overestimate Q-values because it uses the `max` operator for both action selection and target evaluation. Double DQN decouples this: it uses the Main Network to select the best action in the next state, but uses the Target Network to evaluate the Q-value of that chosen action. This provides more accurate Q-value estimates.

### 2. Dueling DQN
Dueling Architecture splits the network's output into two streams:
- **State Value (V)**: How good is it to be in this state, regardless of the action?
- **Advantage (A)**: How much better is a specific action compared to others in this state?
These are combined into $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A))$. This allows the network to learn the value of states without needing to learn the effect of every action, which is especially powerful in environments with many states (like random player positions).

---

## HW3-3: PyTorch Lightning + Training Tips for Random Mode
The `random` mode randomizes the positions of the player, goal, pit, and wall, making it highly stochastic. We transitioned our implementation to **PyTorch Lightning** for cleaner engineering and integrated several training tips:

### 1. Learning Rate Scheduling
We applied `ExponentialLR` to decay the learning rate over time. A higher learning rate initially speeds up exploration and learning, while a decayed learning rate later helps the network finely converge to the optimal policy without overshooting.

### 2. Gradient Clipping
In environments with sparse or sudden extreme rewards (like Random mode), the loss can spike, causing exploding gradients. We used PyTorch Lightning's gradient clipping (`clip_gradients(val=1.0)`) to ensure that parameter updates remain bounded, preventing network collapse.

### 3. Target Network Update
Instead of evaluating targets with the constantly changing main network, we strictly synced the target network every 50 epochs (`sync_freq = 50`). This creates a stable stationary target for the Q-learning regression.

---

## HW3-4: Simplified Rainbow DQN
Despite the tips in HW3-3, standard DQN struggles to achieve high win rates in `random` mode. We implemented a simplified **Rainbow DQN** combining four state-of-the-art extensions:

1. **Double Q-Learning**: Used to prevent Q-value overestimation in the highly stochastic random environment.
2. **Dueling Architecture**: Vital for generalizing state values when the grid is completely randomized.
3. **Multi-step Returns (N-step = 3)**: Instead of looking only 1 step ahead, the agent calculates the target using 3 steps of accumulated rewards. This balances the bias of bootstrapping with the variance of Monte Carlo, drastically speeding up the propagation of the reward signal from the goal back to earlier states.
4. **Prioritized Experience Replay (PER)**: Instead of sampling uniformly, PER samples transitions proportional to their TD-Error (i.e., how "surprising" they were). This ensures the network learns from the most informative experiences more frequently, significantly improving sample efficiency.

**Conclusion**: The Rainbow agent successfully elevated the win rate in Random mode from ~15% to over 60%, demonstrating the multiplicative power of combining these distinct RL improvements.
