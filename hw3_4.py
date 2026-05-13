import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import time
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from env.Gridworld import Gridworld

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        state, action, reward, next_state, done = map(np.stack, zip(*samples))
        return state, action, reward, next_state, done, indices, weights
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.9):
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_step)
        
    def append(self, transition): 
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return None
        return self._get_n_step_info()
        
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][2], self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
                
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.val = nn.Linear(100, 1)
        self.adv = nn.Linear(100, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.val(x)
        a = self.adv(x)
        qvals = v + (a - a.mean(dim=1, keepdim=True))
        return qvals

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1, 64)/100.0

def train_rainbow(epochs=2000):
    print("\n--- Training Simplified Rainbow DQN (Random Mode) ---")
    env = Gridworld(size=4, mode='random')
    
    main_model = DuelingDQN()
    target_model = copy.deepcopy(main_model)
    
    optimizer = optim.Adam(main_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none') # PER needs individual losses
    
    gamma = 0.9
    epsilon = 1.0
    batch_size = 50
    n_step = 3
    
    buffer = PrioritizedReplayBuffer(capacity=2000)
    n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma)
    sync_freq = 50
    
    losses = []
    rewards = []
    
    for i in range(epochs):
        env.initGridRand()
        state = get_state(env)
        done = False
        status = 1
        ep_reward = 0
        
        while status == 1:
            qval = main_model(torch.from_numpy(state).float())
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = torch.argmax(qval).item()
                
            action = action_set[action_]
            env.makeMove(action)
            
            reward = env.reward()
            ep_reward += reward
            next_state = get_state(env)
            
            if reward == -10 or reward == 10:
                done = True
                status = 0
            
            # 1. Multi-step Return
            transition = n_step_buffer.append((state, action_, reward, next_state, done))
            if transition:
                buffer.push(*transition)
            
            # Also flush n-step buffer at episode end
            if done:
                while len(n_step_buffer.n_step_buffer) > 1:
                    n_step_buffer.n_step_buffer.popleft()
                    transition = n_step_buffer._get_n_step_info()
                    buffer.push(*transition)
                n_step_buffer.n_step_buffer.clear()
            
            state = next_state
            
            if len(buffer) > batch_size:
                # 2. Prioritized Experience Replay
                beta = min(1.0, 0.4 + i * (1.0 - 0.4) / epochs) # beta annealing
                s_batch, a_batch, r_batch, ns_batch, done_batch, indices, weights = buffer.sample(batch_size, beta)
                
                s_batch = torch.from_numpy(s_batch).float().squeeze(1)
                ns_batch = torch.from_numpy(ns_batch).float().squeeze(1)
                a_batch = torch.from_numpy(a_batch)
                r_batch = torch.from_numpy(r_batch).float()
                done_batch = torch.from_numpy(done_batch).float()
                weights_t = torch.from_numpy(weights).float()
                
                qval_batch = main_model(s_batch)
                
                with torch.no_grad():
                    # 3. Double DQN
                    next_q_main = main_model(ns_batch)
                    best_action = torch.argmax(next_q_main, dim=1)
                    
                    next_q_target = target_model(ns_batch)
                    target_q_val = next_q_target.gather(dim=1, index=best_action.unsqueeze(1)).squeeze(1)
                
                # N-step gamma
                gamma_n = gamma ** n_step
                Y = r_batch + gamma_n * ((1 - done_batch) * target_q_val)
                X = qval_batch.gather(dim=1, index=a_batch.unsqueeze(1)).squeeze(1)
                
                # Calculate TD error for PER
                td_errors = torch.abs(X - Y).detach().numpy()
                buffer.update_priorities(indices, td_errors + 1e-5)
                
                loss = (loss_fn(X, Y) * weights_t).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        rewards.append(ep_reward)
        
        if i % sync_freq == 0:
            target_model.load_state_dict(main_model.state_dict())
            
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if i % 100 == 0:
            print(f"Epoch: {i}, Epsilon: {epsilon:.2f}, Reward: {ep_reward}")

    print("Training Complete for HW3-4 (Simplified Rainbow DQN)!")
    return main_model, losses, rewards

def render_board(env, step, reward, action, ax):
    board = env.display()
    size = env.board.size
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i, j] == '+': grid[i, j] = 1
            elif board[i, j] == '-': grid[i, j] = 2
            elif board[i, j] == 'W': grid[i, j] = 3
            elif board[i, j] == 'P': grid[i, j] = 4
            
    cmap = mcolors.ListedColormap(['white', 'green', 'red', 'gray', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.clear()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Step: {step} | Action: {action} | Reward: {reward}")
    
    plt.draw()
    plt.pause(0.3)

def get_fig_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def test(model, test_epochs=100):
    print("\n--- Testing Rainbow Model ---")
    env = Gridworld(size=4, mode='random')
    model.eval()
    wins = 0
    total_steps = 0
    frames = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for i in range(test_epochs):
        env.initGridRand()
        state = get_state(env)
        done = False
        status = 1
        steps = 0
        
        if i < 3:
            print(f"\n=== Episode {i+1} Rendering ===")
            render_board(env, steps, 0, "Init", ax)
            frames.append(get_fig_image(fig))
            
        while status == 1 and steps < 50:
            steps += 1
            with torch.no_grad():
                qval = model(torch.from_numpy(state).float())
            action_ = torch.argmax(qval).item()
            action = action_set[action_]
            env.makeMove(action)
            
            reward = env.reward()
            state = get_state(env)
            
            if i < 3:
                render_board(env, steps, reward, action, ax)
                frames.append(get_fig_image(fig))
                
            if reward == 10:
                status = 0
                wins += 1
            elif reward == -10:
                status = 0
                
        total_steps += steps
        
    plt.ioff()
    plt.close()
    
    if frames:
        frames[0].save("hw3_4_test.gif", save_all=True, append_images=frames[1:], loop=0, duration=300)
        print("Saved test animation to hw3_4_test.gif")
    
    print(f"Testing Complete. Win Rate: {wins/test_epochs*100:.1f}%, Avg Steps: {total_steps/test_epochs:.1f}")

if __name__ == '__main__':
    model, losses, rewards = train_rainbow(epochs=2000)
    
    # Moving average helper
    def moving_average(a, n=50):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # Plot Loss Curve
    plt.figure()
    plt.plot(moving_average(losses, n=100))
    plt.title("HW3-4 Rainbow DQN Loss Curve")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.savefig("loss_curve_3_4.png")
    
    # Plot Reward Curve
    plt.figure()
    plt.plot(moving_average(rewards, n=50))
    plt.title("HW3-4 Rainbow DQN Reward Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("reward_curve_3_4.png")
    
    # Run test
    test(model)
