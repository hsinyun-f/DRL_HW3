import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import io
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from env.Gridworld import Gridworld

# 1. Action Mapping
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# 3. DQN Model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Gridworld state is 4x4x4 = 64
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1, 64)/100.0 # add small noise as in book

def train():
    env = Gridworld(size=4, mode='static')
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    epochs = 1000
    gamma = 0.9
    epsilon = 1.0
    batch_size = 50
    buffer = ReplayBuffer(capacity=1000)
    
    losses = []
    rewards = []
    
    for i in range(epochs):
        state = get_state(env)
        done = False
        status = 1
        ep_reward = 0
        
        while status == 1:
            qval = model(torch.from_numpy(state).float())
            
            # Epsilon-greedy
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
            
            buffer.push(state, action_, reward, next_state, done)
            state = next_state
            
            if len(buffer) > batch_size:
                s_batch, a_batch, r_batch, ns_batch, done_batch = buffer.sample(batch_size)
                
                s_batch = torch.from_numpy(s_batch).float().squeeze(1)
                ns_batch = torch.from_numpy(ns_batch).float().squeeze(1)
                a_batch = torch.from_numpy(a_batch)
                r_batch = torch.from_numpy(r_batch).float()
                done_batch = torch.from_numpy(done_batch).float()
                
                qval_batch = model(s_batch)
                with torch.no_grad():
                    nqval_batch = model(ns_batch)
                
                Y = r_batch + gamma * ((1 - done_batch) * torch.max(nqval_batch, dim=1)[0])
                X = qval_batch.gather(dim=1, index=a_batch.unsqueeze(1)).squeeze(1)
                
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        
        rewards.append(ep_reward)
        
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if i % 100 == 0:
            print(f"Epoch: {i}, Epsilon: {epsilon:.2f}, Reward: {ep_reward}")

    print("Training Complete for HW3-1 (Static Mode)!")
    
    # Plot Loss Curve
    plt.figure()
    plt.plot(losses)
    plt.title("HW3-1 Loss Curve")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.savefig("loss_curve_3_1.png")
    
    # Plot Reward Curve
    plt.figure()
    plt.plot(rewards)
    plt.title("HW3-1 Reward Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("reward_curve_3_1.png")
    
    return model, env

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

def test(model, env, test_epochs=10):
    print("\n--- Testing Model ---")
    wins = 0
    total_steps = 0
    frames = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for i in range(test_epochs):
        env.initGridStatic() # Reset static environment
        state = get_state(env)
        done = False
        status = 1
        steps = 0
        
        if i < 3:
            print(f"\n=== Episode {i+1} Rendering ===")
            render_board(env, steps, 0, "Init", ax)
            frames.append(get_fig_image(fig))
            
        while status == 1 and steps < 50: # max 50 steps
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
        print(f"Test Episode {i+1}: {'Win' if reward==10 else 'Loss'}, Steps: {steps}")
        
    plt.ioff()
    plt.close()
    
    if frames:
        frames[0].save("hw3_1_test.gif", save_all=True, append_images=frames[1:], loop=0, duration=300)
        print("Saved test animation to hw3_1_test.gif")
        
    print(f"Testing Complete. Win Rate: {wins/test_epochs*100:.1f}%, Avg Steps: {total_steps/test_epochs:.1f}")

if __name__ == '__main__':
    model, env = train()
    test(model, env)
