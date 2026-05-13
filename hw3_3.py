import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return idx

class LitDQN(pl.LightningModule):
    def __init__(self, epochs=2000):
        super().__init__()
        self.env = Gridworld(size=4, mode='random')
        self.main_model = DQN()
        self.target_model = copy.deepcopy(self.main_model)
        self.buffer = ReplayBuffer(capacity=1000)
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.batch_size = 50
        self.sync_freq = 50
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        
        self.state = self.get_state(self.env)
        self.ep_reward = 0
        self.losses = []
        self.rewards = []
        
        self.automatic_optimization = False
        
    def get_state(self, env):
        state = env.board.render_np().reshape(1, 64)
        return state + np.random.rand(1, 64)/100.0

    def forward(self, x):
        return self.main_model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.main_model.parameters(), lr=1e-3)
        # Requirement 2: Learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        done = False
        status = 1
        
        # Rollout an entire episode
        while status == 1:
            qval = self.main_model(torch.from_numpy(self.state).float().to(self.device))
            
            if random.random() < self.epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = torch.argmax(qval).item()
                
            action = action_set[action_]
            self.env.makeMove(action)
            
            reward = self.env.reward()
            self.ep_reward += reward
            next_state = self.get_state(self.env)
            
            if reward == -10 or reward == 10:
                done = True
                status = 0
            
            self.buffer.push(self.state, action_, reward, next_state, done)
            self.state = next_state
            
            if len(self.buffer) > self.batch_size:
                s_batch, a_batch, r_batch, ns_batch, done_batch = self.buffer.sample(self.batch_size)
                
                s_batch = torch.from_numpy(s_batch).float().squeeze(1).to(self.device)
                ns_batch = torch.from_numpy(ns_batch).float().squeeze(1).to(self.device)
                a_batch = torch.from_numpy(a_batch).to(self.device)
                r_batch = torch.from_numpy(r_batch).float().to(self.device)
                done_batch = torch.from_numpy(done_batch).float().to(self.device)
                
                qval_batch = self.main_model(s_batch)
                
                with torch.no_grad():
                    next_q_target = self.target_model(ns_batch)
                    target_q_val = torch.max(next_q_target, dim=1)[0]
                
                Y = r_batch + self.gamma * ((1 - done_batch) * target_q_val)
                X = qval_batch.gather(dim=1, index=a_batch.unsqueeze(1)).squeeze(1)
                
                loss = self.loss_fn(X, Y)
                
                opt.zero_grad()
                self.manual_backward(loss)
                # Requirement 1: Gradient clipping
                self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt.step()
                
                self.losses.append(loss.item())

        self.rewards.append(self.ep_reward)
        self.log('train_reward', float(self.ep_reward), prog_bar=True)
        
        # Reset environment for next epoch
        self.ep_reward = 0
        self.state = self.get_state(self.env)
        
        # Requirement 3: Target network update
        if self.current_epoch % self.sync_freq == 0:
            self.target_model.load_state_dict(self.main_model.state_dict())
            
        if self.epsilon > 0.1:
            self.epsilon -= (1/self.epochs)
            
        sch.step()
            
        if self.current_epoch % 100 == 0:
            print(f"Epoch: {self.current_epoch}, Epsilon: {self.epsilon:.2f}, Reward: {self.rewards[-1]}")
            
        return None

    def train_dataloader(self):
        dataset = DummyDataset(1)
        return DataLoader(dataset, batch_size=1)

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
    print("\n--- Testing Model ---")
    env = Gridworld(size=4, mode='random')
    model.eval()
    wins = 0
    total_steps = 0
    frames = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for i in range(test_epochs):
        env.initGridRand()
        state = model.get_state(env)
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
                qval = model(torch.from_numpy(state).float().to(model.device))
            action_ = torch.argmax(qval).item()
            action = action_set[action_]
            env.makeMove(action)
            
            reward = env.reward()
            state = model.get_state(env)
            
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
        frames[0].save("hw3_3_test.gif", save_all=True, append_images=frames[1:], loop=0, duration=300)
        print("Saved test animation to hw3_3_test.gif")
    
    print(f"Testing Complete. Win Rate: {wins/test_epochs*100:.1f}%, Avg Steps: {total_steps/test_epochs:.1f}")

def main():
    epochs = 2000
    model = LitDQN(epochs=epochs)
    
    # Use PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    
    print("Training Complete for HW3-3 (PyTorch Lightning)!")
    
    # Moving average helper
    def moving_average(a, n=50):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # Plot Loss Curve
    plt.figure()
    plt.plot(moving_average(model.losses, n=100))
    plt.title("HW3-3 Loss Curve")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.savefig("loss_curve_3_3.png")
    
    # Plot Reward Curve
    plt.figure()
    plt.plot(moving_average(model.rewards, n=50))
    plt.title("HW3-3 Reward Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.savefig("reward_curve_3_3.png")
    
    # Test Model
    test(model)

if __name__ == '__main__':
    main()
