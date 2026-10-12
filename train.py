import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
from collections import deque
from environment import HeadlessCartPoleEnv

# ==========================================
# 🧠 THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Input: 4 game variables (pos, vel, angle, angular_vel)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Output: 2 values (Q-values for Left and Right)
        )
        
    def forward(self, x):
        return self.network(x)

# Hyperparameters
LR = 0.001
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPISODES = 150

env = HeadlessCartPoleEnv()
policy_net = DQN(state_dim=4, action_dim=2)
target_net = DQN(state_dim=4, action_dim=2)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = 1.0

print("🚀 Starting training loops in Codespaces...")

# Simple DQN Training Loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    
    while True:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()
                
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        # Optimize the Neural Network
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
            
            current_q = policy_net(states).gather(1, actions).squeeze()
            max_next_q = target_net(next_states).max(1)[0]
            expected_q = rewards + (GAMMA * max_next_q * (1 - dones))
            
            loss = nn.MSELoss()(current_q, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if done:
            break
            
    epsilon = max(0.01, epsilon * 0.98)
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode} | Score: {total_reward} | Epsilon: {epsilon:.2f}")

print("\n🎉 Training complete! Rendering the final agent's gameplay to a video file...")

# ==========================================
# 📹 RECORD THE RESULTS TO A VIDEO FILE
# ==========================================
state = env.reset()
video_writer = cv2.VideoWriter('agent_gameplay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (600, 400))

for _ in range(300): # Record up to 300 steps of testing
    frame = env.render_frame()
    # Convert RGB (Pygame) to BGR (OpenCV expects BGR)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)
    
    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state)).argmax().item()
    state, _, done = env.step(action)
    if done:
        break

video_writer.release()
print("💾 Done! Open 'agent_gameplay.mp4' in your Codespaces sidebar to view the model playing!")
