# agent.py

import paddle
import paddle.nn as nn
from paddle.nn import ClipGradByNorm
import paddle.optimizer as optim
import numpy as np
import random
import os

from replay_buffer import ReplayBuffer
from config import (STATE_DIM, ACTION_DIM, LEARNING_RATE, GAMMA, BUFFER_SIZE, BATCH_SIZE,
                    EPSILON_START, EPSILON_DECAY, EPSILON_MIN, TARGET_UPDATE_FREQ, 
                    MODEL_SAVE_PATH, GRAD_CLIP_NORM)

# --- 定义 Dueling DQN 网络结构 ---
class DuelingDQNNet(nn.Layer):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享层
        self.shared_fc1 = nn.Linear(state_dim, 128)
        self.shared_fc2 = nn.Linear(128, 128)

        # Value 流
        self.value_fc = nn.Linear(128, 64)
        self.value_output = nn.Linear(64, 1) # V(s)

        # Advantage 流
        self.advantage_fc = nn.Linear(128, 64)
        self.advantage_output = nn.Linear(64, action_dim) # A(s, a)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.shared_fc1(state))
        x = self.relu(self.shared_fc2(x))

        value = self.relu(self.value_fc(x))
        value = self.value_output(value) # V(s)

        advantage = self.relu(self.advantage_fc(x))
        advantage = self.advantage_output(advantage) # A(s, a)

        advantage_mean = paddle.mean(advantage, axis=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values

# --- 定义 Dueling DQN Agent ---
class DuelingDQNAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 learning_rate=LEARNING_RATE, gamma=GAMMA,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 epsilon_start=EPSILON_START, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                 target_update_freq=TARGET_UPDATE_FREQ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0

        self.main_network = DuelingDQNNet(state_dim, action_dim)
        self.target_network = DuelingDQNNet(state_dim, action_dim)
        self.update_target_network() 

        grad_clipper = ClipGradByNorm(clip_norm=GRAD_CLIP_NORM)
        
        self.optimizer = optim.Adam(
            learning_rate=learning_rate, 
            parameters=self.main_network.parameters(),
            grad_clip=grad_clipper
        )

        self.loss_function = nn.SmoothL1Loss()

        print("PaddlePaddle Dueling DQN Agent Initialized.")

    def remember(self, state, action, reward, next_state, done, mask, next_mask):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        next_mask = np.asarray(next_mask, dtype=bool)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.memory.store((state, action, reward, next_state, done, mask, next_mask))

    def select_action(self, state, mask, explore=True):
        # 1. 探索阶段 (Epsilon-Greedy)
        if explore and np.random.rand() <= self.epsilon:
            # 即使是随机探索，也只在 Mask 为 True 的有效动作里选
            valid_actions = np.where(mask)[0]
            if valid_actions.size > 0:
                return np.random.choice(valid_actions)
            return 0 # 保底

        # 2. 利用阶段 (Argmax Q with Hard Masking)
        state_tensor = paddle.to_tensor(state[np.newaxis, :], dtype='float32')
        self.main_network.eval() 
        with paddle.no_grad():
            q_values = self.main_network(state_tensor)[0] 
        self.main_network.train() 

        # [核心修改] 硬约束 Action Masking
        # 将 Mask 为 False 的位置 Q 值设为 -1e9 (负无穷)，确保 Argmax 绝不选中
        mask_tensor = paddle.to_tensor(mask, dtype='bool')
        inf_mask = paddle.full_like(q_values, -1e9) 
        
        # 如果 mask 为 True，保留原 Q 值；如果为 False，替换为 -1e9
        masked_q_values = paddle.where(mask_tensor, q_values, inf_mask)

        best_action = paddle.argmax(masked_q_values).item()
        return best_action

    def train(self):
        """采样批次并训练主网络，返回当前的 loss 值"""
        if len(self.memory) < self.batch_size:
            return None # 还没开始训练
        
        minibatch = self.memory.sample(self.batch_size)
        states_np, actions_np, rewards_np, next_states_np, dones_np, masks_np, next_masks_np = \
            map(np.array, zip(*minibatch))

        states = paddle.to_tensor(states_np, dtype='float32')
        actions = paddle.to_tensor(actions_np, dtype='int64')
        rewards = paddle.to_tensor(rewards_np, dtype='float32')
        next_states = paddle.to_tensor(next_states_np, dtype='float32')
        dones = paddle.to_tensor(dones_np, dtype='bool')
        next_masks = paddle.to_tensor(next_masks_np, dtype='bool')

        # --- Double DQN 逻辑 ---
        # 1. 使用 主网络 选择最佳动作 (Selection)
        self.main_network.eval()
        with paddle.no_grad():
            next_q_values_main = self.main_network(next_states)
            
            # [训练时的 Hard Masking]
            # 对 Next State 的 Q 值也应用 Mask，防止 Target 计算错误
            next_inf_mask = paddle.full_like(next_q_values_main, -1e9)
            next_q_values_main = paddle.where(next_masks, next_q_values_main, next_inf_mask)
            
            best_next_actions = paddle.argmax(next_q_values_main, axis=1, dtype='int64')
        self.main_network.train()

        # 2. 使用 目标网络 评估该动作的价值 (Evaluation)
        self.target_network.eval()
        with paddle.no_grad():
            next_q_values_target = self.target_network(next_states)
            
            # 从目标网络输出中提取 best_next_actions 对应的 Q 值
            batch_indices = paddle.arange(self.batch_size, dtype='int64')
            best_actions_indices = paddle.stack([batch_indices, best_next_actions], axis=1)
            target_q_values_of_best_actions = paddle.gather_nd(next_q_values_target, best_actions_indices)

        # 3. 计算 TD Targets
        dones_float = paddle.cast(dones, 'float32')
        td_targets = rewards + self.gamma * target_q_values_of_best_actions * (1.0 - dones_float)

        # --- 训练主网络 ---
        self.main_network.train()
        q_values_main = self.main_network(states)
        actions_long = paddle.cast(actions, 'int64')
        batch_indices = paddle.arange(self.batch_size, dtype='int64')
        action_indices = paddle.stack([batch_indices, actions_long], axis=1)
        action_q_values = paddle.gather_nd(q_values_main, action_indices)

        loss = self.loss_function(action_q_values, td_targets)
        loss_value = loss.item() # 获取 loss 数值

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_counter += 1
        return loss_value

    def update_target_network_episode(self, episode):
        if episode > 0 and episode % self.target_update_freq == 0:
            self.target_network.set_state_dict(self.main_network.state_dict())

    def update_target_network(self):
        self.target_network.set_state_dict(self.main_network.state_dict())

    def get_q_value(self, state, action):
        if state is None: return 0.0
        state_tensor = paddle.to_tensor(state[np.newaxis, :], dtype='float32')
        self.main_network.eval()
        with paddle.no_grad():
            q_values = self.main_network(state_tensor)[0]
        self.main_network.train()
        action = int(action)
        if 0 <= action < self.action_dim:
            return float(q_values[action].item())
        else:
            return 0.0

    def save_model(self, filepath=MODEL_SAVE_PATH):
        try:
            paddle.save(self.main_network.state_dict(), filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath=MODEL_SAVE_PATH):
        if os.path.exists(filepath):
            try:
                state_dict = paddle.load(filepath)
                self.main_network.set_state_dict(state_dict)
                self.target_network.set_state_dict(state_dict)
                self.epsilon = self.epsilon_min 
                print(f"Model loaded from {filepath}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No model found, starting scratch.")

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)