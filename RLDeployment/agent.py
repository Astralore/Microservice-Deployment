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
        # 共享层计算
        x = self.relu(self.shared_fc1(state))
        x = self.relu(self.shared_fc2(x))

        # Value 流计算
        value = self.relu(self.value_fc(x))
        value = self.value_output(value) # V(s)

        # Advantage 流计算
        advantage = self.relu(self.advantage_fc(x))
        advantage = self.advantage_output(advantage) # A(s, a)

        # 合并: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        # 计算均值时保持维度以便广播
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

        # 创建主网络和目标网络
        self.main_network = DuelingDQNNet(state_dim, action_dim)
        self.target_network = DuelingDQNNet(state_dim, action_dim)
        self.update_target_network() # 初始化目标网络权重

        # 定义梯度裁剪器
        grad_clipper = ClipGradByNorm(clip_norm=GRAD_CLIP_NORM)
        
        # 定义优化器并应用裁剪器
        self.optimizer = optim.Adam(
            learning_rate=learning_rate, 
            parameters=self.main_network.parameters(),
            grad_clip=grad_clipper
        )

        # 定义损失函数 (Huber Loss)
        self.loss_function = nn.SmoothL1Loss()

        print("PaddlePaddle Dueling DQN Agent Initialized.")

    def remember(self, state, action, reward, next_state, done, mask, next_mask):
        """存储经验"""
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        next_mask = np.asarray(next_mask, dtype=bool)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.memory.store((state, action, reward, next_state, done, mask, next_mask))

    def select_action(self, state, mask, explore=True):
        """使用 epsilon-greedy 和掩码选择动作"""
        if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.shape[0] != self.action_dim:
             print(f"Error: Invalid mask received in select_action. Shape: {mask.shape if isinstance(mask, np.ndarray) else 'Not ndarray'}")
             return 0

        valid_actions = np.where(mask)[0]
        if not valid_actions.size:
            print("Warning: No valid actions available!")
            return 0 # 回退动作

        if explore and np.random.rand() <= self.epsilon:
            # 探索: 从有效动作中随机选择
            return np.random.choice(valid_actions)
        else:
            # 利用: 选择最佳有效动作
            state_tensor = paddle.to_tensor(state[np.newaxis, :], dtype='float32')
            self.main_network.eval() 
            with paddle.no_grad():
                q_values = self.main_network(state_tensor)[0] 
            self.main_network.train() 

            # 应用掩码: 将无效动作的 Q 值设为负无穷
            mask_tensor = paddle.to_tensor(mask, dtype='bool')
            masked_q_values = paddle.where(mask_tensor, q_values, paddle.full_like(q_values, -float('inf')))

            # 选择 Q 值最大的动作
            best_action = paddle.argmax(masked_q_values).item()

            # 安全检查
            if not mask[best_action]:
                 return np.random.choice(valid_actions)

            return best_action

    def train(self):
        """采样批次并训练主网络"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = self.memory.sample(self.batch_size)
        states_np, actions_np, rewards_np, next_states_np, dones_np, masks_np, next_masks_np = \
            map(np.array, zip(*minibatch))

        states = paddle.to_tensor(states_np, dtype='float32')
        actions = paddle.to_tensor(actions_np, dtype='int64')
        rewards = paddle.to_tensor(rewards_np, dtype='float32')
        next_states = paddle.to_tensor(next_states_np, dtype='float32')
        dones = paddle.to_tensor(dones_np, dtype='bool')
        masks = paddle.to_tensor(masks_np, dtype='bool')
        next_masks = paddle.to_tensor(next_masks_np, dtype='bool')

        # --- 计算 TD Targets (使用目标网络) ---
        self.target_network.eval()
        with paddle.no_grad():
            next_q_values_main = self.target_network(next_states)
            next_q_values_main[~next_masks] = -float('inf')
            best_next_actions = paddle.argmax(next_q_values_main, axis=1, dtype='int64')

        self.target_network.eval()
        with paddle.no_grad():
            next_q_values_target = self.target_network(next_states)
            batch_indices = paddle.arange(self.batch_size, dtype='int64')
            best_actions_indices = paddle.stack([batch_indices, best_next_actions], axis=1)
            target_q_values_of_best_actions = paddle.gather_nd(next_q_values_target, best_actions_indices)

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

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_counter += 1

    def update_target_network_episode(self, episode):
        """根据 episode 频率硬更新目标网络"""
        if episode > 0 and episode % self.target_update_freq == 0:
            self.target_network.set_state_dict(self.main_network.state_dict())

    def update_target_network(self):
        """硬更新目标网络权重"""
        self.target_network.set_state_dict(self.main_network.state_dict())

    def get_q_value(self, state, action):
        """获取特定 state-action 对的 Q 值"""
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
        """保存主网络参数"""
        try:
            print(f"Saving model parameters to {filepath}...")
            paddle.save(self.main_network.state_dict(), filepath)
            print("Model parameters saved.")
        except Exception as e:
            print(f"Error saving model parameters: {e}")

    def load_model(self, filepath=MODEL_SAVE_PATH):
        """加载模型参数"""
        if os.path.exists(filepath):
            try:
                print(f"Loading model parameters from {filepath}...")
                state_dict = paddle.load(filepath)
                self.main_network.set_state_dict(state_dict)
                self.target_network.set_state_dict(state_dict)
                self.epsilon = self.epsilon_min 
                print("Model parameters loaded successfully.")
            except Exception as e:
                print(f"Error loading model parameters: {e}. Starting from scratch.")
        else:
            print(f"Model file {filepath} not found. Starting training from scratch.")

    def decay_epsilon(self):
        """衰减 Epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)