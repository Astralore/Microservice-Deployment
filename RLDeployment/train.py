# train.py

import time
import numpy as np
import json
import paddle
import os
import matplotlib.pyplot as plt 
from datetime import datetime

from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from data_logger import DataLogger
from config import (MAX_EPISODES, BATCH_SIZE, MODEL_SAVE_FREQ,
                    STATE_DIM, ACTION_DIM, EPSILON_DECAY, EPSILON_MIN, START_TRAIN_EPISODE)

class ExperimentManager:
    """管理实验路径和文件保存"""
    def __init__(self, base_dir="D:/Code/MD_DATA/experiments"):
        # 生成时间戳文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, timestamp)
        
        # 创建文件夹
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        
        print(f"=== Experiment Output Directory: {self.exp_dir} ===")

        # 定义各文件的完整路径
        self.log_file = os.path.join(self.exp_dir, 'rl_training_log.csv')
        self.dataset_file = os.path.join(self.exp_dir, 'llm_finetuning_data.jsonl')
        self.model_file = os.path.join(self.exp_dir, 'dueling_dqn_model.pdparams')
        self.plot_file = os.path.join(self.exp_dir, 'training_convergence.png')

    def save_llm_data(self, description, action, reward):
        """保存高质量的 (Instruction, Input, Output) 三元组"""
        # [修改] 阈值设为 35 (边缘部署的平均得分线)
        if reward > 35.0 and description:
            entry = {
                "instruction": "You are an intelligent scheduler for edge computing. Given the system state and resource requirements, select the optimal node ID for microservice deployment. Prioritize edge nodes with sufficient resources to minimize latency.",
                "input": description, # 这是决策前的环境快照
                "output": str(action),
                "reward": round(reward, 2)
            }
            try:
                with open(self.dataset_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error saving dataset: {e}")

    def plot_results(self, rewards, losses, q_values):
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(rewards, label='Episode Reward', color='blue', alpha=0.6)
        if len(rewards) >= 50:
            moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(rewards)), moving_avg, label='Moving Avg (50)', color='red', linewidth=2)
        plt.title('Convergence of Returns (Total Reward)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend(); plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(losses, label='Avg Loss', color='orange')
        plt.title('Convergence of Loss Function')
        plt.xlabel('Episode')
        plt.ylabel('SmoothL1 Loss')
        plt.legend(); plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(q_values, label='Avg Q-Value', color='green')
        plt.title('Convergence of Q-Values')
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.plot_file)
        print(f"Training visualization saved to '{self.plot_file}'.")

def train_agent():
    exp_manager = ExperimentManager()
    print("Initializing components for PaddlePaddle...")
    env = EnvironmentClient()
    agent = DuelingDQNAgent() 
    logger = DataLogger(filename=exp_manager.log_file)

    history_rewards = []
    history_losses = []
    history_q_values = []

    start_time = time.time()
    print(f"Starting PaddlePaddle training for {MAX_EPISODES} episodes...")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            # 1. 重置环境，获取初始状态和初始描述
            state, mask, info = env.reset()
            current_desc = info.get('description', "") # [关键] 缓存当前描述
            
            # --- 动态环境模拟 ---
            if episode < MAX_EPISODES: 
                valid_indices = np.where(mask)[0]
                candidates = valid_indices[2:]
                if len(candidates) > 5:
                    num_drop = np.random.randint(0, 6)
                    if num_drop > 0:
                        drop_indices = np.random.choice(candidates, num_drop, replace=False)
                        mask[drop_indices] = False
                        for idx in drop_indices:
                            state[idx*4:(idx+1)*4] = 0.0
            # -------------------
            
            if state is None or mask is None:
                print(f"Episode {episode}: Failed reset. Stopping.")
                break

            total_reward = 0
            step = 0
            episode_losses = []
            episode_q_values = []
            episode_start_time = time.time()

            while True:
                step += 1
                
                if not np.any(mask):
                    final_reward = env.get_final_reward()
                    if len(agent.memory) > 0:
                        agent.memory.update_last_reward(final_reward)
                    total_reward += final_reward
                    break

                if episode < START_TRAIN_EPISODE:
                    valid_actions = np.where(mask)[0]
                    action = np.random.choice(valid_actions) if valid_actions.size > 0 else 0
                else:
                    action = agent.select_action(state, mask, explore=True)

                # 2. 执行动作
                next_state, next_mask, reward, done, next_info = env.step(action)

                if next_state is None or next_mask is None:
                    break

                # 3. 存储经验
                agent.remember(state, action, reward, next_state, done, mask, next_mask)
                
                # 记录 Q 值
                q_value = agent.get_q_value(state, action)
                episode_q_values.append(q_value)
                logger.log(state, action, q_value, episode, step)
                
                total_reward += reward

                # 4. [关键] 保存微调数据
                # 使用 current_desc (决策前的环境) 和 当前的 action/reward
                if episode >= START_TRAIN_EPISODE and current_desc:
                    exp_manager.save_llm_data(current_desc, action, reward)

                if done:
                    if np.any(mask): 
                        final_reward = env.get_final_reward()
                        total_reward += final_reward
                        agent.memory.update_last_reward(final_reward)
                    break

                # 5. 状态流转
                state = next_state
                mask = next_mask
                # 更新描述，用于下一步的记录
                current_desc = next_info.get('description', "") 

                # 训练网络
                if episode >= START_TRAIN_EPISODE and len(agent.memory) >= BATCH_SIZE:
                    loss_val = agent.train()
                    if loss_val is not None:
                        episode_losses.append(loss_val)
                
                if step > (ACTION_DIM * 5): 
                    break

            # ... (后续处理逻辑不变)
            agent.decay_epsilon()
            agent.update_target_network_episode(episode)
            
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0.0
            
            history_rewards.append(total_reward)
            history_losses.append(avg_loss)
            history_q_values.append(avg_q)

            episode_duration = time.time() - episode_start_time
            if episode % 10 == 0:
                print(f"Ep {episode}: Reward={total_reward:.2f} | Avg Loss={avg_loss:.4f} | Avg Q={avg_q:.4f} | Epsilon={agent.epsilon:.4f} | Time={episode_duration:.1f}s")

            if episode % MODEL_SAVE_FREQ == 0:
                agent.save_model(filepath=exp_manager.model_file)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_training_time = time.time() - start_time
        print(f"\nTraining finished. Total time: {total_training_time:.2f}s")
        print(f"Saving final model to {exp_manager.model_file}...")
        agent.save_model(filepath=exp_manager.model_file)
        
        print("Generating convergence plots...")
        exp_manager.plot_results(history_rewards, history_losses, history_q_values)
        
        try:
            env.stop_server()
        except:
            pass

if __name__ == "__main__":
    print("PaddlePaddle using device:", paddle.get_device())
    train_agent()