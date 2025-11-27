# train.py

import time
import numpy as np
import json
import paddle
import os
import matplotlib.pyplot as plt 

from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from data_logger import DataLogger
from config import (MAX_EPISODES, BATCH_SIZE, MODEL_SAVE_PATH, MODEL_SAVE_FREQ,
                    STATE_DIM, ACTION_DIM, EPSILON_DECAY, EPSILON_MIN, START_TRAIN_EPISODE, DATASET_FILE)

# --- 数据收集函数 ---
def save_llm_data(description, action, reward):
    """保存 (Instruction, Input, Output) 三元组用于大模型微调"""
    if reward > 0 and description:
        entry = {
            "instruction": "You are an intelligent scheduler for edge computing. Assign the microservice to the best node.",
            "input": description,
            "output": str(action),
            "reward": reward
        }
        try:
            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving dataset: {e}")

def plot_training_results(rewards, losses, q_values):
    """绘制训练过程的收敛情况"""
    plt.figure(figsize=(15, 10))

    # 子图 1: Episode Returns (Rewards)
    plt.subplot(3, 1, 1)
    plt.plot(rewards, label='Episode Reward', color='blue', alpha=0.6)
    # 绘制移动平均线 (Window=50)
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), moving_avg, label='Moving Avg (50)', color='red', linewidth=2)
    plt.title('Convergence of Returns (Total Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)

    # 子图 2: Average Loss per Episode
    plt.subplot(3, 1, 2)
    plt.plot(losses, label='Avg Loss', color='orange')
    plt.title('Convergence of Loss Function')
    plt.xlabel('Episode')
    plt.ylabel('SmoothL1 Loss')
    plt.legend()
    plt.grid(True)

    # 子图 3: Average Q-Value per Episode
    plt.subplot(3, 1, 3)
    plt.plot(q_values, label='Avg Q-Value', color='green')
    plt.title('Convergence of Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_convergence.png')
    print("Training visualization saved to 'training_convergence.png'.")
    # plt.show() # 如果在无头服务器上运行，请保持注释

def train_agent():
    """主训练函数"""
    print("Initializing components for PaddlePaddle...")
    env = EnvironmentClient()
    agent = DuelingDQNAgent() 
    logger = DataLogger()

    agent.load_model() 

    # --- 训练指标记录 ---
    history_rewards = []
    history_losses = []
    history_q_values = []

    start_time = time.time()
    print(f"Starting PaddlePaddle training for {MAX_EPISODES} episodes...")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            state, mask = env.reset()
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
                
                # 死局检测
                if not np.any(mask):
                    print(f"Episode {episode}: Step {step}: No valid actions (Dead End).")
                    final_reward = env.get_final_reward()
                    if len(agent.memory) > 0:
                        agent.memory.update_last_reward(final_reward)
                    total_reward += final_reward
                    break

                # 动作选择
                if episode < START_TRAIN_EPISODE:
                    valid_actions = np.where(mask)[0]
                    action = np.random.choice(valid_actions) if valid_actions.size > 0 else 0
                else:
                    action = agent.select_action(state, mask, explore=True)

                # 环境步进
                next_state, next_mask, reward, done, info = env.step(action)

                if next_state is None or next_mask is None:
                    print(f"Episode {episode}, Step {step}: Env step failed. Ending episode.")
                    break

                # 存储经验
                agent.remember(state, action, reward, next_state, done, mask, next_mask)
                
                # 记录 Q 值 (用于分析收敛性)
                q_value = agent.get_q_value(state, action)
                episode_q_values.append(q_value)
                logger.log(state, action, q_value, episode, step)
                
                total_reward += reward

                # 收集大模型数据
                if episode >= START_TRAIN_EPISODE and "description" in info:
                    save_llm_data(info["description"], action, reward)

                if done:
                    if np.any(mask): 
                        final_reward = env.get_final_reward()
                        total_reward += final_reward
                        agent.memory.update_last_reward(final_reward)
                    break

                state = next_state
                mask = next_mask

                # --- 训练网络并捕获 Loss ---
                if episode >= START_TRAIN_EPISODE and len(agent.memory) >= BATCH_SIZE:
                    loss_val = agent.train()
                    if loss_val is not None:
                        episode_losses.append(loss_val)
                
                if step > (ACTION_DIM * 5): 
                    print(f"Warning: Episode {episode} exceeded max step limit.")
                    break

            # Episode 结束后的处理
            agent.decay_epsilon()
            agent.update_target_network_episode(episode)
            
            # 记录本局统计数据
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0.0
            
            history_rewards.append(total_reward)
            history_losses.append(avg_loss)
            history_q_values.append(avg_q)

            episode_duration = time.time() - episode_start_time
            if episode % 10 == 0:
                print(f"Ep {episode}: Reward={total_reward:.2f} | Avg Loss={avg_loss:.4f} | Avg Q={avg_q:.4f} | Epsilon={agent.epsilon:.4f} | Time={episode_duration:.1f}s")

            if episode % MODEL_SAVE_FREQ == 0:
                agent.save_model()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_training_time = time.time() - start_time
        print(f"\nTraining finished. Total time: {total_training_time:.2f}s")
        print("Saving final model...")
        agent.save_model()
        
        # --- 绘制并保存收敛图 ---
        print("Generating convergence plots...")
        plot_training_results(history_rewards, history_losses, history_q_values)
        
        try:
            env.stop_server()
        except:
            pass

if __name__ == "__main__":
    print("PaddlePaddle using device:", paddle.get_device())
    train_agent()