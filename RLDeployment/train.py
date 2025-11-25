# train.py

import time
import numpy as np
import json
import paddle
import os
# 如果没有 matplotlib 可以注释掉绘图部分，或者 pip install matplotlib
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

def train_agent():
    """主训练函数"""
    print("Initializing components for PaddlePaddle...")
    env = EnvironmentClient()
    agent = DuelingDQNAgent() 
    logger = DataLogger()

    agent.load_model() 

    episode_rewards = []
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

                # --- [核心修复点] 接收 5 个返回值 ---
                next_state, next_mask, reward, done, info = env.step(action)

                # 异常处理
                if next_state is None or next_mask is None:
                    print(f"Episode {episode}, Step {step}: Env step failed. Ending episode.")
                    break

                # 存储经验
                agent.remember(state, action, reward, next_state, done, mask, next_mask)
                
                q_value = agent.get_q_value(state, action)
                logger.log(state, action, q_value, episode, step)
                total_reward += reward

                # --- 收集大模型数据 ---
                if episode >= START_TRAIN_EPISODE:
                    if "description" in info:
                        save_llm_data(info["description"], action, reward)

                if done:
                    if np.any(mask): 
                        final_reward = env.get_final_reward()
                        total_reward += final_reward
                        agent.memory.update_last_reward(final_reward)
                    
                    episode_duration = time.time() - episode_start_time
                    if episode % 10 == 0:
                        print(f"Episode {episode}: Steps={step}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.4f}, Duration={episode_duration:.2f}s")
                    break

                state = next_state
                mask = next_mask

                if episode >= START_TRAIN_EPISODE and len(agent.memory) >= BATCH_SIZE:
                    agent.train()
                
                if step > (ACTION_DIM * 5): 
                    print(f"Warning: Episode {episode} exceeded max step limit.")
                    break

            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            agent.update_target_network_episode(episode)

            if episode % MODEL_SAVE_FREQ == 0:
                agent.save_model()

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_training_time = time.time() - start_time
        print(f"\nTraining finished/stopped after {len(episode_rewards)} episodes.")
        print(f"Total training time: {total_training_time:.2f} seconds.")
        print("Saving final model...")
        agent.save_model()
        try:
            env.stop_server()
        except:
            pass

if __name__ == "__main__":
    print("PaddlePaddle using device:", paddle.get_device())
    train_agent()