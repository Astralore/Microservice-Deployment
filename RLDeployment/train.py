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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, timestamp)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        print(f"=== Experiment Output Directory: {self.exp_dir} ===")
        self.log_file = os.path.join(self.exp_dir, 'rl_training_log.csv')
        self.dataset_file = os.path.join(self.exp_dir, 'llm_finetuning_data.jsonl')
        self.model_file = os.path.join(self.exp_dir, 'dueling_dqn_model.pdparams')
        self.plot_file = os.path.join(self.exp_dir, 'training_convergence.png')

    # def save_llm_data(self, description, action, reward):
    #     # 记录高质量的决策数据
    #     if reward > 35.0 and description:
    #         entry = {
    #             "instruction": "You are an intelligent scheduler for edge computing. Given the system state and resource requirements, select the optimal node ID for microservice deployment. Prioritize edge nodes with low latency and balanced load.",
    #             "input": description,
    #             "output": str(action),
    #             "reward": round(reward, 2)
    #         }
    #         try:
    #             with open(self.dataset_file, "a", encoding="utf-8") as f:
    #                 f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    #         except Exception as e:
    #             print(f"Error saving dataset: {e}")
    def save_llm_data(self, description, action, reward, instruction_type="Edge_Optimization"):
        """
        保存微调数据，支持不同的指令类型。
        """
        # 根据类型生成不同的 System Prompt
        if instruction_type == "Cloud_Fallback":
            # 云端场景：强调资源紧缺时的兜底策略
            instruction = "You are a robust system scheduler. The edge layer is currently overloaded or lacks resources for this task. Identify the appropriate fallback strategy (Cloud) to ensure service availability."
        else:
            # 边缘场景：强调性能优化
            instruction = "You are an intelligent scheduler for edge computing. Select the optimal edge node to minimize latency and balance load."

        entry = {
            "instruction": instruction,
            "input": description,
            "output": str(action),
            "reward": round(reward, 2),
            "type": instruction_type  # 额外记录类型，方便后续筛选
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

    resource_monitor = ResourceMonitor()
    load_monitor = LoadBalanceMonitor(ACTION_DIM)

    history_rewards = []
    history_losses = []
    history_q_values = []

    start_time = time.time()
    print(f"Starting PaddlePaddle training for {MAX_EPISODES} episodes...")

    tracking_edge_count = 0
    tracking_total_steps = 0
    tracking_margin_ratios = []

    try:
        for episode in range(1, MAX_EPISODES + 1):
            state, mask, info = env.reset()
            current_desc = info.get('description', "") 
            
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
                
                # 如果所有节点都不可用
                if not np.any(mask):
                    final_reward = env.get_final_reward()
                    if len(agent.memory) > 0:
                        agent.memory.update_last_reward(final_reward)
                    total_reward += final_reward
                    break
                
                # 专家策略衰减
                expert_decay_steps = 10000
                expert_prob = max(0.0, 1.0 - (episode / expert_decay_steps))
                
                forced_action = -1   
                use_expert = np.random.rand() < expert_prob

                if use_expert:
                    weighted_candidates = []
                    valid_cloud_candidates = []  
                    debug_logs = [] 
                    
                    # --- 专家打分循环 ---
                    for node_id in range(ACTION_DIM):
                        if not mask[node_id]: continue

                        # 步长改为 3 
                        base_idx = node_id * 3
                        if base_idx + 2 >= len(state): continue
                        
                        # 读取特征 [Load, Link, Margin]
                        load_pressure = state[base_idx + 0] 
                        link_cost     = state[base_idx + 1] 
                        margin_ratio  = state[base_idx + 2] 

                        # 判断节点类型 (Cloud 的 linkCost 通常是 1.0)
                        is_cloud = (link_cost > 0.9)

                        if not is_cloud: # Edge
                            # 1. 距离优先 (Locality): 权重 0.5
                            # 2. 负载均衡 (Load): 权重 0.3
                            # 3. 资源余量 (Margin): 权重 0.2
                            
                            score = (1.0 - link_cost) * 0.5 + \
                                    (1.0 - load_pressure) * 0.3 + \
                                    (margin_ratio) * 0.2
                            
                            weighted_candidates.append((node_id, score))
                            
                            # 日志记录
                            if step == 1 and episode % 50 == 0 and len(debug_logs) < 5:
                                debug_logs.append(f"E{node_id}(Ld:{load_pressure:.1f}|Lk:{link_cost:.1f}|Sc:{score:.2f})")
                                
                        else: # Cloud
                            valid_cloud_candidates.append(node_id)
                            if step == 1 and episode % 50 == 0 and len(debug_logs) < 5:
                                debug_logs.append(f"C{node_id}")

                    # [调试打印] 
                    if step == 1 and episode % 50 == 0:
                        print(f"\n[Expert Debug Ep{episode}] Candidates: {len(weighted_candidates)} | Clouds: {len(valid_cloud_candidates)}")
                        if debug_logs:
                            print(f"Sample Scores: {', '.join(debug_logs)}")

                    # [决策逻辑] - Top-K 随机采样
                    if len(weighted_candidates) > 0:
                        # 按分数排序
                        weighted_candidates.sort(key=lambda x: x[1], reverse=True)
                        # 从前 50% 的好节点里随机选
                        top_k = max(1, len(weighted_candidates) // 2)
                        choice_idx = np.random.randint(top_k)
                        forced_action = weighted_candidates[choice_idx][0]
                        
                    elif len(valid_cloud_candidates) > 0:
                        forced_action = np.random.choice(valid_cloud_candidates)
                        if step == 1 and episode % 50 == 0:
                            print("Expert Advice: All edges FULL, fallback to CLOUD.")
                
                # --- 动作选择与执行 ---
                if forced_action != -1:
                    action = forced_action
                elif episode < START_TRAIN_EPISODE:
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        action = 0 
                else:
                    action = agent.select_action(state, mask, explore=True)
                
                # --- 记录数据 ---
                resource_monitor.record(state, action, mask)
                load_monitor.update(action)
                
                tracking_total_steps += 1
                base_idx = action * 3 
                
                if base_idx + 2 < len(state):
                    link_cost = state[base_idx + 1]
                    margin_ratio = state[base_idx + 2]
                    
                    # 判定为边缘节点: LinkCost < 0.9 (Cloud=1.0)
                    if link_cost < 0.9: 
                        tracking_edge_count += 1
                    
                    tracking_margin_ratios.append(margin_ratio)

                # Env Step
                next_state, next_mask, reward, done, next_info = env.step(action)

                if next_state is None or next_mask is None:
                    break

                # 存储经验
                agent.remember(state, action, reward, next_state, done, mask, next_mask)
                
                # 记录 Q 值
                q_value = agent.get_q_value(state, action)
                episode_q_values.append(q_value)
                logger.log(state, action, q_value, episode, step)
                
                total_reward += reward
                if episode >= START_TRAIN_EPISODE and current_desc:
                    # 1. 判断当前 Action 是否为云端节点
                    # 根据 Java 端特征定义：Feature 2 (Index 1) 是 LinkCost。Cloud 的 LinkCost 通常为 1.0 (或 > 0.9)
                    base_idx = action * 3
                    is_cloud_action = False
                    
                    # 确保索引不越界
                    if base_idx + 1 < len(state):
                        link_cost = state[base_idx + 1]
                        if link_cost > 0.9:
                            is_cloud_action = True
                    
                    # 2. 应用双重阈值
                    should_save = False
                    instruction_type = "Edge_Optimization"

                    if is_cloud_action:
                        # [策略 A] 云端兜底：标准放宽
                        # 只要不是失败(-50)，哪怕是 -5 分（正常云端分），也是正确的兜底决策
                        if reward > -10.0:
                            should_save = True
                            instruction_type = "Cloud_Fallback"
                    else:
                        # [策略 B] 边缘优化：标准严格
                        # 我们只希望 LLM 学习那些低延迟(+20)、低负载的优质边缘部署
                        # 30分通常意味着：Base(50) + Link(-10或+10) - LoadPenalty(如果有)
                        # 设置 > 30 可以过滤掉严重过载或链路极差的边缘节点
                        if reward > 30.0:
                            should_save = True
                            instruction_type = "Edge_Optimization"
                    
                    # 3. 执行保存
                    if should_save:
                        exp_manager.save_llm_data(current_desc, action, reward, instruction_type)

                # 保存微调数据
                if episode >= START_TRAIN_EPISODE and current_desc:
                    exp_manager.save_llm_data(current_desc, action, reward)

                if done:
                    if np.any(mask): 
                        final_reward = env.get_final_reward()
                        total_reward += final_reward
                        agent.memory.update_last_reward(final_reward)
                    break

                state = next_state
                mask = next_mask
                current_desc = next_info.get('description', "") 

                # 训练网络
                if episode >= START_TRAIN_EPISODE and len(agent.memory) >= BATCH_SIZE:
                    loss_val = agent.train()
                    if loss_val is not None:
                        episode_losses.append(loss_val)
                
                if step > (ACTION_DIM * 5): 
                    break
        
            # --- 周期统计 ---
            if episode % 100 == 0:
                resource_monitor.print_summary(episode)
                stats = load_monitor.get_load_distribution_stats()
                print(f"负载均衡 - 基尼系数: {stats['gini']:.3f}, 方差: {stats['variance']:.6f}")
                
                # 重置监控
                resource_monitor = ResourceMonitor()
                load_monitor = LoadBalanceMonitor(ACTION_DIM)
                
            agent.decay_epsilon()
            agent.update_target_network_episode(episode)
            
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0.0
            
            history_rewards.append(total_reward)
            history_losses.append(avg_loss)
            history_q_values.append(avg_q)

            # 实时进度
            if episode % 10 == 0:
                edge_rate = tracking_edge_count / tracking_total_steps if tracking_total_steps > 0 else 0
                avg_margin = np.mean(tracking_margin_ratios) if tracking_margin_ratios else 0
                
                print(f"Ep {episode}: Reward={total_reward:.2f} ... [Edge率={edge_rate:.1%} | AvgMarginRatio={avg_margin:.3f}]")

                tracking_edge_count = 0
                tracking_total_steps = 0
                tracking_margin_ratios = []

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

class LoadBalanceMonitor:
    def __init__(self, num_nodes):
        self.node_loads = np.zeros(num_nodes)
        self.decision_count = 0
    
    def update(self, action):
        """更新负载统计"""
        self.node_loads[action] += 1
        self.decision_count += 1
    
    def get_load_distribution_stats(self):
        """获取负载分布统计"""
        if self.decision_count == 0:
            return {'gini': 0, 'variance': 0}
        
        probabilities = self.node_loads / self.decision_count
        active_probs = probabilities[probabilities > 0]
        
        if len(active_probs) == 0:
            return {'gini': 0, 'variance': 0}
        
        # 计算基尼系数
        prob_sorted = np.sort(active_probs)
        n = len(prob_sorted)
        index = np.arange(1, n + 1)
        numerator = np.sum((2 * index - n - 1) * prob_sorted)
        denominator = n * np.sum(prob_sorted)
        gini = numerator / denominator if denominator != 0 else 0
        
        return {
            'gini': gini,
            'variance': np.var(active_probs)
        }

class ResourceMonitor:
    def __init__(self):
        self.margins = []
        self.edge_decisions = [] 
    
    def record(self, state, action, mask):
        """记录资源余量和决策"""
        base_idx = action * 3 
        if base_idx + 2 < len(state):
            link_cost = state[base_idx + 1]
            margin_ratio = state[base_idx + 2]
            
            self.margins.append(margin_ratio)
            # LinkCost < 0.9 认为是 Edge
            self.edge_decisions.append(1 if link_cost < 0.9 else 0)
    
    def print_summary(self, episode):
        if self.margins:
            avg_margin = np.mean(self.margins)
            edge_ratio = np.mean(self.edge_decisions)
            print(f"Ep{episode} 统计: AvgMarginRatio={avg_margin:.3f}, 边缘选择率={edge_ratio:.1%}")

if __name__ == "__main__":
    print("PaddlePaddle using device:", paddle.get_device())
    train_agent()