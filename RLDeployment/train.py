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

    def save_llm_data(self, description, action, reward):
        # 阈值保持 30 或 35 均可，这里用 30 保证收集足够数据
        if reward > 35.0 and description:
            entry = {
                "instruction": "You are an intelligent scheduler for edge computing. Given the system state and resource requirements, select the optimal node ID for microservice deployment. Prioritize edge nodes with sufficient resources to minimize latency.",
                "input": description,
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

    resource_monitor = ResourceMonitor()
    load_monitor = LoadBalanceMonitor(ACTION_DIM)

    history_rewards = []
    history_losses = []
    history_q_values = []

    start_time = time.time()
    print(f"Starting PaddlePaddle training for {MAX_EPISODES} episodes...")

    tracking_edge_count = 0
    tracking_total_steps = 0
    tracking_cpu_margins = []

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
                
                        # # [插入这段 Debug 代码] ==================================
                        # if step == 1 and episode % 50 == 0 and node_id == 5:
                        #     print(f"\n>>> [PYTHON DEBUG Node 5] <<<")
                        #     # 打印这一段 State 的原始 4 个数值
                        #     raw_slice = state[base_idx : base_idx+4]
                        #     print(f"Raw Slice (Indices {base_idx}-{base_idx+3}): {raw_slice}")
                            
                        #     # 打印我们认为的变量
                        #     print(f"Read CPU_SAFE (Idx {base_idx+0}): {state[base_idx+0]}")
                        #     print(f"Read RAM_SAFE (Idx {base_idx+1}): {state[base_idx+1]}")
                        #     print(f"Read LEVEL    (Idx {base_idx+3}): {state[base_idx+3]}")
                        #     print("===============================\n")
                        # # ======================================================
                expert_decay_steps = 10000
                expert_prob = max(0.0, 1.0 - (episode / expert_decay_steps))
                
                forced_action = -1   
                use_expert = np.random.rand() < expert_prob

                if use_expert:
                    valid_edge_candidates = []
                    weighted_candidates = []
                    valid_cloud_candidates = []  
                    debug_logs = [] 
                    
                    
                    # --- 专家打分循环 ---
                    for node_id in range(ACTION_DIM):
                        if not mask[node_id]: continue

                        # [关键修改 1] 步长改为 7 (匹配 Java 端 7 个特征)
                        base_idx = node_id * 7
                        if base_idx + 6 >= len(state): continue
                        
                        # [关键修改 2] 读取所有特征
                        # Java顺序: [cpuMar, ramMar, cpuCap, ramCap, level, util, count]
                        cpu_margin = state[base_idx + 0] 
                        ram_margin = state[base_idx + 1]
                        cpu_capacity = state[base_idx + 2] 
                        ram_capacity = state[base_idx + 3]
                        level      = state[base_idx + 4]
                        curr_util  = state[base_idx + 5] # 当前利用率
                        dep_count  = state[base_idx + 6] # 部署计数

                        # 资源判断 (保持与 Java Mask 一致)
                        is_resource_enough = (cpu_margin >= -0.05) and (ram_margin >= -0.05)

                        if is_resource_enough:
                            if level > 0.5: # Edge
                                # === [智能专家策略：能效 + 负载 + 亲和] ===
                                # 优先选小容量节点(cpu_capacity小)，把大节点留给大任务，且小节点省电。
                                score_efficiency = (1.0 - cpu_capacity) * 0.5 
                                # 优先选空闲节点(curr_util小)
                                score_lb = (1.0 - curr_util) * 0.3
                                # 优先选已有服务的节点(dep_count大)，提高局部性
                                score_collo = dep_count * 0.3
                                
                                final_score = score_efficiency + score_lb + score_collo
                                weighted_candidates.append((node_id, final_score))
                                
                                # 日志记录
                                if step == 1 and episode % 50 == 0 and len(debug_logs) < 5:
                                    debug_logs.append(f"E{node_id}(Eff:{score_efficiency:.2f}|Sc:{final_score:.2f})")
                                    
                            elif level < 0.2: # Cloud
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
                        # 从前 50% 的好节点里随机选 (保证多样性)
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
                    # 随机填充时，也要遵守 Mask，且尽量避开 Padding
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        action = 0 # 兜底 Cloud
                else:
                    # 正常的 RL 策略 (Epsilon-Greedy)
                    action = agent.select_action(state, mask, explore=True)
                 # === 调用监控器记录数据 ===
                resource_monitor.record(state, action, mask)
                load_monitor.update(action)
                # 记录当前步的决策数据
                tracking_total_steps += 1
                base_idx = action * 7  # 确保这里系数是 5
                
                # 安全检查防止越界
                if base_idx + 6 < len(state):
                    # Index 0 是 CPU Margin, Index 4 是 Level
                    cpu_margin = state[base_idx + 0]
                    level = state[base_idx + 4]
                    
                    if level > 0.5: # 判定为边缘节点
                        tracking_edge_count += 1
                    
                    tracking_cpu_margins.append(cpu_margin)
                # 执行动作
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
        
            if episode % 100 == 0:
                resource_monitor.print_summary(episode)
                # 重置监控器
                resource_monitor = ResourceMonitor()
            if episode % 100 == 0:
                gini = load_monitor.get_gini_coefficient()
                variance = load_monitor.get_utilization_variance()
                print(f"负载均衡指标 - 基尼系数: {gini:.3f}, 方差: {variance:.3f}")
                load_monitor = LoadBalanceMonitor(ACTION_DIM)  # 重置
            agent.decay_epsilon()
            agent.update_target_network_episode(episode)
            
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0.0
            
            history_rewards.append(total_reward)
            history_losses.append(avg_loss)
            history_q_values.append(avg_q)

            episode_duration = time.time() - episode_start_time
            if episode % 10 == 0:
                # 计算并打印统计信息
                edge_rate = tracking_edge_count / tracking_total_steps if tracking_total_steps > 0 else 0
                avg_margin = np.mean(tracking_cpu_margins) if tracking_cpu_margins else 0
                
                # 修改原有的 print，或者在下面加一行
                print(f"Ep {episode}: Reward={total_reward:.2f} ... [Edge率={edge_rate:.1%} | AvgMargin={avg_margin:.3f}]")

                # 打印完必须重置，为下一个 10 轮做准备
                tracking_edge_count = 0
                tracking_total_steps = 0
                tracking_cpu_margins = []

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
        self.node_loads[action] += 1
        self.decision_count += 1
    
    def get_gini_coefficient(self):
        """计算基尼系数（负载不均衡度）"""
        if self.decision_count == 0:
            return 0
        loads = self.node_loads / self.decision_count
        loads = np.sort(loads)
        n = len(loads)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * loads)) / (n * np.sum(loads))
        return gini
    
    def get_utilization_variance(self):
        """计算负载方差"""
        loads = self.node_loads / self.decision_count if self.decision_count > 0 else self.node_loads
        return np.var(loads)
class ResourceMonitor:
    def __init__(self):
        self.cpu_margins = []
        self.ram_margins = []
        self.decisions = []  # 0=云, 1=边缘
    
    def record(self, state, action, mask):
        """记录节点资源余量和决策"""
        base_idx = action * 7
        if base_idx + 6 < len(state):
            cpu_margin = state[base_idx]
            ram_margin = state[base_idx + 1]
            level = state[base_idx + 4]
            
            self.cpu_margins.append(cpu_margin)
            self.ram_margins.append(ram_margin)
            self.decisions.append(1 if level > 0.5 else 0)
    
    def print_summary(self, episode):
        if self.cpu_margins:
            avg_cpu = np.mean(self.cpu_margins)
            avg_ram = np.mean(self.ram_margins)
            edge_ratio = np.mean(self.decisions)
            print(f"Ep{episode} 资源余量: CPU={avg_cpu:.3f}, RAM={avg_ram:.3f}, 边缘选择率={edge_ratio:.1%}")

if __name__ == "__main__":
    print("PaddlePaddle using device:", paddle.get_device())
    train_agent()