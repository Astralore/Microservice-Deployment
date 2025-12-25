# train.py

import time
import numpy as np
import json
import paddle
import os
import matplotlib.pyplot as plt 
from datetime import datetime
from llm_direct import LLMClient  
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
        self.raw_data_file = os.path.join(self.exp_dir, 'raw_training_metadata.json')
        self.model_file = os.path.join(self.exp_dir, 'dueling_dqn_model.pdparams')
        self.plot_file = os.path.join(self.exp_dir, 'training_convergence.png')

    
    def save_raw_metadata(self, description, node_ids_map, q_values, mask, action, reward):
        """
        [高效版] 只保存元数据，不进行文本处理和排序。
        description: Java 发来的环境描述字符串
        node_ids_map: ID 映射表
        q_values: RL 预测的全量 Q 值 (Numpy array)
        mask: 有效性掩码
        """
        # 构造轻量级 Entry
        entry = {
            "desc": description,         # 环境 Context
            "node_ids": node_ids_map,    # 翻译字典
            "q_vals": q_values.tolist(), # RL 的大脑状态 (关键!)
            "mask": mask.tolist(),       # 约束条件
            "act": int(action),          # RL 实际选择的动作
            "rew": round(reward, 4),     # 分数
            "ts": time.time()            # 时间戳
        }

        try:
            with open(self.raw_data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving raw data: {e}")
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

    llm_client = LLMClient(api_url="http://127.0.0.1:6006/predict")
    
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
                
                # 如果所有节点都不可用 (物理约束)
                if not np.any(mask):
                    final_reward = env.get_final_reward()
                    if len(agent.memory) > 0:
                        agent.memory.update_last_reward(final_reward)
                    total_reward += final_reward
                    break
        
                # 1. 默认使用物理掩码 (Resource Mask)
                # mask 是 Java 端传来的，已经把 CPU/RAM 不足的节点设为 False 了
                final_mask = mask.copy()
                
                # 2. 调用 LLM 进行剪枝
                # 建议：始终开启，或者在训练初期开启。这里设为 True
                use_llm = True 
                
                if use_llm and current_desc:
                    # A. 发送请求给云端
                    suggested_phy_ids = llm_client.get_suggestions(current_desc)
                    
                    if len(suggested_phy_ids) > 0:
                        # B. 构建 LLM 掩码
                        llm_mask = np.zeros(ACTION_DIM, dtype=bool)
                        
                        # 获取 ID 映射表 (Action Index -> Physical Node ID)
                        current_node_mapping = info.get('node_ids', [])
                        
                        match_count = 0
                        for action_idx, phy_id in enumerate(current_node_mapping):
                            # 如果该物理节点在 LLM 的推荐列表中
                            if phy_id in suggested_phy_ids:
                                llm_mask[action_idx] = True
                                match_count += 1
                        
                        # C. 融合掩码 (Intersection)
                        # 逻辑：必须 "物理资源足够" 且 "LLM 推荐"
                        combined_mask = mask & llm_mask
                        
                        # D. 兜底机制 (Safety Net)
                        # 如果交集不为空，则生效；
                        # 如果交集为空 (说明 LLM 推荐的节点正好都没资源了)，则回退到 final_mask (即物理 mask)
                        if np.any(combined_mask):
                            final_mask = combined_mask
                            # print(f"  [LLM Pruning] Scope reduced: {np.sum(mask)} -> {np.sum(final_mask)}")
                        else:
                            # print(f"  [LLM Warning] All suggested nodes are resource-constrained. Fallback to full mask.")
                            pass
                
                if episode < START_TRAIN_EPISODE:
                    # 随机预热 (只在 final_mask 允许的范围内随机)
                    valid_indices = np.where(final_mask)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        action = 0 
                else:
                    # RL 决策 
                    # select_action 会自动把 final_mask 为 False 的动作 Q 值设为 -inf
                    # 从而迫使 Agent 只在 LLM 推荐的节点中选最优
                    action = agent.select_action(state, final_mask, explore=True)
                
                resource_monitor.record(state, action, final_mask)
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
                print(f"\rEp {episode} Step {step} | LLM Responded | Action: {action} | Reward: {reward:.2f}", end="")
                if next_state is None or next_mask is None:
                    break

                # 存储经验 (State, Action, Reward, NextState, Done, Mask, NextMask)
                # 注意：这里存的是物理 mask 还是 final_mask 取决于你想让 Agent 学什么
                # 通常存 final_mask 可以让 Off-policy 学习意识到约束
                agent.remember(state, action, reward, next_state, done, final_mask, next_mask)
                
                # 记录 Q 值
                q_value = agent.get_q_value(state, action)
                episode_q_values.append(q_value)
                logger.log(state, action, q_value, episode, step)
                
                total_reward += reward
                
                # --- 数据收集 (用于未来微调) ---
                if episode >= START_TRAIN_EPISODE and current_desc:
                    should_save = False
                    base_idx = action * 3
                    is_cloud = (base_idx + 1 < len(state)) and (state[base_idx + 1] > 0.9)

                    if is_cloud and reward > -15.0: should_save = True
                    elif not is_cloud and reward > 20.0: should_save = True
                    
                    if should_save:
                        state_tensor = paddle.to_tensor(state, dtype='float32').unsqueeze(0)
                        with paddle.no_grad():
                            all_q_values = agent.main_network(state_tensor).numpy()[0]
                        node_ids_map = info.get('node_ids', [])
                        
                        exp_manager.save_raw_metadata(
                            current_desc, 
                            node_ids_map, 
                            all_q_values, 
                            final_mask, # 记录实际使用的掩码
                            action, 
                            reward
                        )

                if done:
                    if np.any(mask): 
                        final_reward = env.get_final_reward()
                        total_reward += final_reward
                        agent.memory.update_last_reward(final_reward)
                    break

                state = next_state
                mask = next_mask
                current_desc = next_info.get('description', "") 
                TRAIN_FREQUENCY = 10
                # 训练网络
                if episode >= START_TRAIN_EPISODE and len(agent.memory) >= BATCH_SIZE:
                    if step % TRAIN_FREQUENCY == 0:  
                        loss_val = agent.train()
                        if loss_val is not None: episode_losses.append(loss_val)
                    # loss_val = agent.train()
                    # if loss_val is not None:
                    #     episode_losses.append(loss_val)
                
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