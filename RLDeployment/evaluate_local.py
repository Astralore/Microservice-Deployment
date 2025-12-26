# evaluate_local.py
import time
import numpy as np
import paddle
import os
import csv
from datetime import datetime
from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from llm_direct import DirectLLM 
from config import ACTION_DIM 


RL_MODEL_PATH = "D:/Code/MD_DATA/experiments/20251215_133758/dueling_dqn_model.pdparams"

ENABLE_LLM = False  

# [核心参数] 训练时的物理节点数
TRAIN_NODE_COUNT = 49 
TRAIN_MAX_DIM = 50

# 计算旧模型的维度: (50 * 3) + 2 = 152
OLD_STATE_DIM = (TRAIN_MAX_DIM * 3) + 2
OLD_ACTION_DIM = TRAIN_MAX_DIM

class ExperimentRecorder:
    def __init__(self, mode_name):
        self.mode_name = mode_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file = os.path.join("logs", f"detail_{mode_name}_{self.timestamp}.csv")
        if not os.path.exists("logs"): os.makedirs("logs")
        with open(self.detail_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Reward', 'Action_Index', 'Node_ID', 'Link_Cost', 'Decision_Type'])
        self.rewards = []
        self.actions = []

    def record_step(self, step, reward, action_idx, node_id, raw_state, decision_type):
        self.rewards.append(reward)
        self.actions.append(action_idx)
        
        link_cost = 0.0
        # raw_state 是长向量，按 3 维步长取 LinkCost (Index 1)
        if action_idx < len(raw_state) // 3:
            link_cost = raw_state[action_idx * 3 + 1]
            
        with open(self.detail_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 记录 Action Index 和 Node ID 方便排查
            writer.writerow([step, f"{reward:.2f}", action_idx, node_id, f"{link_cost:.2f}", decision_type])

    def save_summary(self, energy, makespan):
        avg_reward = np.mean(self.rewards)
        cloud_rate = (self.actions.count(0) / len(self.actions)) * 100
        new_node_rate = (sum(1 for a in self.actions if a >= TRAIN_NODE_COUNT) / len(self.actions)) * 100
        
        summary_file = "experiment_summary.csv"
        file_exists = os.path.isfile(summary_file)
        with open(summary_file, mode='a', newline='') as f:
            headers = ['Timestamp', 'Mode', 'Avg_Reward', 'Total_Energy_kJ', 'Makespan_sec', 'Cloud_Rate%', 'New_Node_Rate%', 'Detail_File']
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists: writer.writeheader()
            writer.writerow({
                'Timestamp': self.timestamp, 'Mode': self.mode_name,
                'Avg_Reward': f"{avg_reward:.2f}", 'Total_Energy_kJ': f"{energy:.4f}",
                'Makespan_sec': f"{makespan:.2f}", 'Cloud_Rate%': f"{cloud_rate:.1f}",
                'New_Node_Rate%': f"{new_node_rate:.1f}", 'Detail_File': self.detail_file
            })
        print(f"\n✅ 数据已汇总至: {summary_file}")

class SafeEnvironmentClient(EnvironmentClient):
    def parse_state_adaptive(self, raw_state_vec, raw_mask):
        current_nodes = len(raw_mask)
        target_nodes = TRAIN_NODE_COUNT 
        global_feats = raw_state_vec[-2:] 
        node_feats = raw_state_vec[:-2]   
        
        if current_nodes >= target_nodes:
            sliced_feats = node_feats[:target_nodes * 3]
            sliced_mask = raw_mask[:target_nodes]
            # 补 Padding 到 50
            pad_len = TRAIN_MAX_DIM - target_nodes
            if pad_len > 0:
                padding_feat = np.array([1.0, 1.0, -1.0] * pad_len, dtype=np.float32)
                padding_mask = np.zeros(pad_len, dtype=bool)
                s_out = np.concatenate([sliced_feats, padding_feat, global_feats])
                m_out = np.concatenate([sliced_mask, padding_mask])
            else:
                s_out = np.concatenate([sliced_feats, global_feats])
                m_out = sliced_mask
        else:
            pad_len = TRAIN_MAX_DIM - current_nodes
            padding_feat = np.array([1.0, 1.0, -1.0] * pad_len, dtype=np.float32)
            s_out = np.concatenate([node_feats, padding_feat, global_feats])
            padding_mask = np.zeros(pad_len, dtype=bool)
            m_out = np.concatenate([raw_mask, padding_mask])
            
        return s_out.astype('float32'), m_out.astype('bool')

def evaluate():
    mode_name = "Ours_LLM" if ENABLE_LLM else "Baseline_RL"
    print(f" 启动自动化实验: {mode_name}")
    print(f" (Agent View: {TRAIN_NODE_COUNT} Nodes | Global View: Configured MAX Nodes)")
    
    env = SafeEnvironmentClient()
    recorder = ExperimentRecorder(mode_name)
    agent = DuelingDQNAgent(state_dim=OLD_STATE_DIM, action_dim=OLD_ACTION_DIM)
    
    if os.path.exists(RL_MODEL_PATH):
        try: 
            agent.main_network.set_state_dict(paddle.load(RL_MODEL_PATH))
            print(" RL Model Loaded Successfully")
        except Exception as e: 
            print(f" Model Load Failed: {e}")
    else:
        print(" No model file found! (Check RL_MODEL_PATH)")

    llm = DirectLLM() 

    # Episode 循环
    raw_state, raw_mask, info = env.reset()
    state_for_rl, mask_for_rl = env.parse_state_adaptive(raw_state, raw_mask)
    current_desc = info.get('description', "")
    
    # 建立 ID 映射表 (用于日志记录和 LLM 转换)
    real_ids = info.get('node_ids', [])
    id_to_idx = {uid: i for i, uid in enumerate(real_ids)}
    
    step = 0
    
    while True:
        step += 1
        
        # --- 1. RL Decision ---
        state_tensor = paddle.to_tensor(state_for_rl, dtype='float32').unsqueeze(0)
        with paddle.no_grad(): 
            q_values = agent.main_network(state_tensor).numpy()[0]
        
        valid_q = np.where(mask_for_rl, q_values, -1e9)
        rl_act_idx = np.argmax(valid_q) 
        
        if rl_act_idx >= TRAIN_NODE_COUNT:
            rl_act_idx = 0 
            
        final_act_idx = rl_act_idx
        dec_type = "RL"

        # --- 2. LLM + Arbiter ---
        if ENABLE_LLM and current_desc:
            sug_ids = llm.get_suggestions(current_desc) # LLM 返回 ID
            
            # ID 转 Index
            valid_sug_indices = []
            for uid in sug_ids:
                if uid in id_to_idx:
                    idx = id_to_idx[uid]
                    if idx < len(raw_mask) and raw_mask[idx]:
                        valid_sug_indices.append(idx)
            
            if valid_sug_indices:
                top_idx = valid_sug_indices[0] # Index
                
                # 仅当推荐的是新节点时介入
                if top_idx >= TRAIN_NODE_COUNT:
                    rl_load = raw_state[rl_act_idx * 3]
                    new_load = raw_state[top_idx * 3]
                    
                    is_better = False
                    if rl_act_idx == 0: 
                        if new_load < 0.8: is_better = True
                    else: 
                        if (rl_load - new_load > 0.15): is_better = True
                        elif rl_load > 0.8 and new_load < 0.5: is_better = True
                    
                    if is_better:
                        final_act_idx = top_idx
                        dec_type = "LLM_NEW_WIN"
                    else:
                        dec_type = "RL_DEFEND"

        # --- 3. Step ---
        # [核心修正] 发送 Index (final_act_idx) 而不是 ID
        next_s, next_m, r, done, next_info = env.step(final_act_idx)
        
        # 获取真实 ID 仅用于打印日志
        real_node_id = real_ids[final_act_idx] if final_act_idx < len(real_ids) else -1
        
        recorder.record_step(step, r, final_act_idx, real_node_id, raw_state, dec_type)
        print(f"\rStep {step} | Act: {final_act_idx} (ID:{real_node_id}) | R: {r:.2f}", end="", flush=True)
        
        if done: break
        
        raw_state, raw_mask, info, current_desc = next_s, next_m, next_info, next_info.get('description', "")
        state_for_rl, mask_for_rl = env.parse_state_adaptive(raw_state, raw_mask)
        
        # 更新映射表
        real_ids = info.get('node_ids', [])
        id_to_idx = {uid: i for i, uid in enumerate(real_ids)}

    print("\n\n 决策完成! 自动请求 Java 进行物理仿真...")
    
    env.start_simulation() 
    energy, makespan = env.get_physical_metrics() 
    recorder.save_summary(energy, makespan) 
    env.shutdown_java() 

if __name__ == "__main__":
    evaluate()