# evaluate_local.py
import time
import numpy as np
import paddle
import os
import json
from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from llm_direct import DirectLLM 
from config import ACTION_DIM # 这里的 ACTION_DIM 应该是 50 (训练时的配置)

# [配置]
# 请确认路径正确
RL_MODEL_PATH = "D:/Code/MD_DATA/experiments/20251215_133758/dueling_dqn_model.pdparams"
ENABLE_LLM = True 

# 训练时的节点数量 (RL 的视界边界)
TRAIN_NODE_COUNT = 49

class SafeEnvironmentClient(EnvironmentClient):
    """ 处理维度不匹配的客户端 """
    def parse_state_adaptive(self, raw_state_vec, raw_mask):
        current_nodes = len(raw_mask)
        target_nodes = TRAIN_NODE_COUNT 
        
        global_feats = raw_state_vec[-2:] 
        node_feats = raw_state_vec[:-2]   
        
        if current_nodes >= target_nodes:
            # [场景: 节点增加] 截断给 RL 看
            sliced_feats = node_feats[:target_nodes * 3]
            s_out = np.concatenate([sliced_feats, global_feats])
            m_out = raw_mask[:target_nodes]
        else:
            # [场景: 节点减少] 补零填充 (虽然本次实验用不到，但为了健壮性保留)
            pad_nodes = target_nodes - current_nodes
            dummy_feat = np.array([1.0, 1.0, -1.0] * pad_nodes, dtype=np.float32)
            s_out = np.concatenate([node_feats, dummy_feat, global_feats])
            dummy_mask = np.zeros(pad_nodes, dtype=bool)
            m_out = np.concatenate([raw_mask, dummy_mask])
            
        return s_out.astype('float32'), m_out.astype('bool')

def evaluate():
    print(f"🚀 Starting Evaluation (Nodes: {ACTION_DIM} -> Generalized, LLM={ENABLE_LLM})...")
    env = SafeEnvironmentClient()
    
    agent = DuelingDQNAgent()
    if os.path.exists(RL_MODEL_PATH):
        try:
            agent.main_network.set_state_dict(paddle.load(RL_MODEL_PATH))
            print(f"✅ RL Model Loaded")
        except:
            print("❌ Model Load Failed, using random weights.")
    else:
        print("⚠️ No model found.")

    llm = DirectLLM() # 连接 localhost:6006

    for episode in range(1, 6): # 跑 5 轮看看效果
        # Reset 得到的是 61 个节点的原始数据
        raw_state, raw_mask, info = env.reset()
        current_desc = info.get('description', "")
        
        # 适配给 RL (只给它看前 50 个)
        state_for_rl, mask_for_rl = env.parse_state_adaptive(raw_state, raw_mask)
        
        total_reward = 0
        step = 0
        stats = {"RL_Native": 0, "LLM_NewNode": 0, "LLM_OldNode_Opt": 0}
        
        while True:
            step += 1
            
            # --- 1. RL 计算 Q 值 (基于前 50 个节点) ---
            state_tensor = paddle.to_tensor(state_for_rl, dtype='float32').unsqueeze(0)
            with paddle.no_grad():
                q_values = agent.main_network(state_tensor).numpy()[0]
            
            valid_q = np.where(mask_for_rl, q_values, -1e9)
            rl_action_idx = np.argmax(valid_q)
            
            # 映射回真实物理 ID (从 info['node_ids'] 里取)
            real_node_ids = info.get('node_ids', [])
            if rl_action_idx < len(real_node_ids):
                rl_target_id = real_node_ids[rl_action_idx]
            else:
                rl_target_id = 0
            
            final_target_id = rl_target_id
            decision_type = "RL"

            # --- 2. LLM 介入 (基于全量 61 个节点) ---
            if ENABLE_LLM and current_desc:
                suggestions = llm.get_suggestions(current_desc) # [55, 12, 52...]
                
                # 过滤掉 Mask 为 False 的无效建议
                valid_suggestions = [pid for pid in suggestions if pid < len(raw_mask) and raw_mask[pid]]
                
                if valid_suggestions:
                    top_pick = valid_suggestions[0] # LLM 的 No.1 推荐
                    
                    # [情况 A]: LLM 推荐了 RL 看不见的新节点 (ID >= 50)
                    if top_pick >= TRAIN_NODE_COUNT:
                        final_target_id = top_pick
                        decision_type = "LLM_NEW"
                        stats["LLM_NewNode"] += 1
                        
                    # [情况 B]: LLM 推荐了 RL 能看见的老节点 (ID < 50)
                    # 只有当 RL 选了 Cloud (0) 或者 RL 的选择不在 LLM 推荐列表里时，才考虑修正
                    elif rl_target_id == 0: 
                        # 既然是老节点，让 RL 在 LLM 推荐的列表里挑一个 Q 值最高的
                        # (利用 RL 的微操能力)
                        visible_suggestions = [pid for pid in valid_suggestions if pid < TRAIN_NODE_COUNT]
                        if visible_suggestions:
                            # 注意：这里假设物理 ID == Action Index
                            best_rescue = max(visible_suggestions, key=lambda x: q_values[x])
                            final_target_id = best_rescue
                            decision_type = "LLM_OPT"
                            stats["LLM_OldNode_Opt"] += 1

            # --- 3. 执行动作 ---
            # Java 端能接收任意合法的 ID (包括 55)
            next_raw_state, next_raw_mask, reward, done, next_info = env.step(final_target_id)
            
            print(f"\rEp {episode} Step {step} | Act: {final_target_id} ({decision_type}) | R: {reward:.2f}", end="", flush=True)
            
            total_reward += reward
            if done:
                if np.any(raw_mask): total_reward += env.get_final_reward()
                break
            
            # 更新状态
            raw_state = next_raw_state
            raw_mask = next_raw_mask
            info = next_info
            current_desc = info.get('description', "")
            
            # 重新适配给 RL
            state_for_rl, mask_for_rl = env.parse_state_adaptive(raw_state, raw_mask)

        print(f"\nEpisode {episode} Done. Reward: {total_reward:.2f} | Stats: {json.dumps(stats)}")

if __name__ == "__main__":
    evaluate()