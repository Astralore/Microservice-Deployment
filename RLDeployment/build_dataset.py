import json
import numpy as np
import os
import hashlib
import random
from datetime import datetime

# ================= 配置区域 (Config) =================

# 1. 旧的 Edge 数据路径 (训练得最好的实验)
OLD_EDGE_FILE = "D:/Code/MD_DATA/experiments/20251215_133758/raw_training_metadata.json"

# 2. 新的 Cloud 数据路径 (test.py 收集的)
NEW_CLOUD_FILE = "D:/Code/Microservice-Deployment/RLDeployment/raw_cloud_data.json"

RAW_FILES = [OLD_EDGE_FILE, NEW_CLOUD_FILE]

# 3. 筛选阈值
EDGE_REWARD_THRESHOLD = 35.0  
CLOUD_REWARD_THRESHOLD = -10.0

# 4. 目标数据量
TARGET_EDGE_COUNT = 3000  
TARGET_CLOUD_COUNT = 1000 

# ====================================================

def generate_alpaca_entry(desc, top_k_ids, instruction_type):
    """构造 Alpaca 格式数据"""
    if instruction_type == "Cloud_Fallback":
        instruction = (
            "Role: Robustness Scheduler.\n"
            "Task: The edge layer is critically overloaded. Identify the Top-5 fallback nodes (Cloud/Edge).\n"
            "Output: A JSON list of valid node IDs ranked by priority."
        )
    else:
        instruction = (
            "Role: Intelligent Edge Scheduler.\n"
            "Task: Analyze the system state and select the Top-5 optimal Edge Node IDs.\n"
            "Logic Chain:\n"
            "1. FILTER: Exclude nodes with insufficient resources.\n"
            "2. RANK: Prioritize 'Link: Local' or 'Link: Neighbor' nodes to minimize latency.\n"
            "3. TRADEOFF: If local nodes are overloaded (>85%), select remote nodes with abundant resources.\n"
            "Output: A JSON list of the top 5 node IDs."
        )

    return {
        "instruction": instruction,
        "input": desc,
        "output": json.dumps(top_k_ids)
    }

def get_input_hash(desc):
    """计算哈希去重"""
    if not desc: return "empty"
    return hashlib.md5(desc.encode('utf-8')).hexdigest()

def process_dataset():
    if os.path.exists(OLD_EDGE_FILE):
        CURRENT_DIR = os.path.dirname(OLD_EDGE_FILE)
    else:
        CURRENT_DIR = os.path.dirname(NEW_CLOUD_FILE)
        
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = os.path.join(CURRENT_DIR, f"final_best_dataset_{current_time}.jsonl")

    # [修改] 临时存储列表，存元组: (reward, entry)
    edge_candidates = []
    cloud_candidates = []
    seen_hashes = set() 

    total_read = 0
    duplicate_count = 0
    low_quality_count = 0
    empty_desc_count = 0

    print("=== 开始处理数据 ===")

    for target_file in RAW_FILES:
        if not os.path.exists(target_file):
            print(f"[警告] 文件不存在，跳过: {target_file}")
            continue
            
        print(f"正在读取: {target_file} ...")
        
        with open(target_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                total_read += 1
                try:
                    data = json.loads(line)
                    
                    q_values = np.array(data['q_vals'])
                    mask = np.array(data['mask'], dtype=bool)
                    node_ids_map = data['node_ids']
                    desc = data.get('desc', "") 
                    reward = data['rew']
                    action = data['act']

                    if not desc:
                        empty_desc_count += 1
                        continue

                    h = get_input_hash(desc)
                    if h in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(h)

                    is_cloud_action = False
                    if node_ids_map and action < len(node_ids_map) and node_ids_map[action] == 2:
                        is_cloud_action = True
                    
                    if is_cloud_action:
                        if reward < CLOUD_REWARD_THRESHOLD: 
                            low_quality_count += 1
                            continue
                        i_type = "Cloud_Fallback"
                    else:
                        if reward < EDGE_REWARD_THRESHOLD: 
                            low_quality_count += 1
                            continue
                        i_type = "Edge_Optimization"

                    masked_q = q_values.copy()
                    masked_q[~mask] = -1e9
                    K = 5
                    top_k_indices = np.argsort(masked_q)[-K:][::-1].tolist()
                    
                    real_ids = []
                    if node_ids_map:
                        for idx in top_k_indices:
                            if 0 <= idx < len(node_ids_map):
                                rid = node_ids_map[idx]
                                if rid != -1: 
                                    real_ids.append(rid)
                    else:
                        real_ids = top_k_indices

                    entry = generate_alpaca_entry(desc, real_ids, i_type)
                    
                    # [修改] 这里同时存入 reward，方便后续排序
                    if i_type == "Cloud_Fallback":
                        cloud_candidates.append((reward, entry))
                    else:
                        edge_candidates.append((reward, entry))

                except Exception as e:
                    continue

    print("-" * 30)
    print(f"原始数据读取统计:")
    print(f"  - 总读取行数: {total_read}")
    print(f"  - 丢弃空描述: {empty_desc_count}")
    print(f"  - 丢弃重复: {duplicate_count}")
    print(f"  - 丢弃低分: {low_quality_count}")
    print(f"  - 有效 Edge 候选: {len(edge_candidates)}")
    print(f"  - 有效 Cloud 候选: {len(cloud_candidates)}")
    print("-" * 30)

    # --- 2. 数据平衡与优选 (Balancing & Selection) ---
    
    # A. Edge 数据处理：[修改] 排序取 Top-K
    if len(edge_candidates) > TARGET_EDGE_COUNT:
        print(f"Edge 样本过多 ({len(edge_candidates)}条)，正在按 Reward 降序筛选前 {TARGET_EDGE_COUNT} 条最佳数据...")
        
        # 1. 按 reward (x[0]) 降序排序
        edge_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 2. 截取前 N 个
        top_k_tuples = edge_candidates[:TARGET_EDGE_COUNT]
        
        # 3. 解包，只保留 entry
        final_edge = [x[1] for x in top_k_tuples]
        
        # 打印一下分数段，看看效果
        min_score = top_k_tuples[-1][0]
        max_score = top_k_tuples[0][0]
        print(f"  -> 筛选完成。保留的分数区间: [{min_score:.2f}, {max_score:.2f}]")
    else:
        # 如果不够，就全要
        final_edge = [x[1] for x in edge_candidates]

    # B. Cloud 数据处理 (保持过采样逻辑，但也需解包)
    if len(cloud_candidates) == 0:
        print("[严重警告] 未找到任何有效的 Cloud 样本！")
        final_cloud = []
    elif len(cloud_candidates) < TARGET_CLOUD_COUNT:
        print(f"Cloud 样本不足 ({len(cloud_candidates)}条)，执行复制增强...")
        repeat_factor = TARGET_CLOUD_COUNT // len(cloud_candidates) + 1
        extended_cloud = (cloud_candidates * repeat_factor)
        # 解包
        final_cloud = [x[1] for x in extended_cloud[:TARGET_CLOUD_COUNT]]
    else:
        # Cloud 够多的话，也选分高的
        cloud_candidates.sort(key=lambda x: x[0], reverse=True)
        top_k_cloud = cloud_candidates[:TARGET_CLOUD_COUNT]
        final_cloud = [x[1] for x in top_k_cloud]

    # --- 3. 合并与写入 ---
    final_dataset = final_edge + final_cloud
    random.shuffle(final_dataset) # 打乱顺序，这对训练很重要

    print(f"正在写入最终文件: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in final_dataset:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成！最终数据集包含 {len(final_dataset)} 条样本。")
    print(f"其中 Edge (Elite): {len(final_edge)}, Cloud (Oversampled): {len(final_cloud)}")

if __name__ == "__main__":
    process_dataset()