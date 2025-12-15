import json
import numpy as np
import os
import hashlib
import random
from datetime import datetime

# ================= 配置区域 (Config) =================

# 1. 旧的 Edge 数据路径 (您训练得最好的那个实验)
OLD_EDGE_FILE = "D:/Code/MD_DATA/experiments/20251215_133758/raw_training_metadata.json"

# 2. 新的 Cloud 数据路径 (您用 test.py 刚刚收集的)
# 请确保这个路径与您 test.py 里设置的 COLLECT_DATA_FILE 一致
NEW_CLOUD_FILE = "D:/Code/Microservice_Deployment/RLDeployment/raw_cloud_data.json"

# 将它们放入列表，脚本会自动遍历读取
RAW_FILES = [OLD_EDGE_FILE, NEW_CLOUD_FILE]

# 3. 筛选阈值
# Edge: >35.0 (过滤掉严重过载的，保留正常的和略有负载的)
# Cloud: >-10.0 (Cloud通常是-5左右，只要不是-50的失败调用都保留)
EDGE_REWARD_THRESHOLD = 35.0  
CLOUD_REWARD_THRESHOLD = -10.0

# 4. 目标数据量 (用于平衡类别)
TARGET_EDGE_COUNT = 3000  
TARGET_CLOUD_COUNT = 1000 

# ====================================================

def generate_alpaca_entry(desc, top_k_ids, instruction_type):
    """
    构造 Alpaca 格式的微调数据
    """
    if instruction_type == "Cloud_Fallback":
        # Cloud 场景指令：强调鲁棒性和兜底
        instruction = (
            "Role: Robustness Scheduler.\n"
            "Task: The edge layer is critically overloaded. Identify the Top-5 fallback nodes (Cloud/Edge).\n"
            "Output: A JSON list of valid node IDs ranked by priority."
        )
    else:
        # Edge 场景指令：加入 Trade-off 逻辑，允许在过载时选远程
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
    """计算环境描述的哈希值，用于去重"""
    if not desc: return "empty"
    return hashlib.md5(desc.encode('utf-8')).hexdigest()

def process_dataset():
    # 确定输出文件路径 (保存在旧实验目录下)
    if os.path.exists(OLD_EDGE_FILE):
        CURRENT_DIR = os.path.dirname(OLD_EDGE_FILE)
    else:
        CURRENT_DIR = os.path.dirname(NEW_CLOUD_FILE)
        
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = os.path.join(CURRENT_DIR, f"final_merged_dataset_{current_time}.jsonl")

    # 临时存储列表
    edge_candidates = []
    cloud_candidates = []
    seen_hashes = set() 

    total_read = 0
    duplicate_count = 0
    low_quality_count = 0
    empty_desc_count = 0

    print("=== 开始处理数据 ===")

    # --- 1. 遍历读取所有源文件 ---
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
                    
                    # 提取字段
                    q_values = np.array(data['q_vals'])
                    mask = np.array(data['mask'], dtype=bool)
                    node_ids_map = data['node_ids']
                    desc = data.get('desc', "") # 使用 get 防止报错
                    reward = data['rew']
                    action = data['act']

                    # [检查] 描述是否为空 (这是 test.py 之前 Bug 导致的问题)
                    if not desc:
                        empty_desc_count += 1
                        continue

                    # [去重] Diversity Check
                    h = get_input_hash(desc)
                    if h in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(h)

                    # [分类] 判断是 Cloud 策略还是 Edge 策略
                    # 假设 ID 2 是 Cloud (根据您的环境设定)
                    is_cloud_action = False
                    if node_ids_map and action < len(node_ids_map) and node_ids_map[action] == 2:
                        is_cloud_action = True
                    
                    # [筛选] Quality Check
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

                    # [计算] 离线生成 Top-5 推荐
                    masked_q = q_values.copy()
                    masked_q[~mask] = -1e9 # Mask 掉无效动作
                    
                    K = 5
                    # 获取 Q 值最大的前 K 个动作索引
                    top_k_indices = np.argsort(masked_q)[-K:][::-1].tolist()
                    
                    # 将动作索引翻译为物理 Node ID
                    real_ids = []
                    if node_ids_map:
                        for idx in top_k_indices:
                            if 0 <= idx < len(node_ids_map):
                                rid = node_ids_map[idx]
                                if rid != -1: 
                                    real_ids.append(rid)
                    else:
                        real_ids = top_k_indices

                    # [生成] 构造数据条目
                    entry = generate_alpaca_entry(desc, real_ids, i_type)
                    
                    if i_type == "Cloud_Fallback":
                        cloud_candidates.append(entry)
                    else:
                        edge_candidates.append(entry)

                except Exception as e:
                    # print(f"解析错误: {e}") 
                    continue

    print("-" * 30)
    print(f"原始数据读取统计:")
    print(f"  - 总读取行数: {total_read}")
    print(f"  - 丢弃空描述(Empty Desc): {empty_desc_count}")
    print(f"  - 丢弃重复(Duplicate): {duplicate_count}")
    print(f"  - 丢弃低分(Low Reward): {low_quality_count}")
    print(f"  - 有效 Edge 样本: {len(edge_candidates)}")
    print(f"  - 有效 Cloud 样本: {len(cloud_candidates)}")
    print("-" * 30)

    # --- 2. 数据平衡 (Balancing) ---
    
    # A. Edge 数据处理 (下采样 / Downsampling)
    if len(edge_candidates) > TARGET_EDGE_COUNT:
        print(f"Edge 样本过多，随机采样至 {TARGET_EDGE_COUNT} 条...")
        final_edge = random.sample(edge_candidates, TARGET_EDGE_COUNT)
    else:
        final_edge = edge_candidates

    # B. Cloud 数据处理 (过采样 / Oversampling)
    if len(cloud_candidates) == 0:
        print("[严重警告] 未找到任何有效的 Cloud 样本！请检查 test.py 是否生成了正确数据。")
        final_cloud = []
    elif len(cloud_candidates) < TARGET_CLOUD_COUNT:
        print(f"Cloud 样本不足 ({len(cloud_candidates)}条)，执行复制增强至约 {TARGET_CLOUD_COUNT} 条...")
        # 计算需要复制几倍
        repeat_factor = TARGET_CLOUD_COUNT // len(cloud_candidates) + 1
        # 复制列表
        extended_cloud = (cloud_candidates * repeat_factor)
        # 截取到目标数量
        final_cloud = extended_cloud[:TARGET_CLOUD_COUNT]
    else:
        # 如果 Cloud 样本本身就够多 (比如跑了很久的 test.py)，就随机采样
        final_cloud = random.sample(cloud_candidates, TARGET_CLOUD_COUNT)

    # --- 3. 合并与写入 ---
    final_dataset = final_edge + final_cloud
    random.shuffle(final_dataset) # 打乱顺序，这对训练很重要

    print(f"正在写入最终文件: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in final_dataset:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成！最终数据集包含 {len(final_dataset)} 条样本。")
    print(f"其中 Edge: {len(final_edge)}, Cloud: {len(final_cloud)}")

if __name__ == "__main__":
    process_dataset()