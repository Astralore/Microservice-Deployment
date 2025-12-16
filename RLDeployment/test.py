import os
import json
import numpy as np
import paddle
import time
from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from config import MODEL_SAVE_PATH

# --- 配置区域 ---
COLLECT_DATA_FILE = "raw_cloud_data.json"
NUM_EPISODES = 20  
# ----------------

def save_raw_metadata(file_path, description, node_ids_map, q_values, mask, action, reward):
    entry = {
        "desc": description,
        "node_ids": node_ids_map,
        "q_vals": q_values.tolist(),
        "mask": mask.tolist(),
        "act": int(action),
        "rew": round(float(reward), 4), # [修改] 统一保留 4 位小数，与 train.py 一致
        "ts": time.time()               # [新增] 补上时间戳，虽然不用，但保持格式整齐
    }
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def test_agent():
    print(f"=== Starting Data Collection Mode (Target: {NUM_EPISODES} Episodes) ===")
    
    env = EnvironmentClient()
    agent = DuelingDQNAgent()
    
    # 加载模型
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading trained model from {MODEL_SAVE_PATH}...")
        try:
            layer_state_dict = paddle.load(MODEL_SAVE_PATH)
            agent.main_network.set_state_dict(layer_state_dict)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Error: Model file not found!")
        return

    agent.epsilon = 0.0 # 关闭探索
    
    # 清理旧文件
    if os.path.exists(COLLECT_DATA_FILE):
        try:
            os.remove(COLLECT_DATA_FILE)
            print(f"Removed old {COLLECT_DATA_FILE}, starting fresh.")
        except:
            print("Warning: Could not remove old file, appending instead.")

    print(f"Data will be saved to: {os.path.abspath(COLLECT_DATA_FILE)}")

    try:
        # --- [新增] 外层大循环 ---
        for episode in range(NUM_EPISODES):
            print(f"--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")
            
            # 重置环境 (开始新的一轮)
            state, mask, info = env.reset()
            if state is None:
                print("Failed to reset environment. Retrying...")
                time.sleep(1)
                continue

            current_desc = info.get('description', "")
            current_node_ids = info.get('node_ids', [])
            
            step = 0
            ep_reward = 0
            
            while True:
                step += 1
                if not np.any(mask): break

                # 计算 Q 值
                state_tensor = paddle.to_tensor([state], dtype='float32')
                with paddle.no_grad():
                    q_values_tensor = agent.main_network(state_tensor)
                    q_values = q_values_tensor.numpy()[0]

                # 选动作
                masked_q = q_values.copy()
                masked_q[~mask] = -1e9
                action = np.argmax(masked_q)
                
                # 执行
                next_state, next_mask, reward, done, next_info = env.step(action)

                # 保存数据
                save_raw_metadata(
                    COLLECT_DATA_FILE,
                    current_desc,
                    current_node_ids,
                    q_values,
                    mask,
                    action,
                    reward
                )

                ep_reward += reward
                state = next_state
                mask = next_mask
                
                current_desc = next_info.get('description', "")
                current_node_ids = next_info.get('node_ids', [])
                
                if done:
                    # 打印单轮简报
                    print(f"  Ep {episode+1} Done. Steps: {step}, Reward: {ep_reward:.2f}")
                    break
            
            # 可选：每轮中间休息一下，防止 Java 端过热
            # time.sleep(0.1) 

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping Java server...")
        env.stop_server()
        print("Done.")

if __name__ == "__main__":
    test_agent()