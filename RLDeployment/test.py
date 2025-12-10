import os
import time
import numpy as np
import paddle
from environment_client import EnvironmentClient
from agent import DuelingDQNAgent 
from config import MODEL_SAVE_PATH, ACTION_DIM

def test_agent():
    print("=== Starting Inference Mode (No Training) ===")
    
    # 1. 初始化环境
    # 这会尝试连接 Java 端
    env = EnvironmentClient()
    
    # 2. 初始化智能体
    agent = DuelingDQNAgent()
    
    # 3. [关键修复] 加载训练好的模型
    # 在您的 agent.py 中，网络变量名叫 main_network，不是 model 也不是 policy_net
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading trained model from {MODEL_SAVE_PATH}...")
        try:
            # 加载参数字典
            layer_state_dict = paddle.load(MODEL_SAVE_PATH)
            
            # 将参数加载到主网络 (main_network) 和目标网络 (target_network)
            agent.main_network.set_state_dict(layer_state_dict)
            agent.target_network.set_state_dict(layer_state_dict)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    else:
        print(f"Error: Model file '{MODEL_SAVE_PATH}' not found! Please run train.py first.")
        return

    # 4. [关键] 关闭探索模式
    # 设置 epsilon 为 0，确保 AI 只选 Q 值最高的动作（确定性策略）
    agent.epsilon = 0.0
    
    print("Agent is ready. Starting deployment inference...")

    try:
        # 我们只跑 1 个 Episode 来看最终结果
        # Java 端会在所有请求处理完后生成 Final Report
        state, mask, info = env.reset()
        
        if state is None:
            print("Failed to reset environment.")
            return

        total_reward = 0
        step = 0
        
        while True:
            step += 1
            
            # 检查是否有可用动作
            if not np.any(mask):
                print("All nodes masked out! Cannot proceed.")
                break

            # --- 核心：直接让 AI 预测 ---
            # explore=False 表示完全不随机，只选 Q 值最高的
            action = agent.select_action(state, mask, explore=False)
            
            # 执行动作
            next_state, next_mask, reward, done, next_info = env.step(action)

            # 打印每一步的决策 (可选，用于调试)
            # print(f"Step {step}: Action {action}, Reward {reward:.2f}")

            total_reward += reward
            state = next_state
            mask = next_mask
            
            if done:
                print(f"\nDeployment Finished!")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("Testing interrupted by user.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止 Java 端，这会触发 Java 控制台打印最终的 "FINAL RL DEPLOYMENT REPORT"
        try:
            print("Stopping Java server to generate report...")
            env.stop_server()
        except:
            pass

if __name__ == "__main__":
    # 检查 Paddle 设备
    print("PaddlePaddle using device:", paddle.get_device())
    test_agent()