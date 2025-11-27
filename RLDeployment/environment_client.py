# environment_client.py
import requests
import json
import numpy as np
import time
from config import API_URL, STATE_DIM, ACTION_DIM

class EnvironmentClient:
    def __init__(self, base_url=API_URL):
        self.base_url = base_url
        self.session = requests.Session()
        # 尝试连接，等待 Java 启动
        for i in range(5):
            try:
                requests.get(base_url + "/", timeout=1) # 只是ping一下
                print("Connected to Java environment.")
                break
            except:
                print("Waiting for Java server...")
                time.sleep(2)

    def reset(self):
        try:
            resp = self.session.post(f"{self.base_url}/reset", timeout=10)
            data = resp.json()
            
            # 兼容逻辑：尝试从不同位置获取状态数据
            # 有些 Java 实现直接返回 stateVector，有些包裹在 nextStateRepresentation 中
            ns = data
            if 'nextStateRepresentation' in data:
                ns = data['nextStateRepresentation']
            elif 'stateVector' not in data:
                # 如果既没有 nextStateRepresentation 也没有 stateVector，可能是包裹在 data 自身
                pass 
                
            # 解析状态和掩码
            state, mask = self._parse(ns)
            
            # 提取描述信息 (用于 LLM)
            description = ns.get('description', "")
            # 如果 ns 里没有，尝试从最外层 data 找
            if not description:
                description = data.get('description', "")
                
            info = {"description": description}
            
            return state, mask, info # [修复] 返回 3 个值
            
        except Exception as e:
            print(f"Reset error: {e}")
            return None, None, {} # [修复] 异常时也返回 3 个值

    def step(self, action):
        try:
            resp = self.session.post(f"{self.base_url}/step", json={'action': int(action)}, timeout=10)
            data = resp.json()
            ns = data.get('nextStateRepresentation', {})
            state, mask = self._parse(ns)
            reward = float(data.get('immediateReward', 0.0))
            done = bool(data.get('done', False))
            
            # 提取 LLM 描述
            info = {"description": ns.get('description', "")}
            return state, mask, reward, done, info
        except Exception as e:
            print(f"Step error: {e}")
            return None, None, 0, True, {}

    # --- [核心修复] 补全缺失的方法 ---
    def get_final_reward(self):
        """获取 Episode 结束时的最终奖励"""
        try:
            resp = self.session.get(f"{self.base_url}/get_final_reward", timeout=10)
            data = resp.json()
            return float(data.get('finalReward', 0.0))
        except Exception as e:
            print(f"Get final reward error: {e}")
            return 0.0
    # -------------------------------

    def _parse(self, data):
        # s = np.array(data.get('stateVector', []), dtype=np.float32)
        # m = np.array(data.get('actionMask', []), dtype=bool)
        # if s.shape[0] != STATE_DIM:
        #     # 补零以防万一
        #     s = np.resize(s, (STATE_DIM,))
        # return s, m
        # 1. 解析状态向量
        s = np.array(data.get('stateVector', []), dtype=np.float32)
        
        # [修复] State Padding
        if s.shape[0] < STATE_DIM:
            # 计算需要补多少个 0
            pad_len = STATE_DIM - s.shape[0]
            # 在末尾补 0
            padding = np.zeros(pad_len, dtype=np.float32)
            s = np.concatenate([s, padding])
        elif s.shape[0] > STATE_DIM:
            # 截断 (理论上不应发生，除非 Java 端发的数据超过了 config 里的 MAX_NODES)
            s = s[:STATE_DIM]

        # 2. 解析动作掩码
        m = np.array(data.get('actionMask', []), dtype=bool)
        
        # [修复] Mask Padding
        if m.shape[0] < ACTION_DIM:
            # 计算需要补多少个 False
            pad_len = ACTION_DIM - m.shape[0]
            # 在末尾补 False (表示这些补出来的节点不可用)
            padding_m = np.zeros(pad_len, dtype=bool)
            m = np.concatenate([m, padding_m])
        elif m.shape[0] > ACTION_DIM:
            m = m[:ACTION_DIM]
            
        return s, m

    def stop_server(self):
        try: self.session.get(f"{self.base_url}/stop", timeout=1)
        except: pass