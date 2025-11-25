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
            return self._parse(data)
        except Exception as e:
            print(f"Reset error: {e}")
            return None, None

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
        s = np.array(data.get('stateVector', []), dtype=np.float32)
        m = np.array(data.get('actionMask', []), dtype=bool)
        if s.shape[0] != STATE_DIM:
            # 补零以防万一
            s = np.resize(s, (STATE_DIM,))
        return s, m

    def stop_server(self):
        try: self.session.get(f"{self.base_url}/stop", timeout=1)
        except: pass