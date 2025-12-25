# [本地] llm_direct.py
import requests
import json

class DirectLLM:
    # 兼容性构造函数：即使传入 model_path 也会被忽略，防止报错
    def __init__(self, base_model_path=None, lora_path=None, api_url="http://localhost:6006/predict"):
        self.api_url = api_url
        # 如果通过 SSH 隧道映射了端口，localhost:6006 就是云端的 6006
        print(f"🔗 Remote LLM Bridge Initialized -> {self.api_url}")

    def get_suggestions(self, description):
        if not description:
            return []
            
        try:
            payload = {'description': description}
            # 设置 15秒超时，防止网络波动导致仿真卡死
            response = requests.post(self.api_url, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # 假设云端返回格式为 {"node_ids": [1, 2, 3]}
                suggestions = data.get('node_ids', [])
                if suggestions:
                    # 可选：打印一下日志确认收到推荐
                    # print(f" [LLM Recvd] {suggestions}")
                    pass
                return suggestions
            else:
                print(f"⚠️ Remote LLM Error (Status: {response.status_code})")
                return []
                
        except requests.exceptions.ConnectionError:
            print("⚠️ Connection Refused: 请检查 SSH 隧道是否开启 (ssh -L 6006:...)")
            return []
        except Exception as e:
            print(f"⚠️ Network Error: {e}")
            return []