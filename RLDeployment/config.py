# config.py


# --- 环境交互 ---
API_URL = "http://localhost:4567"  # Java 环境 REST API 的 URL

# --- 状态和动作维度 ---
# 计算公式: STATE_DIM = (2 * num_deployable_nodes) + (2 * num_total_modules) + 4
#           ACTION_DIM = num_deployable_nodes
# 请根据您 MicroservicePlacement.java 中的拓扑结构和应用定义，计算并替换下面的示例值
# NUM_DEPLOYABLE_NODES = 8  
# NUM_TOTAL_MODULES = 3     
# STATE_DIM = (3 * NUM_DEPLOYABLE_NODES) + (2 * NUM_TOTAL_MODULES) + 4
# ACTION_DIM = NUM_DEPLOYABLE_NODES
MAX_DEPLOYABLE_NODES = 50  
STATE_DIM = (4 * MAX_DEPLOYABLE_NODES) + 2 
ACTION_DIM = MAX_DEPLOYABLE_NODES
# ---  ---

# --- Dueling DQN 超参数 ---
GAMMA = 0.99                # 折扣因子 (Discount factor)
LEARNING_RATE = 0.00001       # Adam 优化器的学习率
BUFFER_SIZE = 50000         # 经验回放缓冲区的最大容量
BATCH_SIZE = 256             # 每次训练时从缓冲区采样的批次大小
TARGET_UPDATE_FREQ = 100     # 目标网络更新频率 (以 Episodes 为单位)
GRAD_CLIP_NORM = 1.0
# --- Epsilon-Greedy 探索策略参数 ---
EPSILON_START = 1.0         # 初始探索率
EPSILON_DECAY = 0.999      # 探索率衰减因子 (每个 episode 结束后乘以该值)
EPSILON_MIN = 0.05     # 最小探索率

# --- 训练过程控制 ---
MAX_EPISODES = 15000         # 最大训练回合数 (Episodes)
START_TRAIN_EPISODE = 50   # 在开始训练前收集的最小经验数量
MODEL_SAVE_FREQ = 50        # 模型权重保存频率 (以 Episodes 为单位)

# --- 文件路径 ---
DATA_LOG_FILE = 'rl_deployment_data.csv'   # (State, Action, Q-value) 数据记录文件
DATASET_FILE = 'llm_finetuning_data.jsonl'
MODEL_SAVE_PATH = 'dueling_dqn_model.pdparams' # 模型权重保存路径 (使用 .weights.h5)