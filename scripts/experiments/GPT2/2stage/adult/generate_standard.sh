# ================= 路径配置 (最关键) =================
# 这里填你刚才训练成功的那个 latest 目录的绝对路径
# 注意：代码逻辑里 load_model_from_checkpoint 需要具体到文件
# 优先找 model.safetensors，如果没有就找 resume_checkpoint_dict.pt
CHECKPOINT_DIR="/data/users/magg13_d_/DP-2Stage/runs/2Stage_Standard/adult_Standard_Stage2_Eps1/adult/entire/ts30932-bs32-epoch10-mbs16-eps1-clip1/latest"

# 指定模型文件路径 (二选一，推荐 safetensors)
CHECKPOINT_PATH="${CHECKPOINT_DIR}/model.safetensors"
# 如果报错找不到 safetensors，就解开下面这行用 pt 文件：
# CHECKPOINT_PATH="${CHECKPOINT_DIR}/resume_checkpoint_dict.pt"

# 输出目录 (生成的合成数据存哪)
OUTPUT_DIR="./runs/stage2_generation"
SYNTH_FOLDER="./data/synthetic/adult_stage2"

# ================= 模型配置 (必须和训练时一致) =================
MODEL_NAME_OR_PATH="gpt2"
MODEL_TYPE="gpt2"

# 训练时用的 entire 还是 lora？看你的路径里写的是 entire，这里就填 entire
FINETUNE_ADAPTER="entire" 

# ================= 生成参数 =================
DEVICE="cuda"
SEED=42

# 原始训练数据路径 (用于读取 Schema/Structure)
TRAIN_FILE="./data/adult/train.csv"

# 生成多少条数据？(通常和训练集大小一致，或者你自己定)
N_SYNTH_SAMPLES=32561 
# 生成几套数据集？
N_SYNTH_SET=1

# 批次大小 (H800显存大，生成时可以开大点加速)
SAMPLE_BATCH_SIZE=256

# 生成多样性控制
TOP_P=1.0
TEMPERATURE=0.7 # 0.7 会更保守，1.0 会更多样

# 其他开关
LOADING_4_BIT=False
DO_IMPUTE=True
MAX_FINETUNE_TRAIN_SIZE=50000 # 设为 None 表示读取全部
SYNTH_SAVE_AS=dp_synth256
GENERATION_SEED=2000
