LLM=GPT2
# 请确保路径正确，如果你的项目路径不同请修改 ROOT_FOLDER
ROOT_FOLDER=~
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage
source $PROJECT_FOLDER/scripts/experiments/${LLM}.sh
# 加载 Adult 数据集的通用配置 (Batch Size 等)
source $PROJECT_FOLDER/scripts/experiments/GPT2/2stage/adult_general.sh

# ==========================================
# 1. 数据集设置 (使用真实的 Adult 数据)
# ==========================================
DATASET_NAME="adult"
# 强制覆盖 adult_general.sh 可能设置的测试路径，确保指向 train.csv
TRAIN_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv"
VALIDATION_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/test.csv"

# 重新计算训练集大小
TRAIN_SIZE=`cat ${TRAIN_FILE} | wc -l`
MAX_FINETUNE_TRAIN_SIZE=$(($TRAIN_SIZE-1))

# ==========================================
# 2. Stage 2 训练参数 (DP 设置)
# ==========================================
STAGE=2
WEIGHTED_LOSS=0.65       # DP 训练通常使用 Weighted Loss (论文推荐 0.65)
ENABLE_PRIVACY=True      # 开启差分隐私
EPSILON=1000                # 隐私预算 epsilon
CLIP=1                   # 梯度裁剪范数
FINETUNE_ADAPTER="entire" # 全参数微调
SHUFFLE_DATASET=True     # 真实数据训练需要打乱
LEARNING_RATE=0.0005     # 学习率
TRAIN_BATCH_SIZE=64
# 生成时的参数
DO_IMPUTE=True           
REJECTION_SAMPLE=True    

# ==========================================
# 3. 关键：加载 Stage 1 模型路径
# ==========================================
# 这里填入你刚刚跑出来的 Stage 1 结果路径
export MODEL_NAME_OR_PATH_2STAGE="/data/users/magg13_d_/DP-2Stage/runs/2Stage_LR0.0005-k1000-linear/stage1_shuffle-False-adult_wl-1/adult-uniform/NonDP/GPT2/entire/ts30932-bs32-epoch5/epoch5"

# ==========================================
# 4. 输出路径管理
# ==========================================
# 定义一个清晰的输出目录，方便查找
IDENTIFIER="Standard_Stage2_Eps${EPSILON}"
BASEFOLDER="${PROJECT_FOLDER}/runs/2Stage_Standard/${DATASET_NAME}_${IDENTIFIER}"

# 设置 2Stage_train.sh 需要的 OUTPUT_DIR 变量
OUTPUT_DIR=${BASEFOLDER}/ts${MAX_FINETUNE_TRAIN_SIZE}-bs${TRAIN_BATCH_SIZE}-epoch${FINETUNE_EPOCH}-eps${EPSILON}
RESUME_FROM_CHECKPOINT=False
# 确保 Checkpoint 和生成路径正确
# CHECKPOINT_PATH=$OUTPUT_DIR/model.safetensors
SYNTH_FOLDER=$OUTPUT_DIR/synth_data

# 强烈建议：一直开着，没checkpoint时 find_latest_checkpoint 会返回 None，不会报错
RESUME_FROM_CHECKPOINT=True

# 确保每个 epoch 都存
SAVE_EVERY_EPOCH=1

# 可选：epoch 内也存（防止断在 epoch 中间）
# 先保守设大一点，避免存太多；如果仍然经常断在 epoch 中间，就调小
SAVE_EVERY_STEP=100
