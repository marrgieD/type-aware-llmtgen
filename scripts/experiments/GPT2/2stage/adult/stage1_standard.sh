LLM=GPT2
# 请确保路径正确，如果你的项目路径不同请修改 ROOT_FOLDER
ROOT_FOLDER=~
PROJECT_FOLDER=$ROOT_FOLDER/DP-2Stage

# 1. 加载模型基础配置
source $PROJECT_FOLDER/scripts/experiments/${LLM}.sh

# === 2. 补全缺失的参数 (关键修复) ===
TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE=16
EVAL_BATCH_SIZE=32

# 即使 Stage 1 不开隐私，脚本也需要这两个变量作为占位符，否则会报参数错误
EPSILON=1
CLIP=1

# === 3. 数据集配置 ===
DATASET_NAME="adult-uniform"
TRAIN_FILE="${PROJECT_FOLDER}/data/${DATASET_NAME}/k1000/train.csv"
# 验证集使用真实的 adult 数据
VALIDATION_FILE="${PROJECT_FOLDER}/data/adult/k1000/valid.csv"

# 计算数据量
TRAIN_SIZE=`cat ${TRAIN_FILE} | wc -l`
MAX_FINETUNE_TRAIN_SIZE=$(($TRAIN_SIZE-1))

# === 4. 训练参数配置 ===
STAGE=1
FINETUNE_ADAPTER=entire
SHUFFLE_DATASET=False
FINETUNE_EPOCH=5    
SAVE_EVERY_EPOCH=5
WEIGHTED_LOSS=-1    
FINETUNE_STEP=0
SAVE_EVERY_STEP=0
# === 5. 调用主逻辑 ===
source $PROJECT_FOLDER/scripts/experiments/GPT2/2stage/adult/stage1_master.sh