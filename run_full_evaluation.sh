#!/bin/bash

# ================= 配置区域 =================
TRAIN_PATH="./adult/train.csv"
TEST_PATH="./adult/train.csv" 

# 输出目录
OUTPUT_DIR="./eval_results"
mkdir -p "$OUTPUT_DIR"

# 参数设置
TARGET_NAME="income"
BINS="50"
SCORERS="auc accuracy" 
MODELS="xgb logistic"

# ================= 文件列表获取逻辑 =================

# 检查是否有命令行参数输入
if [ "$#" -gt 0 ]; then
    # 如果有参数，将所有参数存入数组
    FAKE_PATHS=("$@")
    echo ">>> 检测到输入参数，将评估以下 ${#FAKE_PATHS[@]} 个文件："
else
    # 如果没有参数，使用默认的硬编码列表
    echo ">>> 未检测到输入参数，使用默认列表："
    FAKE_PATHS=(
        "./data/synthetic/adult_stage2_impute-False/raw_tables/dp_synth256_nodp.csv"
        "./data/synthetic/adult_stage2_impute-True/raw_tables/dp_synth256.csv"
    )
fi

# 打印一下要跑的文件名，方便确认
printf ' - %s\n' "${FAKE_PATHS[@]}"
echo "========================================================"

# ================= 循环评估逻辑 =================

for fake_path in "${FAKE_PATHS[@]}"
do 
    # 0. 检查文件是否存在
    if [ ! -f "$fake_path" ]; then
        echo "错误：找不到文件 $fake_path ，跳过..."
        continue
    fi

    # 提取文件名用于生成报告名
    filename=$(basename -- "$fake_path")
    filename="${filename%.*}"
    RESULT_PATH="${OUTPUT_DIR}/report_${filename}.csv"

    echo ""
    echo "########################################################"
    echo "正在评估: $fake_path"
    echo "结果保存至: $RESULT_PATH"
    echo "########################################################"
    # --- 1. 精确重复 (Exact Duplicates) [覆盖模式] ---
    echo "[1/5] Running Exact Duplicates..."
    python metrics/run.py \
        --fake_path "${fake_path}" \
        --metric_name exact_duplicates \
        --train_path "$TRAIN_PATH" \
        --test_path "$TEST_PATH" \
        --result_path "$RESULT_PATH" \
        --overwrite

    # --- 2. 直方图交叉 (Histogram Intersection) ---
    echo "[2/5] Running Histogram Intersection..."
    python metrics/run.py \
        --fake_path "${fake_path}" \
        --metric_name histogram_intersection \
        --bins $BINS \
        --train_path "$TRAIN_PATH" \
        --test_path "$TEST_PATH" \
        --result_path "$RESULT_PATH" \
        --target_name "$TARGET_NAME"

    # --- 3. 成对相似度 (Pairwise Similarity) ---
    # *这是之前漏掉的*
    echo "[3/5] Running Pairwise Similarity..."
    python metrics/run.py \
        --fake_path "${fake_path}" \
        --metric_name pairwise_similarity \
        --bins $BINS \
        --train_path "$TRAIN_PATH" \
        --test_path "$TEST_PATH" \
        --result_path "$RESULT_PATH"

    # --- 4. 相关性矩阵准确度 (Correlation Accuracy) ---
    # *这是之前漏掉的*
    echo "[4/5] Running Correlation Accuracy..."
    python metrics/run.py \
        --fake_path "${fake_path}" \
        --metric_name correlation_accuracy \
        --train_path "$TRAIN_PATH" \
        --test_path "$TEST_PATH" \
        --result_path "$RESULT_PATH"

    # --- 5. 机器学习效用 (Efficacy Test) ---
    echo "[5/5] Running Efficacy Tests..."
    for scorer in ${SCORERS}
    do
        for model in ${MODELS}
        do
            echo "   -> Model: $model | Metric: $scorer"
            python metrics/run.py \
                --fake_path "${fake_path}" \
                --metric_name efficacy_test \
                --model_name "$model" \
                --scorer "$scorer" \
                --train_path "$TRAIN_PATH" \
                --test_path "$TEST_PATH" \
                --result_path "$RESULT_PATH" \
                --target_name "$TARGET_NAME"
        done
    done
    echo ""
done

echo "所有 5 项指标评估结束！"
