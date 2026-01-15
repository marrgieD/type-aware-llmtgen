#!/bin/bash

# ================= 配置区域 =================
TRAIN_PATH="./data/adult/train.csv"
TEST_PATH="./data/adult/train.csv"

# 您的文件名
FAKE_PATHS=(
    # "./synthetic_data_episilon1.csv"
    # "./synthetic_data_episilon2.csv"
    # "./synthetic_data_episilon05.csv"
    # "./synthetic_data_episilon2_new.csv"
    # "./synthetic_data_episilon2_newt07.csv"
    # "./synthetic_data_episilon2_nnew.csv"
    # "./synthetic_data_episilon1_nnew.csv"
    # "./synthetic_data_episilon0_restage1.csv"
    "./data/synthetic/adult_stage2_impute-False/raw_tables/dp_synth256_nodp.csv"
    "./data/synthetic/adult_stage2_impute-True/raw_tables/dp_synth256.csv"
)

# 创建结果文件夹
OUTPUT_DIR="./eval_results"
mkdir -p "$OUTPUT_DIR"

# 参数设置
TARGET_NAME="income"
BINS="50"
SCORERS="auc accuracy" 
MODELS="xgb logistic"
# ===========================================

for fake_path in "${FAKE_PATHS[@]}"
do 
    # 提取文件名
    filename=$(basename -- "$fake_path")
    filename="${filename%.*}"
    RESULT_PATH="${OUTPUT_DIR}/report_${filename}.csv"

    echo "========================================================"
    echo "正在评估: $fake_path"
    echo "结果保存至: $RESULT_PATH"
    echo "========================================================"

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