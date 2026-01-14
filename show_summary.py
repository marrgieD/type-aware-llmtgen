import pandas as pd
import glob
import os

# ================= 关键修改：强制显示所有行列 =================
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)        # 设置打印宽度，防止换行
pd.set_option('display.max_rows', None)     # 显示所有行
# ===========================================================

# 结果文件夹
RESULT_DIR = "./eval_results"

def show_summary():
    # 读取所有报告文件
    all_files = glob.glob(os.path.join(RESULT_DIR, "report_*.csv"))
    
    if not all_files:
        print(f"❌ 在 {RESULT_DIR} 没找到任何 CSV 文件，请先运行评估脚本！")
        return

    df_list = []
    real_baseline = None

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # 1. 提取真实数据基准 (Real Data Baseline)
            if real_baseline is None:
                real_df = df[df['tag'] == 'real'].copy()
                if not real_df.empty:
                    real_df['Dataset'] = 'Real Data (基准)'
                    real_baseline = real_df

            # 2. 提取合成数据分数 (Fake Data)
            dataset_name = os.path.basename(filename).replace("report_", "").replace(".csv", "")
            
            # 过滤出 tag='fake' 的行
            fake_df = df[df['tag'] == 'fake'].copy()
            fake_df['Dataset'] = dataset_name
            df_list.append(fake_df)

        except Exception as e:
            print(f"⚠️ 读取 {filename} 失败: {e}")

    if not df_list:
        print("没有找到有效数据。")
        return

    # 合并
    if real_baseline is not None:
        df_list.insert(0, real_baseline)
    
    full_df = pd.concat(df_list, ignore_index=True)

    print("\n" + "="*80)
    print(" 📊 合成数据质量完整对比报告")
    print("="*80 + "\n")

    # --- 1. 机器学习效用 ---
    print("【1. 机器学习效用 (Efficacy)】")
    if 'efficacy_test' in full_df['metric'].values:
        eff_df = full_df[full_df['metric'] == 'efficacy_test']
        pivot = pd.pivot_table(
            eff_df, 
            values='score', 
            index=['model_name', 'scorer'], 
            columns=['Dataset']
        )
        print(pivot.round(4))
    else:
        print("(暂无数据)")
    print("-" * 80)

    # --- 2. 统计分布相似度 ---
    print("\n【2. 单列分布相似度 (Histogram Intersection)】")
    if 'histogram_intersection' in full_df['metric'].values:
        hist_df = full_df[full_df['metric'] == 'histogram_intersection']
        pivot = pd.pivot_table(
            hist_df, 
            values='score', 
            index=['metric'], 
            columns=['Dataset'], 
            aggfunc='mean'
        )
        print(pivot.round(4))
    else:
        print("(暂无数据)")
    print("-" * 80)

    # --- 3. 相关性矩阵准确度 ---
    print("\n【3. 相关性矩阵一致性 (Correlation Accuracy)】")
    if 'correlation_accuracy' in full_df['metric'].values:
        corr_df = full_df[full_df['metric'] == 'correlation_accuracy']
        pivot = pd.pivot_table(
            corr_df, 
            values='score', 
            index=['metric'], 
            columns=['Dataset'], 
            aggfunc='mean'
        )
        print(pivot.round(4))
    else:
        print("(暂无数据)")
    print("-" * 80)

    # --- 4. 隐私/重复数据 ---
    print("\n【4. 隐私风险: 精确重复样本数 (Exact Duplicates)】")
    if 'exact_duplicates' in full_df['metric'].values:
        dup_list = []
        for filename in all_files:
            try:
                d = pd.read_csv(filename)
                name = os.path.basename(filename).replace("report_", "").replace(".csv", "")
                
                # 提取 Fake vs Train
                ft = d[d['tag'] == 'fake_train']
                if not ft.empty:
                    ft = ft.copy()
                    ft['Dataset'] = name
                    dup_list.append(ft)
                
                # 提取 Real vs Train (基准)
                if real_baseline is not None and len(dup_list) == 1:
                     rt = d[d['tag'] == 'real']
                     rt = rt[rt['metric'] == 'exact_duplicates']
                     if not rt.empty:
                         rt = rt.copy()
                         rt['Dataset'] = 'Real Data (基准)'
                         dup_list.insert(0, rt)
            except: pass
            
        if dup_list:
            dup_df = pd.concat(dup_list)
            pivot = pd.pivot_table(dup_df, values='score', index=['metric'], columns=['Dataset'])
            print(pivot)
    else:
        print("(暂无数据)")
    print("\n" + "="*80)

if __name__ == "__main__":
    show_summary()