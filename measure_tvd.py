import pandas as pd
import numpy as np
import itertools
import random
from tqdm import tqdm

# ================= 配置 =================
REAL_PATH = "./data/adult/adult.csv"
# FAKE_PATH = "./synthetic_data_episilon2.csv" # 您的合成数据
FAKE_PATH = "./synthetic_data_episilon1_nnew.csv"
K_WAY = [1,2, 3,4 ,5]  # 建议先测 2 和 3，4-way 在小数据集上偏差太大参考价值低
N_BINS = 10     # 降低分箱数以减少稀疏性偏差 (PrivMRF 常用 10-16)
# =======================================

def load_robust(path, tag):
    """
    鲁棒的加载函数：不删数据，只做填充
    """
    print(f"[{tag}] 正在读取 {path} ...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None
    
    print(f"    -> 原始行数: {len(df)}")
    
    # 1. 填充空值，而不是删除
    # 所有的 NaN, None 都填成 "MISSING"
    df = df.fillna("MISSING")
    
    # 2. 统一字符串格式 (去空格、转小写)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        # 把 "?" 统一替换为 "missing" (或者保留 ? 也可以，只要统一就行)
        df[col] = df[col].replace(['?', 'na', 'nan'], 'missing')
        
    return df

def get_quantile_edges(series, n_bins):
    """获取分位数边界"""
    try:
        # 强制转为数字，无法转换的变为 NaN
        series_num = pd.to_numeric(series, errors='coerce').dropna()
        if len(series_num) == 0: return None
        _, bins = pd.qcut(series_num, n_bins, retbins=True, duplicates='drop')
        if len(bins) < 2: return None
        return bins
    except:
        return None

def discretize_aligned(real_df, fake_df):
    """
    对齐离散化：确保 Real 和 Fake 使用相同的编码映射
    """
    real_disc = real_df.copy()
    fake_disc = fake_df.copy()
    
    common_cols = [c for c in real_df.columns if c in fake_df.columns]
    print(f"\n>>> 正在对齐并离散化 {len(common_cols)} 个列...")

    for col in common_cols:
        # 判断列类型：如果 Real Data 该列主要是数字，就当数值列处理
        real_num = pd.to_numeric(real_df[col], errors='coerce')
        is_numeric = real_num.notna().mean() > 0.8  # 超过80%是数字就当数值列
        
        if is_numeric:
            # --- 数值列处理 ---
            edges = get_quantile_edges(real_df[col], N_BINS)
            if edges is not None:
                # 使用 Real 的边界分箱
                # 关键：无法转数字的（比如合成出的 '?'），fill_value=-1
                
                # Real
                r_valid = pd.to_numeric(real_df[col], errors='coerce')
                real_disc[col] = pd.cut(r_valid, bins=edges, labels=False, include_lowest=True)
                real_disc[col] = real_disc[col].fillna(-1).astype(int) # -1 代表异常/缺失/非数字
                
                # Fake
                f_valid = pd.to_numeric(fake_df[col], errors='coerce')
                fake_disc[col] = pd.cut(f_valid, bins=edges, labels=False, include_lowest=True)
                fake_disc[col] = fake_disc[col].fillna(-1).astype(int)
            else:
                # 数值太稀疏，退化为分类
                is_numeric = False
        
        if not is_numeric:
            # --- 分类列处理 ---
            # 获取所有出现的类别并集
            unique_vals = sorted(list(set(real_df[col].unique()) | set(fake_df[col].unique())))
            val_map = {val: i for i, val in enumerate(unique_vals)}
            
            real_disc[col] = real_df[col].map(val_map).fillna(0).astype(int)
            fake_disc[col] = fake_df[col].map(val_map).fillna(0).astype(int)

    return real_disc[common_cols], fake_disc[common_cols]

def compute_tvd(real, fake, k, sample_size=1000):
    cols = real.columns.tolist()
    combs = list(itertools.combinations(cols, k))
    
    if len(combs) > sample_size:
        combs = random.sample(combs, sample_size)
    
    tvds = []
    # 只需要计算一次 len
    n_real = len(real)
    n_fake = len(fake)
    
    for c in tqdm(combs, desc=f"{k}-way", leave=False):
        c = list(c)
        
        # 1. 计算频次 (Value Counts)
        # 此时已经全是 0, 1, 2... 的整数了，直接算
        vc_real = real.value_counts(subset=c, normalize=True)
        vc_fake = fake.value_counts(subset=c, normalize=True)
        
        # 2. 对齐索引 (Outer Join)
        # 这一步会自动把 Real 有但 Fake 没有的组合填为 0，反之亦然
        # 这就是 TVD 捕捉分布差异的核心
        aligned = pd.concat([vc_real, vc_fake], axis=1, keys=['real', 'fake']).fillna(0)
        
        # 3. TVD 公式
        tvd = 0.5 * (aligned['real'] - aligned['fake']).abs().sum()
        tvds.append(tvd)
        
    return np.mean(tvds), np.std(tvds)

def main():
    # 1. 加载
    real_raw = load_robust(REAL_PATH, "Real")
    fake_raw = load_robust(FAKE_PATH, "Fake")
    
    if real_raw is None or fake_raw is None: return

    # 2. 采样对齐 (可选，但推荐)
    # 既然您的合成数据有4万条，如果和真实数据量级差不多(4.5万)，
    # 可以不用采样，或者统一采到 40000 条，保证公平
    min_len = min(len(real_raw), len(fake_raw))
    print(f"\n>>> 数据量对齐: 统一使用 {min_len} 条数据进行计算 (避免小样本偏差)")
    real_sample = real_raw.sample(n=min_len, random_state=42)
    fake_sample = fake_raw.sample(n=min_len, random_state=42)
    
    # 3. 离散化
    real_disc, fake_disc = discretize_aligned(real_sample, fake_sample)
    
    # 4. 计算
    print("\n>>> 开始计算 TVD...")
    for k in K_WAY:
        mean, std = compute_tvd(real_disc, fake_disc, k)
        print(f"   📊 {k}-way Marginal TVD: {mean:.4f} (std: {std:.4f})")

if __name__ == "__main__":
    main()