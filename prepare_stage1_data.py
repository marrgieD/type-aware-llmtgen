import argparse
import pandas as pd
import numpy as np
import string
import random
from tqdm import tqdm

def generate_random_string(length=5):
    """生成随机字符串"""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Stage 1 data based on schema.")
    parser.add_argument("--reference_file", type=str, required=True, help="Path to real data (to steal schema)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save fake data")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of fake rows to generate")
    args = parser.parse_args()

    print(f"Loading schema from {args.reference_file}...")
    try:
        # 读取全部数据以准确判断 Mixed 类型 (为了获取 NaN/Zero 率)
        # 如果数据太大，可以只读前 10000 行
        ref_df = pd.read_csv(args.reference_file) 
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return

    fake_data = {}
    MIXED_THRESHOLD = 0.05  # 必须与 dataset.py 保持一致

    print("Generating fake content with Mixed Type support...")
    
    for col in tqdm(ref_df.columns):
        col_data = ref_df[col]
        dtype = col_data.dtype
        
        # --- 1. 判断是否为 Mixed 类型 ---
        # 逻辑复刻 dataset.py: (zero_rate + nan_rate) > 0.05
        zero_rate = (col_data == 0).mean()
        nan_rate = col_data.isna().mean()
        is_mixed = (zero_rate + nan_rate) > MIXED_THRESHOLD
        
        # --- 2. 生成数据 ---
        
        # 场景 A: Mixed 类型 (数值 + 0/NaN)
        if is_mixed:
            # 生成基础随机数
            fake_col = np.random.randint(0, 10000, size=args.num_samples).astype(float)
            
            # 强制注入 0 和 NaN 以确保 synthetic data 也被识别为 Mixed
            # 我们随机选择 10% 的位置变成 0，10% 的位置变成 NaN (确保 > 0.05)
            mask_zero = np.random.rand(args.num_samples) < 0.1
            mask_nan = np.random.rand(args.num_samples) < 0.1
            
            fake_col[mask_zero] = 0.0
            fake_col[mask_nan] = np.nan
            
            fake_data[col] = fake_col
            
        # 场景 B: 纯数值型
        elif np.issubdtype(dtype, np.number):
            # 生成随机整数或浮点
            fake_col = np.random.randint(0, 10000, size=args.num_samples)
            if np.issubdtype(dtype, np.floating):
                fake_col = fake_col.astype(float) + np.random.rand(args.num_samples)
            fake_data[col] = fake_col
            
        # 场景 C: 类别型
        else:
            # 生成随机字符串
            fake_col = [generate_random_string(random.randint(3, 10)) for _ in range(args.num_samples)]
            
            # 如果原数据有缺失值，我们也随机注入一点 "Missing" 或 NaN
            if nan_rate > 0:
                mask_nan = np.random.rand(args.num_samples) < 0.05
                # 注意：Pandas 保存 CSV 时，None/np.nan 会变成空字符串
                fake_col = np.array(fake_col, dtype=object)
                fake_col[mask_nan] = np.nan 
                
            fake_data[col] = fake_col

    fake_df = pd.DataFrame(fake_data)
    
    # 强制列顺序一致
    fake_df = fake_df[ref_df.columns]

    print(f"Saving {args.num_samples} rows to {args.output_file}...")
    fake_df.to_csv(args.output_file, index=False)
    print("Done! Data generated successfully.")
    
    # 简单验证
    print("\n--- Validation Check ---")
    for col in fake_df.columns:
        z = (fake_df[col] == 0).mean()
        n = fake_df[col].isna().mean()
        if (z + n) > MIXED_THRESHOLD:
            print(f"Column '{col}' generated as MIXED (Zero: {z:.2%}, NaN: {n:.2%})")

if __name__ == "__main__":
    main()