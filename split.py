import os, json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_stratified(input_csv, out_dir, target_col, seed=0, test_size=0.2, val_size_within_train=0.1):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    if target_col not in df.columns:
        raise ValueError(f"target_col={target_col} not in columns: {list(df.columns)}")

    y = df[target_col]
    strat = y if y.nunique() > 1 else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )

    y_tr = train_df[target_col]
    strat2 = y_tr if y_tr.nunique() > 1 else None

    tr_df, val_df = train_test_split(
        train_df, test_size=val_size_within_train, random_state=seed, shuffle=True, stratify=strat2
    )

    train_path = os.path.join(out_dir, "train.csv")
    val_path   = os.path.join(out_dir, "val.csv")
    test_path  = os.path.join(out_dir, "test.csv")

    tr_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "input_csv": input_csv,
        "seed": seed,
        "target_col": target_col,
        "test_size": test_size,
        "val_size_within_train": val_size_within_train,
        "n_total": len(df),
        "n_train": len(tr_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "label_dist_total": df[target_col].value_counts(normalize=True).to_dict(),
        "label_dist_train": tr_df[target_col].value_counts(normalize=True).to_dict(),
        "label_dist_val": val_df[target_col].value_counts(normalize=True).to_dict(),
        "label_dist_test": test_df[target_col].value_counts(normalize=True).to_dict(),
        "paths": {"train": train_path, "val": val_path, "test": test_path},
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved stratified split to {out_dir}: train={len(tr_df)} val={len(val_df)} test={len(test_df)}")

if __name__ == "__main__":
    split_csv_stratified("./data/adult/adult.csv", "./data/adult/splits_seed0", target_col="income", seed=0)
