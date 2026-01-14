# check_oov.py
import argparse
import pandas as pd

DEFAULT_CAT_COLS = [
    "workclass","education","marital-status","occupation",
    "relationship","race","sex","native-country"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Train CSV (your real train split)")
    ap.add_argument("--syn_csv", required=True, help="Synthetic CSV")
    ap.add_argument("--target_col", default="income", help="Target column (default income)")
    ap.add_argument("--cat_cols", default="",
                    help="Comma-separated categorical columns. If empty, use Adult defaults.")
    ap.add_argument("--topk", type=int, default=10, help="Show top-k OOV values per column")
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    syn = pd.read_csv(args.syn_csv)

    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()] if args.cat_cols else DEFAULT_CAT_COLS

    # basic checks
    missing_train = [c for c in cat_cols + [args.target_col] if c not in train.columns]
    missing_syn = [c for c in cat_cols + [args.target_col] if c not in syn.columns]
    if missing_train:
        raise SystemExit(f"Train missing columns: {missing_train}")
    if missing_syn:
        raise SystemExit(f"Synthetic missing columns: {missing_syn}")

    print("=== income distribution ===")
    print("[train]")
    print(train[args.target_col].astype(str).value_counts(dropna=False))
    print("\n[syn]")
    print(syn[args.target_col].astype(str).value_counts(dropna=False))

    print("\n=== OOV rate vs train vocab (per categorical col) ===")
    for c in cat_cols:
        tr = train[c].astype(str).fillna("__NA__")
        sy = syn[c].astype(str).fillna("__NA__")
        tr_set = set(tr.unique())
        oov_mask = ~sy.isin(tr_set)
        oov_rate = float(oov_mask.mean())

        print(f"\n[{c}] OOV_rate = {oov_rate:.4f}  (OOV_count={int(oov_mask.sum())}/{len(sy)})")

        if oov_rate > 0:
            # show top OOV values
            oov_vals = sy[oov_mask].value_counts().head(args.topk)
            print("  Top OOV values:")
            for v, cnt in oov_vals.items():
                print(f"    {repr(v)} : {int(cnt)}")

        # also show top overall values (helps spot '?' explosions)
        top_vals = sy.value_counts().head(5)
        print("  Top values in synthetic:")
        for v, cnt in top_vals.items():
            print(f"    {repr(v)} : {int(cnt)}")

if __name__ == "__main__":
    main()
