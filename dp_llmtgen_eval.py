# dp_llmtgen_eval.py
# DP-LLMTGen-style evaluation (practical version):
# - NO internal splitting. You provide:
#   --fit_csv   : real TRAIN split (for fitting quantile bins + category vocab)
#   --test_csv  : real TEST split (for evaluation reference)
#   --synthetic_csv : synthetic samples
# - k-way TVD (k=1..5) between synthetic and real test
# - numerical features quantiled into 20 groups (fit on fit_csv, applied to test+synthetic)
# - XGBoost downstream: grid search 5-fold CV on synthetic, test on real test

from __future__ import annotations
import argparse, json, math, itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class TabularEncoder:
    num_cols: List[str]
    cat_cols: List[str]
    target_col: str
    n_quantiles: int = 20
    num_edges_: Dict[str, np.ndarray] = None
    cat_maps_: Dict[str, Dict[str, int]] = None
    y_map_: Dict[str, int] = None

    def fit(self, df_fit: pd.DataFrame) -> "TabularEncoder":
        self.num_edges_ = {}
        self.cat_maps_ = {}

        # numeric: fit quantile bin edges on fit set
        for c in self.num_cols:
            x = pd.to_numeric(df_fit[c], errors="coerce").to_numpy()
            x = x[~np.isnan(x)]
            if x.size == 0 or np.unique(x).size <= 1:
                self.num_edges_[c] = np.array([-np.inf, np.inf], dtype=float)
                continue

            q = np.linspace(0, 1, num=self.n_quantiles + 1)
            edges = np.quantile(x, q)
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.array([-np.inf, np.inf], dtype=float)
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            self.num_edges_[c] = edges.astype(float)

        # categorical: vocab from fit set, unknown -> 0
        for c in self.cat_cols:
            s = df_fit[c].astype(str).fillna("__NA__")
            cats = sorted(s.unique().tolist())
            mp = {"__UNK__": 0}
            for i, v in enumerate(cats, start=1):
                mp[v] = i
            self.cat_maps_[c] = mp

        # target mapping from fit set (keeps label space stable)
        y = df_fit[self.target_col].astype(str).fillna("__NA__")
        uniq = sorted(y.unique().tolist())
        self.y_map_ = {v: i for i, v in enumerate(uniq)}
        return self

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        parts = []

        for c in self.num_cols:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            edges = self.num_edges_[c]
            bins = np.digitize(np.nan_to_num(x, nan=-1e308), edges[1:-1], right=False)
            parts.append(bins.astype(np.int32))

        for c in self.cat_cols:
            mp = self.cat_maps_[c]
            s = df[c].astype(str).fillna("__NA__")
            arr = np.array([mp.get(v, 0) for v in s], dtype=np.int32)
            parts.append(arr)

        if not parts:
            raise ValueError("No features: num_cols + cat_cols is empty.")
        return np.stack(parts, axis=1)

    def transform_y(self, df: pd.DataFrame) -> np.ndarray:
        s = df[self.target_col].astype(str).fillna("__NA__")
        # unknown label -> -1 (we will filter those rows)
        return s.map(lambda v: self.y_map_.get(v, -1)).to_numpy(dtype=np.int32)


def infer_num_cat_cols(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c != target_col]
    num_cols, cat_cols = [], []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        frac_numeric = np.mean(~np.isnan(s.to_numpy()))
        if frac_numeric >= 0.95:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _counts_tuples(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if M.ndim == 1:
        M2 = M.reshape(-1, 1)
    else:
        M2 = M
    dtype = np.dtype([("f{}".format(i), M2.dtype) for i in range(M2.shape[1])])
    V = np.ascontiguousarray(M2).view(dtype).ravel()
    keys, cnt = np.unique(V, return_counts=True)
    return keys, cnt.astype(np.float64)


def tvd_for_subset(Xa: np.ndarray, Xb: np.ndarray, cols: Sequence[int]) -> float:
    A = Xa[:, cols]
    B = Xb[:, cols]
    ka, ca = _counts_tuples(A)
    kb, cb = _counts_tuples(B)

    allk = np.union1d(ka, kb)
    pa = np.zeros(allk.shape[0], dtype=np.float64)
    pb = np.zeros(allk.shape[0], dtype=np.float64)

    ia = np.searchsorted(allk, ka)
    ib = np.searchsorted(allk, kb)
    pa[ia] = ca / max(1, Xa.shape[0])
    pb[ib] = cb / max(1, Xb.shape[0])

    return 0.5 * np.sum(np.abs(pa - pb))


def k_way_tvd(
    X_syn: np.ndarray,
    X_test: np.ndarray,
    k: int,
    num_subsets: int = 300,
    seed: int = 0,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n_features = X_syn.shape[1]
    if k < 1 or k > n_features:
        raise ValueError(f"k={k} invalid for n_features={n_features}")

    total_combos = math.comb(n_features, k)
    feats = list(range(n_features))

    # enumerate if small; else sample
    if (k <= 2) and (total_combos <= 20000):
        subsets = list(itertools.combinations(feats, k))
    else:
        m = min(num_subsets, total_combos)
        if total_combos <= 50000:
            all_combos = list(itertools.combinations(feats, k))
            idx = rng.choice(len(all_combos), size=m, replace=False)
            subsets = [all_combos[i] for i in idx]
        else:
            subsets_set = set()
            while len(subsets_set) < m:
                subset = tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist()))
                subsets_set.add(subset)
            subsets = list(subsets_set)

    tvds = np.array([tvd_for_subset(X_syn, X_test, cols=s) for s in subsets], dtype=np.float64)
    mean = float(tvds.mean())
    std = float(tvds.std(ddof=1) if tvds.size > 1 else 0.0)
    return mean, std, len(subsets)


def train_eval_xgboost(X_syn: np.ndarray, y_syn: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int = 0):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("xgboost not installed. Try: pip install xgboost") from e

    # filter unknown labels (-1)
    m_syn = y_syn >= 0
    m_te = y_test >= 0
    X_syn, y_syn = X_syn[m_syn], y_syn[m_syn]
    X_test, y_test = X_test[m_te], y_test[m_te]

    n_classes = int(np.unique(y_syn).size)
    if n_classes <= 1:
        return {"acc": float("nan"), "auc": float("nan"), "note": "Only one class in y_syn after filtering."}

    if n_classes == 2:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        scoring = "roc_auc"
    else:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        scoring = "accuracy"

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10, 20],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True, verbose=0)
    gs.fit(X_syn, y_syn)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    try:
        proba = best.predict_proba(X_test)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_test, proba, multi_class="ovo", average="macro"))
    except Exception:
        auc = float("nan")

    return {"acc": acc, "auc": auc, "best_params": gs.best_params_}


def evaluate(
    fit_csv: str,
    test_csv: str,
    synthetic_csv: str,
    target_col: str,
    seeds: Sequence[int] = (0,),
    n_quantiles: int = 20,
    tvd_subset_num: int = 300,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Dict[str, float]:

    fit_df = pd.read_csv(fit_csv)
    test_df = pd.read_csv(test_csv)
    syn_df = pd.read_csv(synthetic_csv)

    # infer columns if not provided
    if num_cols is None or cat_cols is None:
        nc, cc = infer_num_cat_cols(fit_df, target_col)
        num_cols = nc if num_cols is None else num_cols
        cat_cols = cc if cat_cols is None else cat_cols

    needed = set(num_cols + cat_cols + [target_col])
    for name, df in [("fit_csv", fit_df), ("test_csv", test_df), ("synthetic_csv", syn_df)]:
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    tvd_means = {k: [] for k in range(1, 6)}
    tvd_stds = {k: [] for k in range(1, 6)}
    xgb_accs, xgb_aucs = [], []

    for sd in seeds:
        enc = TabularEncoder(
            num_cols=num_cols,
            cat_cols=cat_cols,
            target_col=target_col,
            n_quantiles=n_quantiles,
        ).fit(fit_df)

        X_syn = enc.transform_X(syn_df)
        X_test = enc.transform_X(test_df)

        # TVD
        for k in range(1, 6):
            mean_k, std_k, used = k_way_tvd(
                X_syn=X_syn, X_test=X_test, k=k, num_subsets=tvd_subset_num, seed=sd
            )
            tvd_means[k].append(mean_k)
            tvd_stds[k].append(std_k)

        # XGB
        y_syn = enc.transform_y(syn_df)
        y_test = enc.transform_y(test_df)
        xgb_res = train_eval_xgboost(X_syn, y_syn, X_test, y_test, seed=sd)
        xgb_accs.append(xgb_res["acc"])
        xgb_aucs.append(xgb_res["auc"])

    def _mean_std(arr):
        arr = np.array(arr, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) if arr.size > 1 else 0.0)

    out = {}
    for k in range(1, 6):
        m, s = _mean_std(tvd_means[k])
        out[f"{k}-way_tvd_mean"] = m
        out[f"{k}-way_tvd_std_across_seeds"] = s
        out[f"{k}-way_tvd_within_subset_std_mean"] = float(np.nanmean(np.array(tvd_stds[k], dtype=float)))

    macc, sacc = _mean_std(xgb_accs)
    mauc, sauc = _mean_std(xgb_aucs)
    out["xgb_acc_mean"] = macc
    out["xgb_acc_std"] = sacc
    out["xgb_auc_mean"] = mauc
    out["xgb_auc_std"] = sauc

    out["n_quantiles"] = int(n_quantiles)
    out["tvd_subset_num"] = int(tvd_subset_num)
    out["seeds"] = list(seeds)
    out["num_cols_n"] = len(num_cols)
    out["cat_cols_n"] = len(cat_cols)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_csv", required=True, help="Real TRAIN csv (for fitting quantile bins + vocab)")
    ap.add_argument("--test_csv", required=True, help="Real TEST csv (evaluation reference)")
    ap.add_argument("--synthetic_csv", required=True, help="Synthetic csv")
    ap.add_argument("--target_col", required=True, help="Target/label column name")
    ap.add_argument("--seeds", default="0", help="Comma-separated seeds, default 0")
    ap.add_argument("--n_quantiles", type=int, default=20)
    ap.add_argument("--tvd_subset_num", type=int, default=300)
    ap.add_argument("--num_cols", default="", help="Comma-separated numeric columns (optional)")
    ap.add_argument("--cat_cols", default="", help="Comma-separated categorical columns (optional)")
    ap.add_argument("--out_json", default="", help="Save output JSON here (optional)")
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    num_cols = [c.strip() for c in args.num_cols.split(",") if c.strip()] if args.num_cols else None
    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()] if args.cat_cols else None

    res = evaluate(
        fit_csv=args.fit_csv,
        test_csv=args.test_csv,
        synthetic_csv=args.synthetic_csv,
        target_col=args.target_col,
        seeds=seeds,
        n_quantiles=args.n_quantiles,
        tvd_subset_num=args.tvd_subset_num,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    print(json.dumps(res, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()


