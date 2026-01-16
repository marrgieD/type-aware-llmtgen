#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# -----------------------------
# Metadata schema
# -----------------------------
@dataclass
class NumMeta:
    min: float
    max: float
    kind: str  # "int" or "float"

@dataclass
class CatMeta:
    categories: List[str]

@dataclass
class ColMeta:
    name: str
    dtype: str  # "num" or "cat"
    num: Optional[NumMeta] = None
    cat: Optional[CatMeta] = None


def build_metadata_from_csv(
    csv_path: str,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    n_preview: int = 200000,
) -> List[ColMeta]:
    """
    Build metadata (feature names, categorical names, numeric ranges).
    Assumption: these are non-sensitive general knowledge.
    """
    df = pd.read_csv(csv_path, nrows=n_preview)
    cols = df.columns.tolist()

    cat_cols = set(cat_cols or [])
    num_cols = set(num_cols or [])

    metas: List[ColMeta] = []

    # If user did not provide col types, infer by "numeric parse-ability"
    for c in cols:
        if c in cat_cols:
            s = df[c].astype(str).fillna("?")
            cats = sorted(s.unique().tolist())
            metas.append(ColMeta(name=c, dtype="cat", cat=CatMeta(categories=cats)))
            continue
        if c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            lo = float(np.nanmin(s.values))
            hi = float(np.nanmax(s.values))
            # Heuristic: if values look integral
            kind = "int" if np.all(np.isclose(s.dropna().values, np.round(s.dropna().values))) else "float"
            metas.append(ColMeta(name=c, dtype="num", num=NumMeta(min=lo, max=hi, kind=kind)))
            continue

        # infer
        s_num = pd.to_numeric(df[c], errors="coerce")
        frac_numeric = np.mean(~np.isnan(s_num.to_numpy(dtype=float)))
        if frac_numeric >= 0.95:
            lo = float(np.nanmin(s_num.values))
            hi = float(np.nanmax(s_num.values))
            kind = "int" if np.all(np.isclose(s_num.dropna().values, np.round(s_num.dropna().values))) else "float"
            metas.append(ColMeta(name=c, dtype="num", num=NumMeta(min=lo, max=hi, kind=kind)))
        else:
            s = df[c].astype(str).fillna("?")
            cats = sorted(s.unique().tolist())
            metas.append(ColMeta(name=c, dtype="cat", cat=CatMeta(categories=cats)))

    return metas


def _to_float_array(x: np.ndarray) -> np.ndarray:
    """Convert array with possible '?' / non-numeric to float, fill NaNs with median."""
    s = pd.to_numeric(pd.Series(x), errors="coerce")  # '?' -> NaN
    arr = s.to_numpy(dtype=float)
    if np.isnan(arr).any():
        med = np.nanmedian(arr)
        if np.isnan(med):  # all NaN edge case
            med = 0.0
        arr = np.where(np.isnan(arr), med, arr)
    return arr


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank -> [0,1] (ties averaged)."""
    r = pd.Series(x).rank(pct=True).to_numpy(dtype=float)
    # rank(pct=True) is in (0,1]; shift to [0,1)
    r = np.clip(r, 1e-6, 1.0) - 1e-6
    return r


def _clip_num(vals: np.ndarray, lo: float, hi: float, kind: str) -> np.ndarray:
    vals = np.clip(vals, lo, hi)
    if kind == "int":
        vals = np.round(vals).astype(int)
    return vals


def inject_dependencies(
    df: pd.DataFrame,
    metas: List[ColMeta],
    rng: np.random.Generator,
    num_rules: int = 6,
) -> pd.DataFrame:
    """
    Inject random dependencies among features while staying within:
    - correct categorical names (must be from category list)
    - numerically valid ranges (clip to [min,max])
    Safe with missing symbol '?' (treated as NaN for numeric ops).
    """
    cols = [m.name for m in metas]
    if len(cols) < 2 or num_rules <= 0:
        return df

    # Prefer a random order to reduce cycles (DAG-ish)
    order = cols.copy()
    rng.shuffle(order)

    meta_map: Dict[str, ColMeta] = {m.name: m for m in metas}

    for _ in range(num_rules):
        i = int(rng.integers(0, len(order) - 1))
        j = int(rng.integers(i + 1, len(order)))
        src = order[i]
        tgt = order[j]

        sm = meta_map[src]
        tm = meta_map[tgt]

        src_vals = df[src].to_numpy(dtype=object)

        # NUM -> NUM
        if sm.dtype == "num" and tm.dtype == "num":
            lo, hi, kind = tm.num.min, tm.num.max, tm.num.kind
            src_num = _to_float_array(src_vals)
            u = _rank01(src_num)

            alpha = float(rng.uniform(0.6, 1.4))
            noise = rng.normal(0.0, 0.08, size=len(df))
            v = lo + (hi - lo) * np.clip(alpha * u + noise, 0.0, 1.0)
            df[tgt] = _clip_num(v, lo, hi, kind).astype(object)

        # CAT -> CAT
        elif sm.dtype == "cat" and tm.dtype == "cat":
            src_cats = sm.cat.categories
            tgt_cats = tm.cat.categories

            src_codes = pd.Categorical(df[src].astype(str), categories=src_cats).codes
            perm = rng.permutation(len(tgt_cats))
            tgt_codes = perm[(src_codes % len(tgt_cats))]
            df[tgt] = np.array(tgt_cats, dtype=object)[tgt_codes]

        # NUM -> CAT
        elif sm.dtype == "num" and tm.dtype == "cat":
            tgt_cats = tm.cat.categories
            src_num = _to_float_array(src_vals)
            u = _rank01(src_num)

            K = min(8, max(2, len(tgt_cats)))
            bins = np.floor(u * K).astype(int)
            bins = np.clip(bins, 0, K - 1)

            out = np.empty(len(df), dtype=object)
            for b in range(K):
                idx = np.where(bins == b)[0]
                if len(idx) == 0:
                    continue
                start = int(np.floor(b * len(tgt_cats) / K))
                end = int(np.floor((b + 1) * len(tgt_cats) / K))
                end = max(end, start + 1)
                pool = tgt_cats[start:end]
                out[idx] = rng.choice(pool, size=len(idx), replace=True)
            df[tgt] = out

        # CAT -> NUM
        elif sm.dtype == "cat" and tm.dtype == "num":
            lo, hi, kind = tm.num.min, tm.num.max, tm.num.kind
            src_cats = sm.cat.categories
            src_codes = pd.Categorical(df[src].astype(str), categories=src_cats).codes

            ncat = max(1, len(src_cats))
            centers = lo + (hi - lo) * (np.arange(ncat) + 0.5) / ncat
            noise = rng.normal(0.0, 0.12 * (hi - lo), size=len(df))
            v = centers[np.clip(src_codes, 0, ncat - 1)] + noise
            df[tgt] = _clip_num(v, lo, hi, kind).astype(object)

    return df


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_csv", required=True, help="CSV used to extract feature names/categories/ranges")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num_samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_rules", type=int, default=6, help="number of random dependency rules to inject")
    ap.add_argument("--missing_rate", type=float, default=0.0)
    ap.add_argument("--cat_cols", type=str, default="", help="comma-separated categorical cols (optional)")
    ap.add_argument("--num_cols", type=str, default="", help="comma-separated numeric cols (optional)")
    ap.add_argument("--preview_rows", type=int, default=200000)
    args = ap.parse_args()

    cat_cols = [c.strip() for c in args.cat_cols.split(",") if c.strip()]
    num_cols = [c.strip() for c in args.num_cols.split(",") if c.strip()]

    metas = build_metadata_from_csv(
        args.schema_csv,
        cat_cols=cat_cols if cat_cols else None,
        num_cols=num_cols if num_cols else None,
        n_preview=args.preview_rows,
    )

    print("Metadata summary:")
    for m in metas:
        if m.dtype == "num":
            print(f"  [NUM] {m.name:20s} range=({m.num.min:.4g},{m.num.max:.4g}) kind={m.num.kind}")
        else:
            print(f"  [CAT] {m.name:20s} |cats|={len(m.cat.categories)}")

    df = generate_random_tabular_data(
        metas=metas,
        num_samples=args.num_samples,
        seed=args.seed,
        num_rules=args.num_rules,
        missing_rate=args.missing_rate,
    )
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
