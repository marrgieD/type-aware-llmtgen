import pandas as pd
import numpy as np

train = pd.read_csv("./data/adult/splits_seed0/train.csv")
test  = pd.read_csv("./data/adult/splits_seed0/test.csv")
syn   = pd.read_csv("./synthetic_seed0_eps1_cleaned.csv")  # 用刚清洗后的

def summarize(df, name):
    df = df.copy()
    df["income"] = df["income"].astype(str)
    df["capital-gain"] = pd.to_numeric(df["capital-gain"], errors="coerce").fillna(0)
    df["hours-per-week"] = pd.to_numeric(df["hours-per-week"], errors="coerce").fillna(0)
    print("\n===", name, "===")
    for cls in ["<=50k", ">50k"]:
        sub = df[df["income"] == cls]
        if len(sub)==0:
            print(cls, "EMPTY")
            continue
        print(cls,
              "n=", len(sub),
              "cap_gain_nonzero=", float((sub["capital-gain"]>0).mean()),
              "cap_gain_mean=", float(sub["capital-gain"].mean()),
              "hours_mean=", float(sub["hours-per-week"].mean()))
    # education distribution by class (top 5)
    for cls in ["<=50k", ">50k"]:
        sub = df[df["income"] == cls]
        print("edu top5", cls, sub["education"].astype(str).value_counts().head(5).to_dict())

summarize(train, "train")
summarize(test, "test")
summarize(syn, "syn")
