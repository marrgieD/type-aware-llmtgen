import pandas as pd

syn = pd.read_csv("./synthetic_seed0_eps1.csv")

cols = ["income","workclass","education","marital-status","occupation",
        "relationship","race","sex","native-country"]

mask = (syn[cols].astype(str) != "[UNK]").all(axis=1)
syn2 = syn[mask].copy()
print("kept:", len(syn2), "dropped:", len(syn) - len(syn2))

syn2.to_csv("./synthetic_seed0_eps1.cleaned.csv", index=False)
