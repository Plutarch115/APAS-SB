import pandas as pd
import numpy as np

path = "data/bindingdb/BindingDB_All.tsv"


df = pd.read_csv(path, sep="\t", nrows=1000, low_memory=False)


df = df[
    df["Ligand SMILES"].notna() &
    df["Target Name"].notna() &
    df["BindingDB Target Chain Sequence 1"].notna() &
    df["Ki (nM)"].notna()
].copy()


df["Ki (nM)"] = pd.to_numeric(df["Ki (nM)"], errors="coerce")
df = df[df["Ki (nM)"].notna()]
df = df[df["Ki (nM)"] > 0]


#convert nM -> uM, then take log10

df["affinity_value"] = np.log10(df["Ki (nM)"] / 1000.0)


tiny = df[[
    "Ligand SMILES",
    "Target Name",
    "BindingDB Target Chain Sequence 1",
    "Ki (nM)",
    "affinity_value"
]].head(20).copy()


print("usable rows:", len(tiny))
print(tiny.head(10))


tiny.to_csv("data/bindingdb/tiny_bindingdb_real.csv", index=False)
print("\nsaved: data/bindingdb/tiny_bindingdb_real.csv")