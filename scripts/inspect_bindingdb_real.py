import pandas as pd

path = "data/bindingdb/BindingDB_All.tsv"


#read a tiny chunk first

df = pd.read_csv(path, sep="\t", nrows=100, low_memory=False)


print("rows loaded:", len(df))
print("\nimportant columns present:")
wanted = [
    "Ligand SMILES",
    "Target Name",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "BindingDB Target Chain Sequence 1",
    "PDB ID(s) for Ligand-Target Complex",
]
for col in wanted:
    print(col, "->", col in df.columns)


print("\nnon-null counts:")
for col in wanted:
    if col in df.columns:
        print(col, ":", df[col].notna().sum())


print("\nexample rows:")
cols_to_show = [c for c in wanted if c in df.columns]
print(df[cols_to_show].head(5))