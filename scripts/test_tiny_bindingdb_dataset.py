import pandas as pd
import torch
from torch.utils.data import Dataset

class TinyBindingDBDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = row["BindingDB Target Chain Sequence 1"]
        smiles = row["Ligand SMILES"]
        affinity = float(row["affinity_value"])

        # placeholder numeric features for now
        protein_features = torch.randn(100, 64)
        ligand_features = torch.randn(30, 64)

        return {
            "protein_features": protein_features,
            "ligand_features": ligand_features,
            "target": torch.tensor(affinity, dtype=torch.float32),
            "task": "binding_affinity",
            "data_source": "bindingdb_real",
            "sequence": seq,
            "smiles": smiles,
        }

# Example usage (Assuming the file path is correct)
dataset = TinyBindingDBDataset("data/bindingdb/tiny_bindingdb_real.csv")
print("dataset size:", len(dataset))

sample = dataset[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, v)