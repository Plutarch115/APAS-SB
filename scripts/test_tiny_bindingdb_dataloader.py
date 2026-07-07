import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TinyBindingDBDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "protein_features": torch.randn(100, 64),
            "ligand_features": torch.randn(30, 64),
            "target": torch.tensor(float(row["affinity_value"]), dtype=torch.float32),
            "task": "binding_affinity",
            "data_source": "bindingdb_real",
        }

# --- Execution ---
# Note: Ensure "data/bindingdb/tiny_bindingdb_real.csv" exists with an "affinity_value" column.
dataset = TinyBindingDBDataset("data/bindingdb/tiny_bindingdb_real.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(loader))
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {v}")