import pandas as pd
import torch
from torch import nn
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
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()  # Corrected super() initialization
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, protein_features, ligand_features):
        # simple pooling
        p = protein_features.mean(dim=1)   # Shape: [B, 64]
        l = ligand_features.mean(dim=1)    # Shape: [B, 64]
        x = p + l
        return self.net(x).squeeze(-1)


# --- Execution Pipeline ---
# Ensure "data/bindingdb/tiny_bindingdb_real.csv" exists with an "affinity_value" column.
dataset = TinyBindingDBDataset("data/bindingdb/tiny_bindingdb_real.csv") 
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fetch a single batch
batch = next(iter(loader))

# Initialize model, optimizer, and loss function
model = TinyModel() 
opt = torch.optim.Adam(model.parameters(), lr=1e-3) 
loss_fn = nn.MSELoss()

# Forward pass 1
pred = model(batch["protein_features"], batch["ligand_features"]) 
loss = loss_fn(pred, batch["target"])

print("pred shape:", pred.shape) 
print("target shape:", batch["target"].shape) 
print("loss before backward:", loss.item())

# Optimization step
opt.zero_grad() 
loss.backward() 
opt.step()

# Forward pass 2 (to verify the loss goes down)
pred2 = model(batch["protein_features"], batch["ligand_features"]) 
loss2 = loss_fn(pred2, batch["target"])

print("loss after one optimizer step:", loss2.item())