"""
Quick W&B integration test - runs just a few batches.
"""

import sys
from pathlib import Path

import torch
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pearl.data.multitask_datasets import create_multitask_dataset
from pearl.training.boltz2_losses import CombinedBoltz2Loss
from pearl.models.multitask_pearl import MultiTaskPEARL
from pearl.models.mock_pearl import MockPearl

print("="*60)
print("Quick W&B Integration Test")
print("="*60)

# Initialize W&B
print("\n1. Initializing W&B...")
run = wandb.init(
    project="apas-sb-test",
    name="quick_test",
    config={'test': True},
    mode='online',
)
print(f"✅ W&B initialized: {run.url}")

# Create dataset
print("\n2. Creating dataset...")
data_dirs = {
    'chembl': Path('./data/chembl'),
    'bindingdb': Path('./data/bindingdb'),
}
dataset = create_multitask_dataset(data_dirs, split='train', use_synthetic=True)
print(f"✅ Dataset created: {len(dataset)} samples")

# Create model
print("\n3. Creating model...")
base_pearl = MockPearl(
    protein_feature_dim=64,
    ligand_feature_dim=64,
    pair_dim=128,
    trunk_blocks=2,  # Smaller for testing
    trunk_heads=4,
)
model = MultiTaskPEARL(
    base_pearl=base_pearl,
    pair_dim=128,
    hidden_dim=256,
    num_heads=4,
    freeze_pearl=False
)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Create loss and optimizer
criterion = CombinedBoltz2Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Train for a few batches
print("\n4. Training for 10 batches...")
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for i in range(10):
    # Get sample
    sample = dataset[i]
    
    # Move to device
    batch = {
        'protein_features': sample['protein_features'].unsqueeze(0).to(device),
        'ligand_features': sample['ligand_features'].unsqueeze(0).to(device),
        'target': sample['target'].unsqueeze(0).to(device),
        'task': [sample['task']],
        'weight': torch.tensor([sample['weight']]).to(device),
    }
    
    # Forward pass
    optimizer.zero_grad()
    task = batch['task'][0]
    outputs_dict = model(batch, task=task)
    outputs = outputs_dict['affinity']
    
    # Compute loss
    loss, loss_dict = criterion(
        outputs,
        batch['target'],
        batch['task'],
        batch['weight']
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Log to W&B
    wandb.log({
        'train/loss': loss.item(),
        'train/batch': i,
    }, step=i)
    
    print(f"   Batch {i}: loss={loss.item():.4f}")

print("\n✅ Training complete!")

# Save checkpoint
print("\n5. Saving checkpoint...")
checkpoint_path = Path('./test_checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'test': True,
}, checkpoint_path)

# Upload as artifact
artifact = wandb.Artifact(
    name='quick-test-checkpoint',
    type='model',
    description='Quick test checkpoint'
)
artifact.add_file(str(checkpoint_path))
wandb.log_artifact(artifact)
print(f"✅ Checkpoint saved and uploaded")

# Cleanup
checkpoint_path.unlink()

# Finish
wandb.finish()
print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print(f"\nView your run at: {run.url}")

