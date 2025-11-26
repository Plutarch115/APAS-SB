# Pearl Quick Start Guide

This guide will help you get started with Pearl for protein-ligand cofolding.

## Installation

```bash
cd pearl
pip install -r requirements.txt
```

## Running the Example

The simplest way to understand Pearl is to run the example script:

```bash
cd pearl/examples
python simple_example.py
```

This will demonstrate:
1. Unconditional cofolding (sequence + topology only)
2. Conditional cofolding (with pocket information)
3. Metrics calculation

## Basic Usage

### 1. Initialize the Model

```python
from pearl.models.pearl import Pearl

model = Pearl(
    protein_feature_dim=64,
    ligand_feature_dim=64,
    pair_dim=128,
    trunk_blocks=4,
    diffusion_blocks=8,
    num_diffusion_steps=200
)
```

### 2. Unconditional Cofolding

Predict structure from sequence and topology alone:

```python
from pearl.inference.unconditional import unconditional_cofolding
import torch

# Prepare features (these would come from your data pipeline)
protein_features = torch.randn(1, 100, 64)  # [batch, n_protein_atoms, feat_dim]
ligand_features = torch.randn(1, 20, 64)    # [batch, n_ligand_atoms, feat_dim]

# Generate structures
structures = unconditional_cofolding(
    model=model,
    protein_features=protein_features,
    ligand_features=ligand_features,
    num_samples=20,  # Generate 20 samples for best@5 evaluation
    device='cuda'
)

# structures shape: [20, 120, 3] (20 samples, 120 atoms, 3D coords)
```

### 3. Conditional Cofolding

Use structural priors (e.g., known binding pocket):

```python
from pearl.inference.conditional import conditional_cofolding
from pearl.models.templating import Template

# Create a pocket template
pocket_template = Template(
    protein_coords=torch.randn(20, 3),      # Pocket residue coordinates
    protein_features=torch.randn(20, 64),   # Pocket residue features
    confidence=0.9
)

# Generate structures with pocket guidance
structures = conditional_cofolding(
    model=model,
    protein_features=protein_features,
    ligand_features=ligand_features,
    pocket_template=pocket_template,
    num_samples=20,
    device='cuda'
)
```

### 4. Evaluate Predictions

```python
from pearl.evaluation.metrics import compute_ligand_rmsd, MetricsCalculator

# Assuming you have ground truth
true_protein_coords = torch.randn(100, 3)
true_ligand_coords = torch.randn(20, 3)

# Compute RMSD for each sample
for i, structure in enumerate(structures):
    pred_protein = structure[0, :100]  # First 100 atoms are protein
    pred_ligand = structure[0, 100:]   # Remaining atoms are ligand
    
    rmsd = compute_ligand_rmsd(
        pred_ligand, true_ligand_coords,
        pred_protein, true_protein_coords,
        symmetry_correction=True
    )
    print(f"Sample {i+1} RMSD: {rmsd:.2f} Å")

# Use metrics calculator for comprehensive evaluation
calculator = MetricsCalculator(
    rmsd_thresholds=[1.0, 2.0],
    compute_lddt=True
)

metrics = calculator.compute_all_metrics(
    pred_ligand, true_ligand_coords,
    pred_protein, true_protein_coords
)

print(f"Ligand RMSD: {metrics['ligand_rmsd']:.2f} Å")
print(f"Success (RMSD < 2Å): {metrics['success_rmsd<2.0']}")
print(f"lDDT-PLI: {metrics['lddt_pli']:.1f}")
```

## Training

### Basic Training Loop

```python
from pearl.training.trainer import PearlTrainer
from torch.utils.data import DataLoader

# Initialize trainer
trainer = PearlTrainer(
    model=model,
    learning_rate=1e-4,
    use_mixed_precision=True,
    device='cuda'
)

# Prepare your dataloader (you need to implement this)
# Each batch should contain:
# - protein_features: [batch, n_protein, feat_dim]
# - ligand_features: [batch, n_ligand, feat_dim]
# - coordinates: [batch, n_atoms, 3]
# - mask: [batch, n_atoms] (optional)
# - bond_indices: [n_bonds, 2] (optional)

train_loader = DataLoader(your_dataset, batch_size=4, shuffle=True)

# Train for one epoch
losses = trainer.train_epoch(train_loader, log_interval=100)

print(f"Average loss: {losses['total']:.4f}")
```

### Curriculum Training

```python
from pearl.training.trainer import CurriculumScheduler

scheduler = CurriculumScheduler()

for stage_idx in range(5):
    stage_config = scheduler.get_current_stage()
    print(f"Training {stage_config['name']}")
    
    # Adjust your dataloader based on stage_config
    # - max_atoms: Maximum number of atoms
    # - use_templates: Whether to use templates
    # - use_synthetic: Whether to include synthetic data
    
    for epoch in range(stage_config['epochs']):
        losses = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}: Loss = {losses['total']:.4f}")
    
    # Save checkpoint
    trainer.save_checkpoint(f'checkpoint_stage{stage_idx}.pt')
    
    # Advance to next stage
    if not scheduler.is_final_stage():
        scheduler.advance_stage()
```

## Advanced Usage

### Multiple Templates

```python
from pearl.inference.conditional import conditional_cofolding_with_multiple_templates

# Create multiple templates (e.g., different conformations)
templates = [
    Template(protein_coords=..., protein_features=..., confidence=0.9),
    Template(protein_coords=..., protein_features=..., confidence=0.8),
    Template(protein_coords=..., protein_features=..., confidence=0.7)
]

structures = conditional_cofolding_with_multiple_templates(
    model=model,
    protein_features=protein_features,
    ligand_features=ligand_features,
    templates=templates,
    num_samples=20
)
```

### Using Apo Structure

```python
from pearl.inference.conditional import conditional_cofolding_from_apo_structure

# Start from unbound (apo) structure
structures = conditional_cofolding_from_apo_structure(
    model=model,
    protein_features=protein_features,
    ligand_features=ligand_features,
    apo_structure_coords=apo_coords,
    apo_structure_features=apo_features,
    num_samples=20
)
```

### Best@k Evaluation

```python
from pearl.evaluation.metrics import compute_best_at_k

# Generate multiple samples for multiple structures
all_samples_rmsds = []

for structure_idx in range(num_test_structures):
    # Generate samples
    structures = unconditional_cofolding(...)
    
    # Compute RMSD for each sample
    rmsds = []
    for structure in structures:
        rmsd = compute_ligand_rmsd(...)
        rmsds.append(rmsd)
    
    all_samples_rmsds.append(rmsds)

# Compute best@5 success rate
success_rate = compute_best_at_k(
    all_samples_rmsds,
    k=5,
    threshold=2.0  # RMSD < 2Å
)

print(f"Best@5 success rate: {success_rate:.1f}%")
```

## Tips and Best Practices

### 1. Feature Preparation

- **Protein features**: Should encode sequence information, secondary structure, and optionally MSA
- **Ligand features**: Should encode atom types, bonds, and molecular properties
- Both should be normalized to similar scales

### 2. Memory Management

- Use gradient accumulation for large models:
  ```python
  trainer = PearlTrainer(
      model=model,
      gradient_accumulation_steps=4  # Effective batch size = 4x
  )
  ```

- Enable mixed precision to reduce memory:
  ```python
  trainer = PearlTrainer(
      model=model,
      use_mixed_precision=True  # Uses bfloat16
  )
  ```

### 3. Sampling

- Generate 20 samples for best@5 evaluation (standard in the paper)
- Use fewer diffusion steps for faster inference (trade-off with quality)
- Consider using guidance techniques for better control

### 4. Evaluation

- Always use symmetry-corrected RMSD for ligands
- Report both RMSD < 2Å and RMSD < 1Å thresholds
- Include PoseBusters validation for physical plausibility
- Use best@k protocol instead of confidence-based selection

## Common Issues

### Out of Memory

- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Reduce model size (fewer blocks, smaller dimensions)

### Slow Training

- Enable mixed precision
- Use gradient accumulation to increase effective batch size
- Consider using multiple GPUs (requires additional setup)

### Poor Performance

- Check feature normalization
- Verify data preprocessing
- Ensure sufficient training data diversity
- Use curriculum training
- Include synthetic data

## Next Steps

1. Implement your data pipeline for PDB structures
2. Prepare protein and ligand feature extractors
3. Set up distributed training for large-scale experiments
4. Implement full PoseBusters validation
5. Add confidence scoring for pose selection

## Resources

- **Paper**: Genesis Molecular AI Technical Report (October 2025)
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md` for details
- **Examples**: Check `pearl/examples/` for more examples

## Support

For issues and questions:
1. Check the implementation summary
2. Review the example scripts
3. Examine the inline code documentation
4. Refer to the original Pearl technical report

