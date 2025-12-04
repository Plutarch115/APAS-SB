# Pearl Data Pipeline

This module implements the complete data loading and processing pipeline for Pearl, including:

1. **PDB Data Loading**: Parse and load protein-ligand complexes from PDB files
2. **Synthetic Data Generation**: Generate diverse synthetic complexes using physics-based docking
3. **Preprocessing**: Featurize atoms, normalize coordinates, apply cropping strategies
4. **Curriculum Sampling**: Implement 5-stage curriculum training with progressive complexity

## Overview

Based on the Pearl technical report (October 2025), the data pipeline addresses the key challenge of limited experimental data by:

- **Training on diverse data mixture**: PDB structures + distillation data + synthetic data
- **Synthetic data scaling**: ~582,065 synthetic structures across 910 proteins
- **Curriculum training**: Progressive complexity from 100 atoms → unlimited
- **Multi-chain templating**: Flexible conditioning with structural priors

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install numpy torch

# For PDB parsing
pip install biopython

# For ligand handling
pip install rdkit

# Optional: for advanced docking
pip install openbabel
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Load PDB Data

```python
from pearl.data import PDBDataset

# Load protein-ligand complexes from PDB files
dataset = PDBDataset(
    pdb_dir="./pdb_files",
    release_date_cutoff="2021-09-30",  # Pearl's training cutoff
    max_protein_atoms=5000,
    max_ligand_atoms=100,
)

# Access a sample
sample = dataset[0]
print(f"PDB ID: {sample['pdb_id']}")
print(f"Protein atoms: {len(sample['protein_coords'])}")
print(f"Ligand atoms: {len(sample['ligand_coords'])}")
```

### 2. Generate Synthetic Data

```python
from pearl.data import SyntheticDataGenerator, VirtualLigandLibrary, PhysicsBasedDocker

# Create virtual ligand library
ligand_library = VirtualLigandLibrary(
    min_heavy_atoms=10,
    max_heavy_atoms=50,
)

# Setup docking engine
docker = PhysicsBasedDocker(method='random_placement')

# Generate synthetic complexes
generator = SyntheticDataGenerator(
    protein_structures=pdb_structures,
    ligand_library=ligand_library,
    docker=docker,
    n_ligands_per_protein=640,  # As in Pearl paper
)

synthetic_data = generator.generate_dataset(
    output_dir="./synthetic_data",
    max_structures=100000,
)
```

### 3. Preprocess Data

```python
from pearl.data import (
    ProteinFeaturizer,
    LigandFeaturizer,
    ComplexPreprocessor,
    CroppingStrategy,
)

# Setup preprocessing
protein_featurizer = ProteinFeaturizer(feature_dim=64)
ligand_featurizer = LigandFeaturizer(feature_dim=64)
cropping_strategy = CroppingStrategy(max_atoms=1000, strategy='pocket_centered')

preprocessor = ComplexPreprocessor(
    protein_featurizer=protein_featurizer,
    ligand_featurizer=ligand_featurizer,
    cropping_strategy=cropping_strategy,
    normalize_coords=True,
    augment=True,
)

# Preprocess complex
processed = preprocessor.preprocess(complex_data)
```

### 4. Curriculum Training

```python
from pearl.data import CurriculumSampler, DEFAULT_CURRICULUM

# Setup curriculum sampler
sampler = CurriculumSampler(
    pdb_dataset=pdb_data,
    distillation_dataset=distillation_data,
    synthetic_dataset=synthetic_data,
    curriculum=DEFAULT_CURRICULUM,
)

# Sample batches during training
for epoch in range(num_epochs):
    batch = sampler.sample_batch(batch_size=32)
    
    # Train model on batch
    loss = train_step(model, batch)
    
    # Check curriculum progress
    stats = sampler.get_statistics()
    print(f"Stage: {stats['stage']}, Progress: {stats['progress']:.1%}")
```

## Curriculum Training Stages

Pearl uses a 5-stage curriculum with progressive complexity:

| Stage | Max Atoms | Data Sources | Synthetic Ratio | Templates |
|-------|-----------|--------------|-----------------|-----------|
| 1 | 100 | PDB only | 0% | None |
| 2 | 100 | PDB + Distillation | 0% | None |
| 3 | 200 | PDB + Distillation + Synthetic | 30% | Simple |
| 4 | 500 | PDB + Distillation + Synthetic | 50% | Complex |
| 5 | 1000 | PDB + Distillation + Synthetic | 60% | Full |

## Module Structure

```
pearl/data/
├── __init__.py                 # Module exports
├── pdb_loader.py              # PDB parsing and loading
├── synthetic_generator.py     # Synthetic data generation
├── preprocessing.py           # Featurization and preprocessing
├── curriculum_sampler.py      # Curriculum-based sampling
└── README.md                  # This file
```

## Key Components

### PDBDataset

Loads protein-ligand complexes from PDB files:

- Parses PDB structures using BioPython
- Extracts protein chains and ligand residues
- Filters by release date for train/test splits
- Handles multiple chains and ligands

### SyntheticDataGenerator

Generates synthetic training data:

- **VirtualLigandLibrary**: Creates diverse drug-like molecules
- **PhysicsBasedDocker**: Docks ligands into protein pockets
- Generates ~640 ligands per protein (as in Pearl paper)
- Produces ~582K synthetic structures for full training

### Preprocessing

Featurizes and prepares data for model input:

- **ProteinFeaturizer**: Encodes amino acids, atom types, MSA features
- **LigandFeaturizer**: Encodes atoms, bonds, chemical properties
- **CroppingStrategy**: Manages large structures with pocket-centered cropping
- **ComplexPreprocessor**: Applies normalization and augmentation

### CurriculumSampler

Implements curriculum training strategy:

- Progressive complexity: 100 → 200 → 500 → 1000 atoms
- Dynamic data mixing: Increases synthetic data ratio over stages
- Template complexity: None → Simple → Complex → Full
- Automatic stage advancement based on training steps

## Data Statistics (from Pearl Paper)

### Training Data
- **PDB structures**: Released before 2021-09-30
- **Synthetic structures**: ~582,065 across 910 proteins
- **Ligands per protein**: ~640 on average
- **Tanimoto similarity**: 0.364 (diverse ligands)

### Benchmarks
- **Runs N' Poses**: 702 structures (released after 2023-06-01)
- **PoseBusters**: 297 structures (released after 2021-10-01)
- **Internal Xtals**: 111 proprietary structures

## Advanced Usage

### Custom Curriculum

```python
from pearl.data import CurriculumConfig, CurriculumStage

custom_curriculum = [
    CurriculumConfig(
        stage=CurriculumStage.STAGE_1,
        max_atoms=150,
        use_pdb=True,
        use_synthetic=False,
        synthetic_ratio=0.0,
        use_templates=False,
        template_complexity='none',
        n_steps=5000,
    ),
    # Add more stages...
]

sampler = CurriculumSampler(curriculum=custom_curriculum)
```

### Custom Docking Method

```python
class CustomDocker(PhysicsBasedDocker):
    def dock_ligand(self, protein_coords, ligand_coords, pocket_center):
        # Implement custom docking logic
        # e.g., call AutoDock Vina, GOLD, etc.
        pass

docker = CustomDocker()
generator = SyntheticDataGenerator(docker=docker, ...)
```

### Custom Featurization

```python
class CustomProteinFeaturizer(ProteinFeaturizer):
    def featurize(self, coords, atoms, residues, sequence, msa=None):
        # Add custom features
        features = super().featurize(coords, atoms, residues, sequence, msa)
        # Enhance with additional features
        return features
```

## Performance Tips

1. **Parallel Data Loading**: Use multiple workers for data loading
2. **Caching**: Cache preprocessed features to disk
3. **Batch Processing**: Process synthetic data generation in batches
4. **Memory Management**: Use cropping to limit memory usage
5. **Mixed Precision**: Store coordinates in float32

## Troubleshooting

### BioPython not found
```bash
pip install biopython
```

### RDKit not found
```bash
pip install rdkit
# or via conda:
conda install -c conda-forge rdkit
```

### PDB files not loading
- Check file format (.pdb or .ent)
- Verify file naming convention (e.g., 1ABC.pdb or pdb1abc.ent)
- Ensure files contain HETATM records for ligands

### Synthetic data generation slow
- Use simpler docking method for initial testing
- Reduce n_ligands_per_protein
- Consider distributed generation across multiple machines

## Citation

If you use this data pipeline, please cite the Pearl paper:

```bibtex
@article{pearl2025,
  title={Pearl: A Foundation Model for Placing Every Atom in the Right Location},
  author={Genesis Research Team},
  journal={Genesis Molecular AI Technical Report},
  year={2025}
}
```

## License

See main Pearl repository for license information.

## Contributing

Contributions are welcome! Please see the main Pearl repository for contribution guidelines.

