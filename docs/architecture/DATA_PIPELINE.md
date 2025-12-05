# Pearl Data Pipeline Implementation

Complete implementation of Pearl's data loading and synthetic data generation pipeline based on the technical report (October 2025).

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Usage Examples](#usage-examples)
5. [Curriculum Training](#curriculum-training)
6. [Synthetic Data Generation](#synthetic-data-generation)
7. [Performance Considerations](#performance-considerations)

## Overview

Pearl's data pipeline addresses the fundamental challenge of limited experimental data in protein-ligand structure prediction through three key innovations:

### Key Features

1. **PDB Data Loading**
   - Parse protein-ligand complexes from PDB files
   - Extract protein chains, ligand residues, and coordinates
   - Filter by release date (training cutoff: 2021-09-30)
   - Handle multiple chains and ligands

2. **Synthetic Data Generation**
   - Generate ~582,065 synthetic structures across 910 proteins
   - Create diverse virtual ligands (~640 per protein)
   - Physics-based docking for realistic poses
   - Tanimoto similarity: 0.364 (high diversity)

3. **Curriculum Training**
   - 5-stage progressive training
   - Atom limits: 100 → 200 → 500 → 1000 → unlimited
   - Synthetic data ratio: 0% → 30% → 50% → 60%
   - Template complexity: None → Simple → Complex → Full

4. **Preprocessing & Featurization**
   - Protein: amino acids, atom types, MSA features
   - Ligand: atoms, bonds, chemical properties
   - Pocket-centered cropping for large structures
   - Data augmentation (rotation, translation)

## Architecture

```
pearl/data/
│
├── pdb_loader.py              # PDB parsing and dataset
│   ├── PDBParser              # Parse PDB files
│   ├── PDBDataset             # PyTorch-style dataset
│   └── PDBDataLoader          # Batch loading
│
├── synthetic_generator.py     # Synthetic data generation
│   ├── VirtualLigandLibrary   # Generate diverse ligands
│   ├── PhysicsBasedDocker     # Dock ligands into pockets
│   └── SyntheticDataGenerator # Orchestrate generation
│
├── preprocessing.py           # Featurization and preprocessing
│   ├── ProteinFeaturizer      # Encode protein features
│   ├── LigandFeaturizer       # Encode ligand features
│   ├── CroppingStrategy       # Manage large structures
│   └── ComplexPreprocessor    # Complete preprocessing
│
└── curriculum_sampler.py      # Curriculum training
    ├── CurriculumSampler      # Sample with curriculum
    ├── CurriculumConfig       # Stage configuration
    ├── DataMixer              # Mix data sources
    └── TemplateSelector       # Select templates
```

## Components

### 1. PDB Data Loading

#### PDBParser

Parses PDB files and extracts protein-ligand complexes:

```python
from pearl.data import PDBParser

parser = PDBParser(pdb_dir="./pdb_files")
structure = parser.parse_structure("1ABC")

# Extract components
protein_chains = parser.extract_protein_chains(structure)
ligands = parser.extract_ligands(structure)
sequence = parser.get_chain_sequence(protein_chains[0])
```

**Features:**
- BioPython-based parsing
- Automatic protein chain detection
- Ligand extraction (excludes solvents/ions)
- Sequence extraction

#### PDBDataset

PyTorch-style dataset for batch loading:

```python
from pearl.data import PDBDataset

dataset = PDBDataset(
    pdb_dir="./pdb_files",
    pdb_ids=["1ABC", "2XYZ"],  # Optional: specific IDs
    max_protein_atoms=5000,
    max_ligand_atoms=100,
    release_date_cutoff="2021-09-30",
)

# Access samples
sample = dataset[0]
# Returns: {pdb_id, protein_coords, protein_atoms, protein_sequence,
#           ligand_coords, ligand_atoms, ligand_bonds}
```

### 2. Synthetic Data Generation

#### VirtualLigandLibrary

Generates diverse drug-like molecules:

```python
from pearl.data import VirtualLigandLibrary

library = VirtualLigandLibrary(
    smiles_file="ligands.smi",  # Optional: load from file
    min_heavy_atoms=10,
    max_heavy_atoms=50,
    diversity_threshold=0.3,
)

# Sample ligands
ligand = library.sample_ligand()
coords = library.get_ligand_coords(ligand)
```

**Features:**
- RDKit-based molecule generation
- Lipinski's rule of five filtering
- 3D coordinate generation
- Diversity filtering

#### PhysicsBasedDocker

Docks ligands into protein pockets:

```python
from pearl.data import PhysicsBasedDocker

docker = PhysicsBasedDocker(method='random_placement')

docked_coords, score = docker.dock_ligand(
    protein_coords=protein_coords,
    ligand_coords=ligand_coords,
    pocket_center=pocket_center,
    pocket_radius=10.0,
)
```

**Methods:**
- `random_placement`: Fast random docking (default)
- `vina`: AutoDock Vina integration (requires installation)
- Custom: Extend for other docking tools

#### SyntheticDataGenerator

Orchestrates synthetic dataset generation:

```python
from pearl.data import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    protein_structures=pdb_structures,
    ligand_library=ligand_library,
    docker=docker,
    n_ligands_per_protein=640,  # Pearl paper value
)

synthetic_data = generator.generate_dataset(
    output_dir="./synthetic_data",
    max_structures=582065,  # Pearl paper value
)
```

**Output:**
- Generates ~582K structures (Pearl paper)
- ~640 ligands per protein
- Saves metadata and statistics

### 3. Preprocessing

#### ProteinFeaturizer

Encodes protein features:

```python
from pearl.data import ProteinFeaturizer

featurizer = ProteinFeaturizer(feature_dim=64)

features = featurizer.featurize(
    coords=protein_coords,
    atoms=atom_list,
    residues=residue_indices,
    sequence=amino_acid_sequence,
    msa=msa_features,  # Optional
)
```

**Features:**
- Amino acid vocabulary (20 + unknown)
- Atom type encoding
- Residue indexing
- MSA features (optional)

#### LigandFeaturizer

Encodes ligand features:

```python
from pearl.data import LigandFeaturizer

featurizer = LigandFeaturizer(feature_dim=64)

features = featurizer.featurize(
    coords=ligand_coords,
    atoms=atom_list,
    bonds=bond_list,  # Optional: auto-inferred
)
```

**Features:**
- Extended atom vocabulary (C, N, O, S, P, F, Cl, Br, I)
- Bond type encoding (single, double, triple, aromatic)
- Formal charges
- Aromaticity

#### CroppingStrategy

Manages large structures:

```python
from pearl.data import CroppingStrategy

cropper = CroppingStrategy(
    max_atoms=1000,
    strategy='pocket_centered',  # or 'random', 'full'
)

cropped_coords, cropped_atoms, cropped_residues = cropper.crop_protein(
    protein_coords=protein_coords,
    ligand_coords=ligand_coords,
    protein_atoms=protein_atoms,
    protein_residues=protein_residues,
)
```

**Strategies:**
- `pocket_centered`: Select atoms closest to ligand (recommended)
- `random`: Random sampling
- `full`: No cropping

### 4. Curriculum Training

#### CurriculumSampler

Implements 5-stage curriculum:

```python
from pearl.data import CurriculumSampler, DEFAULT_CURRICULUM

sampler = CurriculumSampler(
    pdb_dataset=pdb_data,
    distillation_dataset=distillation_data,
    synthetic_dataset=synthetic_data,
    curriculum=DEFAULT_CURRICULUM,
)

# Training loop
for epoch in range(num_epochs):
    batch = sampler.sample_batch(batch_size=32)
    
    # Train model
    loss = train_step(model, batch)
    
    # Monitor progress
    stats = sampler.get_statistics()
    print(f"Stage: {stats['stage']}, Progress: {stats['progress']:.1%}")
```

## Usage Examples

### Complete Pipeline

```python
from pearl.data import (
    PDBDataset,
    SyntheticDataGenerator,
    VirtualLigandLibrary,
    PhysicsBasedDocker,
    ComplexPreprocessor,
    ProteinFeaturizer,
    LigandFeaturizer,
    CroppingStrategy,
    CurriculumSampler,
)

# 1. Load PDB data
pdb_dataset = PDBDataset(
    pdb_dir="./pdb_files",
    release_date_cutoff="2021-09-30",
)

# 2. Generate synthetic data
ligand_library = VirtualLigandLibrary()
docker = PhysicsBasedDocker()
generator = SyntheticDataGenerator(
    protein_structures=[pdb_dataset[i] for i in range(len(pdb_dataset))],
    ligand_library=ligand_library,
    docker=docker,
    n_ligands_per_protein=640,
)
synthetic_data = generator.generate_dataset("./synthetic_data")

# 3. Setup preprocessing
preprocessor = ComplexPreprocessor(
    protein_featurizer=ProteinFeaturizer(feature_dim=64),
    ligand_featurizer=LigandFeaturizer(feature_dim=64),
    cropping_strategy=CroppingStrategy(max_atoms=1000),
    normalize_coords=True,
    augment=True,
)

# 4. Setup curriculum training
sampler = CurriculumSampler(
    pdb_dataset=[pdb_dataset[i] for i in range(len(pdb_dataset))],
    synthetic_dataset=synthetic_data,
)

# 5. Training loop
for epoch in range(num_epochs):
    batch = sampler.sample_batch(batch_size=32)
    
    # Preprocess batch
    processed_batch = [preprocessor.preprocess(sample) for sample in batch]
    
    # Train model
    loss = train_step(model, processed_batch)
```

## Curriculum Training

### Stage Configuration

| Stage | Steps | Max Atoms | PDB | Distillation | Synthetic | Synthetic % | Templates |
|-------|-------|-----------|-----|--------------|-----------|-------------|-----------|
| 1 | 10K | 100 | ✓ | ✗ | ✗ | 0% | None |
| 2 | 20K | 100 | ✓ | ✓ | ✗ | 0% | None |
| 3 | 30K | 200 | ✓ | ✓ | ✓ | 30% | Simple |
| 4 | 40K | 500 | ✓ | ✓ | ✓ | 50% | Complex |
| 5 | 50K | 1000 | ✓ | ✓ | ✓ | 60% | Full |

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
```

## Performance Considerations

### Memory Management

- Use cropping to limit structure size
- Store coordinates in float32
- Batch processing for synthetic generation
- Cache preprocessed features

### Speed Optimization

- Parallel data loading (num_workers > 0)
- Vectorized operations (NumPy/PyTorch)
- Simple docking for initial testing
- Distributed synthetic generation

### Disk Usage

- PDB files: ~1-10 MB each
- Synthetic dataset: ~50-100 GB for 582K structures
- Preprocessed features: ~10-20 GB (cached)

## Next Steps

1. **Set up PDB directory**: Download structures from RCSB PDB
2. **Generate synthetic data**: Run synthetic data generation
3. **Integrate with training**: Connect to Pearl training loop
4. **Run curriculum training**: Train through all 5 stages
5. **Evaluate**: Test on Runs N' Poses, PoseBusters benchmarks

## References

- Pearl Technical Report (October 2025)
- Genesis Molecular AI
- PDB: https://www.rcsb.org/
- RDKit: https://www.rdkit.org/
- BioPython: https://biopython.org/

