# Pearl Training Workflow

Complete guide for downloading PDB data, generating synthetic data, and training Pearl.

## 🎯 Overview

This workflow demonstrates the complete Pearl training pipeline:

1. **Download PDB Data** - Get protein-ligand complexes from PDB
2. **Generate Synthetic Data** - Create synthetic training data using virtual ligands
3. **Prepare Training Data** - Preprocess and organize data for curriculum training
4. **Train Pearl** - Run curriculum-based training

## 📋 Prerequisites

### Required Dependencies

```bash
# Core dependencies
pip install numpy scipy torch biopython rdkit

# Optional (for advanced features)
pip install openbabel  # Format conversion
pip install autodock-vina  # Advanced docking
```

### Check Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import Bio; print('BioPython:', Bio.__version__)"
python -c "import rdkit; print('RDKit:', rdkit.__version__)"
```

## 🚀 Quick Start

Run all steps in sequence:

```bash
# Step 1: Download PDB data (15 structures)
python scripts/download_pdb_subset.py

# Step 2: Generate synthetic data (300 structures)
python scripts/generate_synthetic_data.py

# Step 3: Prepare training data
python scripts/prepare_training_data.py

# Step 4: Train Pearl
python scripts/train_pearl.py
```

## 📖 Detailed Workflow

### Step 1: Download PDB Data

Downloads a curated subset of high-quality protein-ligand complexes from the PDB.

```bash
python scripts/download_pdb_subset.py
```

**What it does:**
- Downloads 15 curated PDB structures with known protein-ligand complexes
- Includes diverse targets: kinases, proteases, nuclear receptors, GPCRs
- Verifies structures have both protein and ligand atoms
- Saves metadata about each structure

**Output:**
```
data/pdb_files/
├── 1ATP.pdb          # cAMP-dependent protein kinase
├── 3PY0.pdb          # CDK2 with inhibitor
├── 4HNF.pdb          # ABL kinase with imatinib
├── 1HVR.pdb          # HIV-1 protease
├── ...
└── metadata.txt      # Structure information
```

**Expected results:**
- 14-15 valid protein-ligand complexes
- Total size: ~50-100 MB
- Download time: 1-2 minutes

### Step 2: Generate Synthetic Data

Creates synthetic protein-ligand complexes using virtual ligands and physics-based docking.

```bash
python scripts/generate_synthetic_data.py
```

**What it does:**
- Creates virtual ligand library (~10 diverse drug-like molecules)
- Docks ligands into protein pockets using physics-based methods
- Generates 20 synthetic complexes per protein (configurable)
- Computes docking scores and validates structures

**Configuration:**
```python
# In scripts/generate_synthetic_data.py
n_ligands_per_protein = 20  # Increase for more data (paper uses 640)
```

**Output:**
```
data/synthetic/
├── metadata.json      # Dataset metadata
├── statistics.json    # Dataset statistics
└── structures/        # Individual synthetic structures
```

**Expected results:**
- 300 synthetic structures (15 proteins × 20 ligands)
- Generation time: 2-5 minutes
- Average docking score: ~10 ± 50

### Step 3: Prepare Training Data

Preprocesses and organizes data for curriculum-based training.

```bash
python scripts/prepare_training_data.py
```

**What it does:**
- Loads PDB and synthetic data
- Featurizes proteins and ligands
- Applies curriculum-based cropping (100, 200, 1000 atoms)
- Normalizes coordinates
- Saves preprocessed data for efficient training

**Output:**
```
data/processed/
├── manifest.json      # Training data manifest
├── stage1/
│   ├── pdb.pkl       # PDB data for stage 1 (100 atoms max)
│   └── synthetic.pkl # Synthetic data for stage 1
├── stage3/
│   ├── pdb.pkl       # PDB data for stage 3 (200 atoms max)
│   └── synthetic.pkl # Synthetic data for stage 3
└── stage5/
    ├── pdb.pkl       # PDB data for stage 5 (1000 atoms max)
    └── synthetic.pkl # Synthetic data for stage 5
```

**Expected results:**
- 3 structures per stage (PDB only in this demo)
- Processing time: 1-2 minutes
- Total preprocessed structures: 9 (3 stages × 3 structures)

### Step 4: Train Pearl

Runs curriculum-based training through multiple stages.

```bash
python scripts/train_pearl.py
```

**What it does:**
- Loads preprocessed training data
- Trains through curriculum stages (1, 3, 5)
- Logs training metrics
- Saves checkpoints after each stage

**Configuration:**
```python
# In scripts/train_pearl.py
batch_size = 2              # Batch size
n_steps_per_stage = 50      # Training steps per stage (paper uses 10K-50K)
learning_rate = 1e-4        # Learning rate
```

**Output:**
```
checkpoints/
├── pearl_stage_1.json  # Checkpoint after stage 1
├── pearl_stage_3.json  # Checkpoint after stage 3
└── pearl_stage_5.json  # Checkpoint after stage 5

logs/
└── training_log.json   # Complete training log
```

**Expected results:**
- 150 total training steps (3 stages × 50 steps)
- Training time: 1-2 minutes (demo mode)
- 3 checkpoints saved

## 📊 Data Statistics

### PDB Data
- **Structures:** 14 valid protein-ligand complexes
- **Protein atoms:** 1,000-9,000 per structure
- **Ligand atoms:** 20-100 per structure
- **Diversity:** Kinases, proteases, receptors, enzymes

### Synthetic Data
- **Structures:** 300 (15 proteins × 20 ligands)
- **Ligands per protein:** 20 (configurable)
- **Docking method:** Random placement (fast) or AutoDock Vina (accurate)
- **Validation:** Clash detection, scoring

### Preprocessed Data
- **Total structures:** 9 (3 per stage)
- **Stages:** 3 (stage 1, 3, 5)
- **Cropping:** 100, 200, 1000 atoms max
- **Features:** Protein (64-dim), Ligand (64-dim)

## 🎓 Curriculum Training

Pearl uses 5-stage curriculum training (this demo uses 3 stages):

| Stage | Max Atoms | PDB | Synthetic | Templates | Steps (Paper) |
|-------|-----------|-----|-----------|-----------|---------------|
| 1     | 100       | ✓   | ✗         | ✗         | 10,000        |
| 2     | 100       | ✓   | ✗         | ✗         | 20,000        |
| 3     | 200       | ✓   | ✓ (30%)   | Simple    | 30,000        |
| 4     | 500       | ✓   | ✓ (50%)   | Complex   | 40,000        |
| 5     | 1000      | ✓   | ✓ (60%)   | Full      | 50,000        |

**This demo uses:**
- Stages: 1, 3, 5 (simplified)
- Steps: 50 per stage (vs 10K-50K in paper)
- Batch size: 2 (vs 32-64 in paper)

## 🔧 Customization

### Increase Dataset Size

```python
# In scripts/download_pdb_subset.py
CURATED_PDB_IDS = [
    # Add more PDB IDs here
    "1ATP", "3PY0", "4HNF", ...
]

# In scripts/generate_synthetic_data.py
n_ligands_per_protein = 100  # Increase from 20
```

### Use Real Docking

```python
# In scripts/generate_synthetic_data.py
docker = PhysicsBasedDocker(method='vina')  # Instead of 'random_placement'
```

### Adjust Training

```python
# In scripts/train_pearl.py
batch_size = 32              # Larger batches
n_steps_per_stage = 10000    # More training steps
learning_rate = 1e-4         # Tune learning rate
```

## 📁 Directory Structure

```
APAS-SB/
├── scripts/
│   ├── download_pdb_subset.py      # Step 1: Download PDB data
│   ├── generate_synthetic_data.py  # Step 2: Generate synthetic data
│   ├── prepare_training_data.py    # Step 3: Prepare training data
│   └── train_pearl.py              # Step 4: Train Pearl
├── data/
│   ├── pdb_files/                  # Downloaded PDB structures
│   ├── synthetic/                  # Generated synthetic data
│   └── processed/                  # Preprocessed training data
├── checkpoints/                    # Model checkpoints
├── logs/                           # Training logs
└── pearl/                          # Pearl implementation
    ├── data/                       # Data loading modules
    ├── models/                     # Model architecture
    ├── training/                   # Training utilities
    ├── inference/                  # Inference utilities
    └── evaluation/                 # Evaluation metrics
```

## 🐛 Troubleshooting

### BioPython not installed
```bash
pip install biopython
```

### RDKit not installed
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

### PDB download fails
- Check internet connection
- Try downloading individual structures manually from https://www.rcsb.org/

### Out of memory during training
- Reduce `batch_size` in `train_pearl.py`
- Reduce `max_atoms` in curriculum stages
- Use GPU if available

## 📚 Next Steps

1. **Scale up data:** Download more PDB structures and generate more synthetic data
2. **Implement full model:** Complete Pearl model implementation with SO(3)-equivariance
3. **Add MSA features:** Integrate HHblits/MMseqs2 for MSA generation
4. **Use real docking:** Integrate AutoDock Vina for accurate docking
5. **Distributed training:** Use PyTorch DDP for multi-GPU training
6. **Evaluation:** Implement evaluation on Runs N' Poses and PoseBusters benchmarks

## 📖 References

- **Pearl Paper:** Genesis Molecular AI Technical Report (October 2025)
- **PDB:** https://www.rcsb.org/
- **RDKit:** https://www.rdkit.org/
- **BioPython:** https://biopython.org/

## ✅ Summary

You now have a complete workflow for:
- ✅ Downloading PDB data (14 structures)
- ✅ Generating synthetic data (300 structures)
- ✅ Preprocessing data for training (9 preprocessed structures)
- ✅ Training Pearl with curriculum learning (150 steps)

The pipeline is ready for scaling up to production-level training!

