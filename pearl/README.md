# Pearl: Placing Every Atom in the Right Location

Implementation of the Pearl foundation model for protein-ligand cofolding based on the Genesis Molecular AI technical report.

## Overview

Pearl is a state-of-the-art generative foundation model for protein-ligand structure prediction that addresses key challenges in computational drug discovery:

1. **Large-scale synthetic data training** to overcome data scarcity
2. **SO(3)-equivariant diffusion module** for improved generalization and sample efficiency
3. **Multi-chain templating system** supporting both protein and non-polymeric components
4. **Dual inference modes**: unconditional and pocket-conditional cofolding

## Key Features

- SO(3)-equivariant transformer blocks for 3D rotational symmetry
- Lightweight triangle multiplication trunk module
- Five-stage curriculum training
- Mixed precision (bfloat16/fp32) training
- PoseBusters validation for physical plausibility
- Best@k evaluation protocol

## Architecture

```
Input (Sequence + Ligand Topology)
    ↓
Multi-Chain Templating
    ↓
Trunk Module (Triangle Multiplication)
    ↓
SO(3)-Equivariant Diffusion Module
    ↓
3D Structure Prediction
```

## Project Structure

```
pearl/
├── models/
│   ├── trunk.py              # Trunk module with triangle multiplication
│   ├── equivariant.py        # SO(3)-equivariant transformer blocks
│   ├── diffusion.py          # Diffusion module
│   ├── templating.py         # Multi-chain templating system
│   └── pearl.py              # Main Pearl model
├── data/
│   ├── pdb_loader.py         # PDB data loading
│   ├── synthetic.py          # Synthetic data generation
│   └── preprocessing.py      # Data preprocessing utilities
├── training/
│   ├── curriculum.py         # Five-stage curriculum training
│   ├── losses.py             # Loss functions
│   └── trainer.py            # Training loop
├── inference/
│   ├── unconditional.py      # Unconditional cofolding
│   ├── conditional.py        # Pocket-conditional cofolding
│   └── sampling.py           # Sampling strategies
├── evaluation/
│   ├── metrics.py            # RMSD, lDDT-PLI metrics
│   └── posebusters.py        # PoseBusters validation
└── utils/
    ├── geometry.py           # 3D geometry utilities
    └── mixed_precision.py    # Mixed precision training utilities
```

## Installation

```bash
pip install torch numpy scipy biopython
```

## Usage

### Training

```python
from pearl.models.pearl import Pearl
from pearl.training.trainer import PearlTrainer

model = Pearl()
trainer = PearlTrainer(model)
trainer.train()
```

### Inference

```python
# Unconditional cofolding
from pearl.inference.unconditional import unconditional_cofolding

poses = unconditional_cofolding(
    protein_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O"
)

# Conditional cofolding with pocket information
from pearl.inference.conditional import conditional_cofolding

poses = conditional_cofolding(
    protein_sequence="...",
    ligand_smiles="...",
    pocket_residues=[73, 99, 101, 114]  # Known binding pocket
)
```

## Performance

Pearl achieves state-of-the-art results on multiple benchmarks:

- **Runs N' Poses**: 85.2% success rate (RMSD < 2Å & PB-valid)
- **PoseBusters**: 84.7% success rate (RMSD < 2Å & PB-valid)
- **Internal Xtals**: 62.0% success rate (RMSD < 2Å & PB-valid)

## Citation

```bibtex
@article{pearl2025,
  title={Pearl: A Foundation Model for Placing Every Atom in the Right Location},
  author={Genesis Research Team},
  journal={Genesis Molecular AI Technical Report},
  year={2025}
}
```

## License

This is an implementation based on the published technical report. Please refer to the original paper for licensing information.

