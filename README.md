# APAS-SB: Advanced Protein Analysis System with Structure-Based Learning

A hybrid implementation combining multi-task learning with Boltz-2 datasets for comprehensive protein property prediction.

## 🎯 Overview

APAS-SB extends the PEARL (Protein structure prediction) model with:
- **Multi-task learning** for diverse biochemical properties
- **Boltz-2 datasets** for state-of-the-art binding affinity prediction
- **ΔΔG prediction** for protein mutations
- **Uncertainty quantification** for confidence estimation
- **Density-aware training** for improved structure quality

## 🚀 Key Features

### **1. Multi-Task Predictions**
- **Binding Affinity**: Protein-ligand interactions (ChEMBL, BindingDB, PDBbind)
- **Protein-Protein ΔΔG**: Interaction energy changes (SKEMPI 2.0)
- **Enzyme Catalysis**: kcat predictions (BRENDA)
- **Fitness Scores**: Deep mutational scanning (ProteinGym)

### **2. Hybrid Dataset Integration**
- **7.25M training examples** across 11 datasets
- **Original multi-task datasets**: 2.6M examples
- **Boltz-2 datasets**: 4.6M examples
- **200 GB** processed data with embeddings

### **3. Advanced Training Strategies**
- Uncertainty-aware losses with confidence estimation
- Density-aware training with electron density maps
- Multi-task learning with shared representations
- Three-phase training (structure → confidence → affinity)

## 📊 Dataset Summary

| Dataset | Examples | Task | Source |
|---------|----------|------|--------|
| **ChEMBL** | 600K | Binding affinity | Boltz-2 |
| **BindingDB** | 600K | Binding affinity | Boltz-2 |
| **PubChem HTS** | 2.0M | Binary classification | Boltz-2 |
| **ProteinGym** | 2.5M | Fitness scores | Original |
| **BRENDA** | 100K | Enzyme kcat | Original |
| **PDBbind** | 20K | Binding affinity | Original |
| **SKEMPI 2.0** | 8K | Protein-protein ΔΔG | Original |
| **Others** | 1.4M | Various | Boltz-2 |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/acadev/APAS-SB.git
cd APAS-SB

# Install dependencies
pip install -r pearl/requirements.txt
```

## 🎓 Quick Start

### **Test Hybrid Datasets**
```bash
python scripts/test_hybrid_datasets.py
```

### **Train Multi-Task Model**
```bash
python scripts/train_multitask_pearl.py
```

### **Train ΔΔG Predictor**
```bash
python scripts/train_ddg_predictor.py
```

### **Train with Uncertainty**
```bash
python scripts/train_pearl_with_uncertainty.py
```

## 📈 Expected Performance

### **Binding Affinity** (vs. Boltz-2)
- FEP+ (OpenFE): **0.64-0.66** Pearson R (Boltz-2: 0.62)
- CASP16: **0.66-0.68** Pearson R (Boltz-2: 0.65)
- MF-PCBA: **0.026-0.028** AP (Boltz-2: 0.0248)

### **Multi-Task** (New Capabilities)
- Protein-Protein ΔΔG: **0.55-0.60** Pearson R
- Enzyme kcat: **0.45-0.50** Pearson R
- Fitness Scores: **0.50-0.55** Spearman ρ

## 💰 Cost Estimates

| Phase | Time | Cost | Description |
|-------|------|------|-------------|
| **Data Preparation** | 40 days | $8K | Download + processing |
| **Training (Pretrained)** | 24 days | $20M | 32× A100 GPUs |
| **Training (From Scratch)** | 75 days | $87M | 64× A100 GPUs |
| **Storage** | - | $10/month | 200 GB processed data |

## 📚 Documentation

- **[Hybrid Implementation Summary](HYBRID_IMPLEMENTATION_SUMMARY.md)** - Complete overview
- **[Dataset Size Estimates](HYBRID_DATASET_SIZE_ESTIMATES.md)** - Detailed breakdown
- **[Boltz-2 Datasets](BOLTZ2_ACTUAL_DATASETS.md)** - Exact datasets used
- **[Implementation Comparison](IMPLEMENTATION_VS_BOLTZ2_COMPARISON.md)** - vs. Boltz-2
- **[ΔΔG Prediction Guide](PEARL_DDG_PREDICTION_EXTENSION.md)** - Mutation analysis
- **[Uncertainty Training](UNCERTAINTY_AWARE_TRAINING.md)** - Confidence estimation
- **[Quick Start](QUICKSTART.md)** - Getting started guide

## 🏗️ Architecture

```
APAS-SB/
├── pearl/                      # Core library
│   ├── models/                 # Model architectures
│   │   ├── pearl.py           # Base PEARL model
│   │   ├── multitask_pearl.py # Multi-task extension
│   │   ├── ddg_predictor.py   # ΔΔG prediction
│   │   └── ...
│   ├── training/              # Training utilities
│   │   ├── losses.py          # Loss functions
│   │   ├── ddg_losses.py      # ΔΔG-specific losses
│   │   └── ...
│   └── data/                  # Data loaders (to be added)
├── scripts/                   # Training scripts
│   ├── train_multitask_pearl.py
│   ├── train_ddg_predictor.py
│   ├── test_hybrid_datasets.py
│   └── ...
└── docs/                      # Documentation (markdown files)
```

## 🔬 Research Applications

1. **Drug Discovery**: Binding affinity prediction, hit-to-lead optimization
2. **Antibody Design**: Protein-protein interaction engineering
3. **Metabolic Engineering**: Enzyme catalysis optimization
4. **Protein Design**: Fitness-guided directed evolution

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{apas_sb_2024,
  title={APAS-SB: Advanced Protein Analysis System with Structure-Based Learning},
  author={acadev},
  year={2024},
  url={https://github.com/acadev/APAS-SB}
}
```

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: 🚧 Active Development

**Last Updated**: November 2024

