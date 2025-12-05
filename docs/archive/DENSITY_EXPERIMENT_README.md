# Density-Aware Pearl Experiment

## 🎯 Objective

Validate that using **experimental density maps directly** (instead of just atomic coordinates) improves Pearl's performance, especially on low-resolution structures.

## 📊 Hypothesis

**Current Pearl approach:**
- Uses atomic coordinates from PDB files (the **result** of fitting)
- Loses information from the experimental density maps

**Density-aware approach:**
- Uses experimental density maps as ground truth
- Trains model to place atoms to match density
- Should improve performance by 20-40% on low-resolution structures

## 🧪 Experimental Design

### Two Models

1. **Baseline Model** (Coordinate-only)
   - Loss: MSE between predicted and true coordinates
   - Standard approach used by Pearl

2. **Density-Aware Model** (Coordinate + Density)
   - Loss: 30% coordinate MSE + 70% density correlation
   - Generates density from predicted coordinates
   - Compares with experimental density maps

### Dataset

Your existing data:
- **6 structures** with both PDB files and CryoEM density maps
- Resolution range: 1.65Å to 3.36Å
- Includes SARS-CoV-2 proteins, kinases, proteases

| PDB ID | Resolution | Description |
|--------|-----------|-------------|
| 7BV2 | 2.5Å | SARS-CoV-2 RBD with antibody |
| 6M0J | 2.45Å | SARS-CoV-2 Spike protein |
| 7JTL | 2.04Å | SARS-CoV-2 Mpro with inhibitor |
| 7K3N | 1.65Å | SARS-CoV-2 Nsp12 with inhibitor |
| 6WHA | 3.36Å | Kinase with inhibitor |
| 7JVB | 3.29Å | Protease with substrate |

### Training

- **Epochs:** 100 (for good convergence)
- **Optimizer:** Adam (lr=1e-3)
- **Device:** GPU if available, CPU otherwise
- **Time:** ~5-10 minutes on GPU, ~30 minutes on CPU

### Evaluation Metrics

- **RMSD (Root Mean Square Deviation):** Distance between predicted and true coordinates
- **Improvement:** `(Baseline RMSD - Density RMSD) / Baseline RMSD × 100%`

## 🚀 Running the Experiment

### Option 1: Quick Run (Recommended)

```bash
bash scripts/run_density_experiment.sh
```

### Option 2: Manual Run

```bash
# Install dependencies
pip install torch numpy biopython mrcfile

# Run experiment
python scripts/train_density_aware_comparison.py
```

### Option 3: Custom Configuration

Edit `scripts/train_density_aware_comparison.py`:

```python
# Configuration
data_dir = Path("data")
output_dir = Path("results/density_aware_experiment")
num_epochs = 100  # Adjust as needed
```

## 📈 Expected Results

### Low-Resolution Structures (3-4Å)

**Hypothesis:** +20-40% improvement

| Structure | Resolution | Expected Improvement |
|-----------|-----------|---------------------|
| 6WHA | 3.36Å | **+25-35%** |
| 7JVB | 3.29Å | **+20-30%** |

### Medium-Resolution Structures (2-3Å)

**Hypothesis:** +5-15% improvement

| Structure | Resolution | Expected Improvement |
|-----------|-----------|---------------------|
| 7BV2 | 2.5Å | **+10-15%** |
| 6M0J | 2.45Å | **+10-15%** |
| 7JTL | 2.04Å | **+5-10%** |

### High-Resolution Structures (<2Å)

**Hypothesis:** +3-8% improvement

| Structure | Resolution | Expected Improvement |
|-----------|-----------|---------------------|
| 7K3N | 1.65Å | **+3-8%** |

## 📊 Understanding the Results

### Output Files

Results are saved to `results/density_aware_experiment/`:

1. **`density_aware_comparison_results.json`**
   - Per-structure RMSD values
   - Improvement percentages
   - Summary statistics

### Interpreting Results

**Good results:**
- ✅ Density-aware RMSD < Baseline RMSD (improvement > 0%)
- ✅ Larger improvements on low-resolution structures
- ✅ Overall improvement > 10%

**What if results are worse?**
- ⚠️ May need more training epochs
- ⚠️ May need to tune loss weights (coord_weight vs density_weight)
- ⚠️ May need larger grid size for density generation

### Example Output

```
EVALUATION
================================================================================

7BV2 (2.5Å):
  Baseline RMSD:      2.45 Å
  Density-aware RMSD: 2.12 Å
  Improvement:        +13.5%

6M0J (2.45Å):
  Baseline RMSD:      2.38 Å
  Density-aware RMSD: 2.05 Å
  Improvement:        +13.9%

...

SUMMARY
================================================================================
Baseline mean RMSD:      2.31 Å
Density-aware mean RMSD: 2.01 Å
Overall improvement:     +13.0%
================================================================================
```

## 🔧 Troubleshooting

### Issue: "BioPython not available"

```bash
pip install biopython
```

### Issue: "mrcfile not available"

```bash
pip install mrcfile
```

### Issue: "CUDA out of memory"

Reduce grid size in `scripts/train_density_aware_comparison.py`:

```python
self.density_loss_fn = DensityAwareLoss(
    grid_size=16,  # Reduce from 32 to 16
    voxel_size=3.0,  # Increase from 2.0 to 3.0
)
```

### Issue: "Training is too slow"

Reduce number of epochs:

```python
experiment.run(num_epochs=50)  # Reduce from 100 to 50
```

### Issue: "Results are not improving"

Try adjusting loss weights:

```python
self.density_loss_fn = DensityAwareLoss(
    coord_weight=0.5,    # Increase coordinate weight
    density_weight=0.5,  # Decrease density weight
)
```

## 📝 Next Steps After Validation

### If Results Are Positive (Expected)

1. **Scale to more structures**
   - Download more CryoEM density maps
   - Test on 100-1000 structures

2. **Integrate with full Pearl model**
   - Replace SimplePearlModel with actual Pearl architecture
   - Train on full dataset (64M structures)

3. **Optimize for production**
   - Implement efficient density compression
   - Use hybrid training (density for 10-20% of structures)
   - Deploy on 10,000 GPU cluster

### If Results Need Improvement

1. **Tune hyperparameters**
   - Adjust loss weights
   - Try different grid sizes
   - Experiment with learning rates

2. **Improve density generation**
   - Use better scattering factors
   - Add B-factor prediction
   - Implement multi-resolution grids

3. **Add more loss components**
   - Fourier shell correlation
   - Local resolution weighting
   - Gradient-based losses

## 📚 Related Documentation

- **`DENSITY_AWARE_PEARL_ARCHITECTURE.md`** - Complete architecture description
- **`DENSITY_AWARE_IMPLEMENTATION_SUMMARY.md`** - Implementation details
- **`pearl/data/density_map_loader.py`** - Density map loading code
- **`pearl/models/density_generator.py`** - Differentiable density generation
- **`pearl/training/density_aware_losses.py`** - Loss functions

## 🎉 Summary

This experiment validates a **fundamental improvement** to Pearl:

**Current approach:**
- ❌ Uses fitted coordinates (interpretation of data)
- ❌ Loses information from density maps
- ❌ Poor performance on low-resolution structures

**Density-aware approach:**
- ✅ Uses experimental density (actual measured data)
- ✅ Preserves all information
- ✅ Expected +20-40% improvement on low-resolution structures

**Run the experiment to see if the hypothesis holds!** 🚀

