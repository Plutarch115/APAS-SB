# Uncertainty-Aware Training for Pearl

## 🎯 Overview

This document addresses a critical limitation in the original Pearl implementation: **it doesn't account for experimental uncertainty in structural data**.

### The Problem

1. **Variable Structure Quality**: Different PDB structures have different resolutions (1.0-10.0 Å)
2. **Variable Local Quality**: Within a structure, some regions are well-resolved while others are uncertain
3. **CryoEM Heterogeneity**: CryoEM structures have spatially-varying local resolution
4. **Uniform Weighting**: Original Pearl treats all atoms equally, regardless of confidence

### The Solution

We've implemented **uncertainty-aware training** that:
- ✅ Extracts B-factors (X-ray) and local resolution (CryoEM) from experimental data
- ✅ Converts uncertainty to per-atom confidence scores
- ✅ Weights loss by confidence: high-confidence atoms contribute more
- ✅ Stratifies training by resolution bins
- ✅ Optionally learns to predict its own uncertainty

## 📊 Experimental Uncertainty Sources

### 1. B-Factors (Temperature Factors)

**What they are:**
- Atomic displacement parameters from X-ray crystallography
- Measure thermal motion and static disorder
- Units: Å² (square Angstroms)

**Interpretation:**
- **Low B-factor** (< 20 Å²): Well-ordered, high confidence
- **Medium B-factor** (20-50 Å²): Moderate confidence
- **High B-factor** (> 50 Å²): Disordered, low confidence

**Where found:**
- PDB files: B-factor column (columns 61-66)
- mmCIF files: `_atom_site.B_iso_or_equiv`

### 2. Local Resolution (CryoEM)

**What it is:**
- Spatially-varying resolution across CryoEM maps
- Some regions (e.g., core) are better resolved than others (e.g., flexible loops)
- Units: Å (Angstroms)

**Interpretation:**
- **Good resolution** (< 3 Å): High confidence
- **Moderate resolution** (3-5 Å): Medium confidence
- **Poor resolution** (> 5 Å): Low confidence

**Where found:**
- Local resolution maps (MRC format) from EMDB
- Computed using tools like ResMap, MonoRes, or Bsoft

### 3. Overall Resolution

**What it is:**
- Global quality metric for the entire structure
- X-ray: Based on diffraction limit
- CryoEM: Based on Fourier Shell Correlation (FSC)

**Interpretation:**
- **High resolution** (< 2 Å): Excellent quality
- **Medium resolution** (2-4 Å): Good quality
- **Low resolution** (> 4 Å): Moderate to poor quality

## 🔧 Implementation

### 1. Extract Experimental Metadata

```python
from pearl.data.experimental_metadata import ExperimentalMetadataExtractor

# Initialize extractor
extractor = ExperimentalMetadataExtractor()

# Extract from PDB file (X-ray or CryoEM)
uncertainty = extractor.extract_from_pdb(
    pdb_file="data/pdb_files/1ATP.pdb",
    local_resolution_map=None,  # Optional: path to CryoEM local resolution map
)

# Get confidence scores [0, 1] for each atom
confidence = uncertainty.atom_confidence  # [n_atoms]

# Get loss weights (inverse variance weighting)
weights = uncertainty.get_loss_weights(weighting_scheme='inverse_variance')

print(f"Method: {uncertainty.method}")  # 'xray', 'em', or 'nmr'
print(f"Resolution: {uncertainty.resolution} Å")
print(f"Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
```

### 2. Use Uncertainty-Aware Loss

```python
from pearl.training.uncertainty_aware_losses import UncertaintyWeightedDiffusionLoss
import torch

# Create loss function
loss_fn = UncertaintyWeightedDiffusionLoss(
    weighting_scheme='inverse_variance',  # Weight by 1/sigma^2
    min_weight=0.1,  # Minimum weight for uncertain atoms
    resolution_scaling=True,  # Scale by overall resolution
)

# In training loop
predicted_noise = model(...)  # [batch, n_atoms, 3]
true_noise = ...  # [batch, n_atoms, 3]
confidence = ...  # [batch, n_atoms] - from experimental data
resolution = ...  # [batch] - overall resolution per structure
mask = ...  # [batch, n_atoms] - valid atom mask

loss = loss_fn(
    predicted_noise=predicted_noise,
    true_noise=true_noise,
    confidence=confidence,
    resolution=resolution,
    mask=mask,
)
```

### 3. Resolution-Stratified Training

```python
from pearl.training.uncertainty_aware_losses import ResolutionStratifiedLoss

# Create stratified loss
loss_fn = ResolutionStratifiedLoss(
    resolution_bins=[0.0, 2.0, 3.0, 4.0, 10.0],  # Bin edges (Å)
    bin_weights=[1.0, 1.0, 1.0, 1.0],  # Equal weight per bin
)

# Compute loss
losses = loss_fn(
    predicted_noise=predicted_noise,
    true_noise=true_noise,
    resolution=resolution,
    mask=mask,
)

print(f"Total loss: {losses['total']:.4f}")
print(f"High-res (0-2Å): {losses['bin_0_(0.0-2.0A)']:.4f}")
print(f"Medium-res (2-3Å): {losses['bin_1_(2.0-3.0A)']:.4f}")
print(f"Low-res (3-4Å): {losses['bin_2_(3.0-4.0A)']:.4f}")
```

### 4. Combined Uncertainty-Aware Training

```python
from pearl.training.uncertainty_aware_losses import CombinedUncertaintyAwareLoss

# Create combined loss
loss_fn = CombinedUncertaintyAwareLoss(
    base_weight=1.0,
    resolution_stratify=True,  # Enable resolution stratification
    learn_uncertainty=False,  # Optionally learn predicted uncertainty
)

# Compute loss
losses = loss_fn(
    predicted_noise=predicted_noise,
    true_noise=true_noise,
    confidence=confidence,
    resolution=resolution,
    mask=mask,
)

print(f"Total loss: {losses['total']:.4f}")
print(f"Main loss: {losses['main']:.4f}")
print(f"Stratified losses: {[k for k in losses.keys() if 'stratified' in k]}")
```

## 📈 Expected Benefits

### 1. Better Generalization
- Model learns to focus on high-confidence regions
- Doesn't overfit to uncertain/noisy regions
- Better performance on diverse test sets

### 2. Resolution-Aware Learning
- Learns from both high and low resolution structures
- Doesn't bias toward high-resolution structures
- Better performance across resolution ranges

### 3. CryoEM Integration
- Can now train on CryoEM structures effectively
- Accounts for variable local resolution
- Expands training data significantly

### 4. Improved Confidence Estimates
- Model's predictions are better calibrated
- Can provide uncertainty estimates for predictions
- Useful for downstream applications

## 🧪 Experimental Validation

### Recommended Experiments

#### 1. Ablation Study
Compare performance with and without uncertainty weighting:

```python
# Baseline (no uncertainty weighting)
baseline_loss = DiffusionLoss()

# Uncertainty-aware
uncertainty_loss = UncertaintyWeightedDiffusionLoss()

# Train both and compare on test set
```

#### 2. Resolution Stratification
Evaluate performance across resolution bins:

```python
# Evaluate on high-resolution test set (< 2Å)
high_res_rmsd = evaluate(model, high_res_test_set)

# Evaluate on low-resolution test set (> 3Å)
low_res_rmsd = evaluate(model, low_res_test_set)

# Compare: uncertainty-aware should improve low-res performance
```

#### 3. CryoEM Structures
Train with and without CryoEM data:

```python
# Train on X-ray only
model_xray = train(xray_dataset)

# Train on X-ray + CryoEM (with uncertainty weighting)
model_combined = train(xray_dataset + cryoem_dataset, use_uncertainty=True)

# Compare on CryoEM test set
```

## 📊 Data Statistics

### PDB Data Distribution

**X-ray Crystallography:**
- Structures: ~180,000 (as of 2024)
- Resolution range: 0.5-10.0 Å
- Typical B-factors: 10-80 Å²
- High-resolution (< 2Å): ~30%
- Medium-resolution (2-3Å): ~50%
- Low-resolution (> 3Å): ~20%

**CryoEM:**
- Structures: ~20,000 (rapidly growing)
- Resolution range: 1.5-20.0 Å
- Local resolution variation: ±2-5 Å within structure
- High-resolution (< 3Å): ~40%
- Medium-resolution (3-5Å): ~40%
- Low-resolution (> 5Å): ~20%

### Impact on Training

**Without uncertainty weighting:**
- High-resolution structures dominate loss
- Model overfits to well-resolved regions
- Poor performance on uncertain regions

**With uncertainty weighting:**
- Balanced contribution across resolutions
- Model learns robust features
- Better generalization

## 🔬 Advanced Features

### 1. Learned Uncertainty

The model can learn to predict its own uncertainty:

```python
from pearl.training.uncertainty_aware_losses import AdaptiveUncertaintyLoss

loss_fn = AdaptiveUncertaintyLoss(
    learn_uncertainty=True,
    uncertainty_weight=0.1,
)

# Model must output both prediction and uncertainty
predicted_noise, predicted_uncertainty = model(...)

losses = loss_fn(
    predicted_noise=predicted_noise,
    true_noise=true_noise,
    predicted_uncertainty=predicted_uncertainty,  # log(sigma^2)
    experimental_confidence=confidence,
    mask=mask,
)
```

### 2. Custom Weighting Schemes

```python
# Linear weighting
weights = confidence

# Squared weighting (emphasize high-confidence)
weights = confidence ** 2

# Sigmoid weighting (smooth transition)
weights = 1.0 / (1.0 + np.exp(-5 * (confidence - 0.5)))

# Inverse variance (statistical optimal)
weights = confidence ** 2  # Assuming confidence ~ 1/sigma
```

### 3. CryoEM Local Resolution Maps

```python
from pearl.data.experimental_metadata import CryoEMLocalResolution

# Load local resolution map
cryoem = CryoEMLocalResolution()
resolution_map = cryoem.load_local_resolution_map("local_resolution.mrc")

# Interpolate at atom positions
atom_resolutions = cryoem.interpolate_atom_resolution(
    atom_coords=coords,
    resolution_map=resolution_map,
    voxel_size=1.0,  # Å per voxel
    origin=np.array([0, 0, 0]),
)

# Convert to confidence
confidence = cryoem.resolution_to_confidence(atom_resolutions)
```

## 📚 References

### B-Factors and Uncertainty
- Trueblood et al. (1996) "Atomic Displacement Parameter Nomenclature"
- Cruickshank (1999) "Remarks about protein structure precision"

### CryoEM Local Resolution
- Kucukelbir et al. (2014) "Quantifying the local resolution of cryo-EM density maps"
- Vilas et al. (2018) "MonoRes: Automatic and Accurate Estimation of Local Resolution"

### Uncertainty in Deep Learning
- Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty"

## ✅ Summary

**Key Improvements:**
1. ✅ Extracts B-factors and resolution from PDB files
2. ✅ Converts to per-atom confidence scores
3. ✅ Implements uncertainty-weighted loss functions
4. ✅ Supports resolution stratification
5. ✅ Enables CryoEM data integration
6. ✅ Optionally learns predicted uncertainty

**Expected Impact:**
- Better generalization across resolutions
- Improved performance on CryoEM structures
- More robust predictions
- Better calibrated confidence estimates

**Next Steps:**
1. Integrate into training pipeline
2. Run ablation studies
3. Evaluate on diverse test sets
4. Compare with baseline Pearl

