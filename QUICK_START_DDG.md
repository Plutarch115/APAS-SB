# Quick Start Guide: ΔΔG Prediction with Ensemble PEARL

## 🚀 Getting Started in 5 Minutes

This guide shows you how to quickly get started with ΔΔG prediction using Ensemble PEARL.

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-org/ensemble-pearl.git
cd ensemble-pearl

# Install dependencies
pip install torch numpy scipy matplotlib seaborn pandas biopython

# Download pre-trained model
wget https://your-model-host.com/pearl_with_ddg.pt
```

---

## 🎯 Basic Usage

### Example 1: Predict ΔΔG for a Single Mutation

```python
from ddg_implementation_guide import PEARLWithDDG, predict_ddg
import torch

# Load pre-trained model
model = PEARLWithDDG.from_pretrained('pearl_with_ddg.pt')
model.eval()

# Load structures
wt_structure = load_structure('protein_wt.pdb', ligand='ligand.sdf')
mut_structure = load_structure('protein_L99A.pdb', ligand='ligand.sdf')

# Predict ΔΔG
result = predict_ddg(model, wt_structure, mut_structure)

print(f"Predicted ΔΔG: {result.ddg:.2f} ± {result.confidence:.2f} kcal/mol")
print(f"Interpretation: {'Weaker binding' if result.ddg > 0 else 'Stronger binding'}")
```

**Output:**
```
Predicted ΔΔG: +2.3 ± 0.5 kcal/mol
Interpretation: Weaker binding
```

---

### Example 2: Screen Multiple Mutations

```python
from ddg_implementation_guide import PEARLWithDDG
import pandas as pd

# Load model
model = PEARLWithDDG.from_pretrained('pearl_with_ddg.pt')

# Define mutations to test
mutations = [
    'L99A', 'L99V', 'L99I',  # Hydrophobic substitutions
    'L99S', 'L99T',          # Polar substitutions
    'L99K', 'L99R'           # Charged substitutions
]

# Predict ΔΔG for all mutations
results = []
for mutation in mutations:
    mut_structure = generate_mutant(wt_structure, mutation)
    result = predict_ddg(model, wt_structure, mut_structure)
    
    results.append({
        'mutation': mutation,
        'ddg': result.ddg,
        'confidence': result.confidence,
        'interpretation': 'Favorable' if result.ddg < -0.5 else 'Unfavorable' if result.ddg > 0.5 else 'Neutral'
    })

# Create dataframe
df = pd.DataFrame(results)
df = df.sort_values('ddg')

print(df)
```

**Output:**
```
  mutation   ddg  confidence interpretation
0     L99I -1.2         0.4      Favorable
1     L99V -0.8         0.5      Favorable
2     L99A +2.3         0.5    Unfavorable
3     L99S +3.1         0.6    Unfavorable
4     L99T +3.5         0.7    Unfavorable
5     L99K +5.2         0.8    Unfavorable
6     L99R +5.8         0.9    Unfavorable
```

---

### Example 3: Visualize Per-Residue Contributions

```python
from ddg_visualization import plot_residue_contributions

# Predict ΔΔG with per-residue contributions
result = predict_ddg(model, wt_structure, mut_structure)

# Plot contributions
fig = plot_residue_contributions(
    structure_id='1ABC_L99A',
    residue_contrib=result.residue_contributions,
    sequence=wt_structure['sequence'],
    mutation_site=99,
    save_path='residue_contributions.png'
)
```

**Output:** Bar plot showing which residues contribute to ΔΔG

---

### Example 4: Generate Correlation Plot

```python
from ddg_visualization import plot_ddg_correlation
import numpy as np

# Load test set predictions
test_results = load_test_results('test_predictions.csv')

# Generate correlation plot
fig = plot_ddg_correlation(
    pred_ddg=test_results['pred_ddg'].values,
    exp_ddg=test_results['exp_ddg'].values,
    dataset_name='PDBbind Test Set',
    confidence=test_results['confidence'].values,
    save_path='ddg_correlation.png'
)
```

**Output:** Scatter plot with Pearson R, RMSE, MAE

---

## 🔬 Advanced Usage

### Example 5: Batch Prediction on GPU

```python
import torch
from torch.utils.data import DataLoader

# Create dataset
dataset = DDGDataset(data_points, data_weights)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Predict on GPU
model = model.cuda()
model.eval()

all_predictions = []
with torch.no_grad():
    for batch in loader:
        outputs = model(
            wt_input=batch['wt_input'].cuda(),
            mut_input=batch['mut_input'].cuda()
        )
        all_predictions.append(outputs['ddg'].cpu().numpy())

predictions = np.concatenate(all_predictions)
```

---

### Example 6: Uncertainty-Guided Filtering

```python
# Predict ΔΔG for many mutations
results = predict_many_mutations(model, wt_structure, mutations)

# Filter by confidence
high_confidence = [r for r in results if r.confidence < 1.0]  # < 1 kcal/mol uncertainty
low_confidence = [r for r in results if r.confidence >= 1.0]

print(f"High confidence predictions: {len(high_confidence)}")
print(f"Low confidence predictions: {len(low_confidence)}")

# Only use high-confidence predictions for decision-making
for result in high_confidence:
    if result.ddg < -1.0:  # Strong improvement
        print(f"Recommend testing mutation: {result.mutation}")
```

---

### Example 7: Combine with Experimental Data

```python
from scipy.stats import pearsonr

# Load experimental ΔΔG values
exp_data = load_experimental_data('skempi2_subset.csv')

# Predict ΔΔG for same mutations
predictions = []
for idx, row in exp_data.iterrows():
    wt_structure = load_structure(row['wt_pdb'])
    mut_structure = load_structure(row['mut_pdb'])
    result = predict_ddg(model, wt_structure, mut_structure)
    predictions.append(result.ddg)

# Compute correlation
r, p_value = pearsonr(predictions, exp_data['exp_ddg'])
print(f"Correlation with experiment: R = {r:.3f} (p = {p_value:.2e})")

# Plot
from ddg_visualization import plot_ddg_correlation
fig = plot_ddg_correlation(
    pred_ddg=np.array(predictions),
    exp_ddg=exp_data['exp_ddg'].values,
    dataset_name='SKEMPI 2.0 Subset'
)
```

---

## 📊 Complete Workflow Example

### Drug Discovery: Optimize Ligand Binding

```python
"""
Goal: Optimize a lead compound by predicting ΔΔG for R-group modifications
"""

from ddg_implementation_guide import PEARLWithDDG, predict_ddg
from ddg_visualization import plot_ddg_correlation
import pandas as pd

# Step 1: Load model and structures
model = PEARLWithDDG.from_pretrained('pearl_with_ddg.pt')
protein = load_protein('kinase_target.pdb')
lead_compound = load_ligand('lead_compound.sdf')

# Step 2: Generate ligand variants
r_groups = [
    'methyl', 'ethyl', 'propyl',           # Alkyl
    'phenyl', 'benzyl',                     # Aromatic
    'hydroxyl', 'amino', 'carboxyl'         # Polar
]

ligand_variants = []
for r_group in r_groups:
    variant = modify_ligand(lead_compound, position=5, new_group=r_group)
    ligand_variants.append({
        'name': f'Lead_{r_group}',
        'ligand': variant,
        'r_group': r_group
    })

# Step 3: Predict ΔΔG for all variants
results = []
wt_structure = create_complex(protein, lead_compound)

for variant in ligand_variants:
    mut_structure = create_complex(protein, variant['ligand'])
    prediction = predict_ddg(model, wt_structure, mut_structure)
    
    results.append({
        'name': variant['name'],
        'r_group': variant['r_group'],
        'ddg': prediction.ddg,
        'confidence': prediction.confidence,
        'predicted_improvement': -prediction.ddg  # Negative ΔΔG = improvement
    })

# Step 4: Rank by predicted improvement
df = pd.DataFrame(results)
df = df.sort_values('predicted_improvement', ascending=False)

print("Top 3 predicted improvements:")
print(df.head(3))

# Step 5: Visualize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in df['predicted_improvement']]
ax.barh(df['name'], df['predicted_improvement'], color=colors, alpha=0.7)
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Predicted Improvement (kcal/mol)', fontsize=14)
ax.set_ylabel('Ligand Variant', fontsize=14)
ax.set_title('Predicted Binding Affinity Changes', fontsize=16)
plt.tight_layout()
plt.savefig('ligand_optimization.png', dpi=300)

# Step 6: Select candidates for synthesis
candidates = df[
    (df['predicted_improvement'] > 1.0) &  # > 1 kcal/mol improvement
    (df['confidence'] < 0.8)                # High confidence
]

print(f"\nRecommended for synthesis: {len(candidates)} compounds")
for idx, row in candidates.iterrows():
    print(f"  - {row['name']}: ΔΔG = {row['ddg']:.2f} ± {row['confidence']:.2f} kcal/mol")
```

**Output:**
```
Top 3 predicted improvements:
              name  r_group   ddg  confidence  predicted_improvement
4      Lead_benzyl   benzyl -2.1         0.6                    2.1
3       Lead_phenyl   phenyl -1.8         0.5                    1.8
1        Lead_ethyl    ethyl -1.2         0.4                    1.2

Recommended for synthesis: 2 compounds
  - Lead_benzyl: ΔΔG = -2.1 ± 0.6 kcal/mol
  - Lead_phenyl: ΔΔG = -1.8 ± 0.5 kcal/mol
```

---

## 🎯 Common Use Cases

### 1. Protein Engineering

```python
# Find stabilizing mutations
mutations = generate_all_single_mutations(protein_sequence)
results = screen_mutations(model, protein, mutations)
stabilizing = [r for r in results if r.ddg < -1.0]
```

### 2. Antibody Optimization

```python
# Optimize CDR regions
cdr_mutations = generate_cdr_mutations(antibody, cdr_region='H3')
results = screen_mutations(model, antibody, cdr_mutations)
improved = [r for r in results if r.ddg < -0.5 and r.confidence < 1.0]
```

### 3. Resistance Prediction

```python
# Identify resistance mutations
resistance_mutations = ['T790M', 'L858R', 'C797S']  # Known resistance
results = predict_resistance(model, kinase, drug, resistance_mutations)
resistant = [r for r in results if r.ddg > 2.0]  # > 2 kcal/mol loss
```

### 4. Virtual Screening

```python
# Screen large compound library
library = load_compound_library('enamine_10M.sdf')
predictions = batch_predict_ddg(model, protein, library, batch_size=1000)
top_hits = predictions.nsmallest(100, 'ddg')  # Top 100 binders
```

---

## 📊 Interpreting Results

### ΔΔG Interpretation Guide

| ΔΔG (kcal/mol) | Interpretation | Action |
|----------------|----------------|--------|
| < -2.0 | Strong improvement | **High priority for testing** |
| -2.0 to -1.0 | Moderate improvement | Test if resources allow |
| -1.0 to -0.5 | Slight improvement | Consider in combination |
| -0.5 to +0.5 | No significant change | Neutral |
| +0.5 to +1.0 | Slight decrease | Avoid unless other benefits |
| +1.0 to +2.0 | Moderate decrease | **Do not test** |
| > +2.0 | Strong decrease | **Definitely avoid** |

### Confidence Interpretation

| Confidence (kcal/mol) | Interpretation | Action |
|-----------------------|----------------|--------|
| < 0.5 | Very high confidence | Trust prediction |
| 0.5 - 1.0 | High confidence | Generally reliable |
| 1.0 - 1.5 | Moderate confidence | Use with caution |
| 1.5 - 2.0 | Low confidence | Verify experimentally |
| > 2.0 | Very low confidence | **Do not trust** |

### Combined Decision Matrix

| ΔΔG | Confidence | Decision |
|-----|------------|----------|
| < -2.0 | < 1.0 | ✅ **Strongly recommend** |
| -2.0 to -1.0 | < 1.0 | ✅ Recommend |
| -1.0 to -0.5 | < 0.5 | ⚠️ Consider |
| Any | > 1.5 | ⚠️ Verify experimentally |
| > +1.0 | < 1.0 | ❌ **Do not test** |

---

## 🐛 Troubleshooting

### Issue 1: Low Correlation with Experiment

**Symptoms:** Pearson R < 0.5

**Possible causes:**
1. Model not trained on similar data
2. Experimental data quality issues
3. Structure quality issues

**Solutions:**
```python
# Check structure quality
check_structure_quality(wt_structure)
check_structure_quality(mut_structure)

# Check if mutation is in training distribution
check_mutation_coverage(model, mutation)

# Try ensemble prediction (average multiple runs)
predictions = [predict_ddg(model, wt, mut) for _ in range(10)]
avg_ddg = np.mean([p.ddg for p in predictions])
```

### Issue 2: High Uncertainty

**Symptoms:** Confidence > 2.0 kcal/mol

**Possible causes:**
1. Mutation far from training distribution
2. Large structural changes
3. Poor structure quality

**Solutions:**
```python
# Check mutation type
if is_unusual_mutation(mutation):
    print("Warning: Unusual mutation type")

# Check structural change magnitude
coord_rmsd = compute_rmsd(wt_coords, mut_coords)
if coord_rmsd > 5.0:
    print("Warning: Large structural change")

# Use MD-based refinement
refined_structure = run_md_refinement(mut_structure)
result = predict_ddg(model, wt_structure, refined_structure)
```

### Issue 3: Slow Inference

**Symptoms:** > 10 seconds per prediction

**Solutions:**
```python
# Use GPU
model = model.cuda()

# Use batch prediction
results = batch_predict_ddg(model, wt_structure, mut_structures, batch_size=32)

# Use mixed precision
with torch.cuda.amp.autocast():
    result = predict_ddg(model, wt_structure, mut_structure)
```

---

## 📚 Next Steps

1. **Read full documentation:** `DDG_PREDICTION_SUMMARY.md`
2. **Study implementation:** `ddg_implementation_guide.py`
3. **Explore visualization:** `ddg_visualization.py`
4. **Run examples:** `examples/ddg_prediction_examples.ipynb`
5. **Train your own model:** `training/train_ddg_model.py`

---

## 🎉 Summary

You now know how to:
- ✅ Predict ΔΔG for single mutations
- ✅ Screen multiple mutations
- ✅ Visualize results
- ✅ Interpret predictions
- ✅ Use in drug discovery workflows
- ✅ Troubleshoot common issues

**Happy predicting!** 🚀

---

## 📞 Support

- **Documentation:** See `DDG_PREDICTION_SUMMARY.md`
- **Issues:** Open an issue on GitHub
- **Questions:** Contact the team

**Let's revolutionize structure-based drug design!** 💊🧬

