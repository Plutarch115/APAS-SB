# Implementation vs. Boltz-2: What We Built vs. What They Actually Used

## 🎯 Executive Summary

After analyzing the actual Boltz-2 paper, I discovered that our initial implementation was based on **general knowledge of common datasets** in the field, but **Boltz-2 used a different, more focused set of datasets**. This document compares what we built vs. what they actually used.

---

## 📊 Dataset Comparison

### **What We Initially Implemented** ❌

| Dataset | Status | Reason |
|---------|--------|--------|
| **PDBbind** | ❌ Not in Boltz-2 | We assumed this was used (common in the field) |
| **SKEMPI 2.0** | ❌ Not in Boltz-2 | We assumed protein-protein ΔΔG was included |
| **BRENDA** | ❌ Not in Boltz-2 | We assumed kcat prediction was a goal |
| **ProteinGym** | ❌ Not in Boltz-2 | We assumed fitness scores were included |

### **What Boltz-2 Actually Used** ✅

| Dataset | Status | Size in Boltz-2 |
|---------|--------|------------------|
| **ChEMBL (v34)** | ✅ Used | 1.2M binders (with BindingDB) |
| **BindingDB** | ✅ Used | 1.2M binders (with ChEMBL) |
| **PubChem HTS** | ✅ Used | 200K binders, 1.8M decoys |
| **PubChem Small Assays** | ✅ Used | 10K binders, 50K decoys |
| **CeMM Fragments** | ✅ Used | 25K binders, 115K decoys |
| **MIDAS Metabolites** | ✅ Used | 2K binders, 20K decoys |
| **Synthetic Decoys** | ✅ Used | 1.2M decoys |

---

## 🔍 Key Differences

### 1. **Scope: Multi-Task vs. Single-Task**

**Our Implementation**:
- Multi-task learning with 4 different tasks:
  1. Binding affinity (protein-ligand)
  2. ΔΔG (protein-protein interactions)
  3. kcat (catalytic activity)
  4. Fitness scores (deep mutational scanning)

**Boltz-2 Reality**:
- **Single focus**: Binding affinity prediction only
- Two sub-tasks:
  1. Continuous affinity values (Ki, Kd, IC50, etc.)
  2. Binary classification (binder vs. decoy)
- **No kcat, no protein-protein ΔΔG, no fitness scores**

### 2. **Data Sources**

**Our Implementation**:
```python
# We created 4 dataset classes
class PDBBindDataset(Dataset): ...
class SKEMPI2Dataset(Dataset): ...
class BRENDADataset(Dataset): ...
class ProteinGymDataset(Dataset): ...
```

**Boltz-2 Reality**:
```python
# They used 6 different datasets
class ChEMBLDataset(Dataset): ...
class BindingDBDataset(Dataset): ...
class PubChemHTSDataset(Dataset): ...
class PubChemSmallAssaysDataset(Dataset): ...
class CeMMFragmentsDataset(Dataset): ...
class MIDASDataset(Dataset): ...
class SyntheticDecoysDataset(Dataset): ...
```

### 3. **Training Strategy**

**Our Implementation**:
- Multi-task loss with task-specific weights
- MSE loss for continuous values
- Uncertainty-aware NLL loss
- Task routing based on data source

**Boltz-2 Reality**:
- **Huber loss** (δ=0.5) for continuous values
- **Pairwise intra-assay differences loss** (weight 0.9)
- **Absolute affinity loss** (weight 0.1)
- **Focal loss** (γ=1) for binary classification
- **Specific sampling weights** for each dataset (see table below)

### 4. **Sampling Weights**

**Our Implementation**:
```python
task_weights = {
    'binding_affinity': 1.0,
    'ddg_ppi': 1.0,
    'kcat': 1.0,
    'fitness': 1.0
}
```

**Boltz-2 Reality**:
```python
sampling_weights = {
    'chembl_bindingdb': 0.25,
    'pubchem_small_values': 0.005,
    'pubchem_hts': 0.44,  # Highest weight!
    'pubchem_small_binary': 0.02,
    'cemm_fragments': 0.03,
    'midas': 0.005,
    'synthetic_decoys': 0.25
}
```

**Key Insight**: Boltz-2 gives **highest weight to PubChem HTS** (0.44), not to the high-quality ChEMBL/BindingDB data (0.25). This suggests they prioritize learning from diverse screening data.

### 5. **Data Curation**

**Our Implementation**:
- Basic quality weighting (10.0 for high-quality, lower for others)
- Synthetic data generation for testing
- No specific filtering strategy

**Boltz-2 Reality**:
- **Extensive curation pipeline**:
  1. Only single-protein targets
  2. Only biochemical or functional assays
  3. Exclude low-confidence assays
  4. Apply PAINS filters
  5. Discard ligands with >50 heavy atoms
  6. Filter by structural quality (iptm >0.75)
  7. Standardize all values to log10(µM)
  8. Discard assays with low affinity standard deviation
  9. For HTS: only assays with ≥100 compounds and hit rate <10%
  10. Cross-reference HTS hits with confirmatory assays

---

## 🏗️ Architecture Comparison

### **Our Implementation**

```python
class MultiTaskPEARL(nn.Module):
    def __init__(self, base_pearl, ...):
        self.pearl = base_pearl
        
        # 4 task-specific heads
        self.binding_head = BindingAffinityHead(...)
        self.kcat_head = CatalyticActivityHead(...)
        self.fitness_head = FitnessScoreHead(...)
        self.ddg_head = DDGPredictionHead(...)
    
    def forward(self, batch, task):
        # Route to appropriate head based on task
        if task == 'binding_affinity':
            return self.binding_head(...)
        elif task == 'kcat':
            return self.kcat_head(...)
        # ... etc
```

### **Boltz-2 Reality**

```python
class Boltz2(nn.Module):
    def __init__(self, ...):
        self.trunk = Trunk(...)  # Pairwise stack
        self.denoising_module = DenoisingModule(...)
        self.confidence_module = ConfidenceModule(...)
        
        # Single affinity module with 2 heads
        self.affinity_module = AffinityModule(
            pairformer=PairFormer(...),
            binary_head=BinaryClassificationHead(...),
            continuous_head=ContinuousAffinityHead(...)
        )
    
    def forward(self, batch):
        # Three-phase training:
        # Phase 1: Structure (trunk + denoising)
        # Phase 2: Confidence
        # Phase 3: Affinity (gradients detached from trunk)
        
        pair_repr = self.trunk(batch)
        
        if self.training_phase == 'affinity':
            pair_repr = pair_repr.detach()  # Freeze trunk!
            
        affinity_out = self.affinity_module(pair_repr)
        return {
            'binary_logits': affinity_out['binary'],
            'continuous_affinity': affinity_out['continuous']
        }
```

**Key Difference**: Boltz-2 uses **three-phase training** and **detaches gradients** from the trunk during affinity training. We didn't implement this.

---

## 📈 Loss Function Comparison

### **Our Implementation**

```python
class MultiTaskLoss(nn.Module):
    def forward(self, predictions, targets, task, data_weights):
        # MSE loss
        mse_loss = F.mse_loss(pred_values, targets, reduction='none')
        
        # Uncertainty-aware NLL loss
        nll_loss = 0.5 * (torch.log(2 * np.pi * confidence**2) + 
                         ((pred_values - targets)**2) / (confidence**2))
        
        # Task-weighted total loss
        task_weight = self.task_weights[task]
        total_loss = task_weight * (mse_loss + 0.5 * nll_loss)
        
        return total_loss
```

### **Boltz-2 Reality**

```python
class Boltz2AffinityLoss(nn.Module):
    def forward(self, predictions, targets, qualifiers):
        # Pairwise differences loss (primary)
        y1, y2 = targets['pair']
        ŷ1, ŷ2 = predictions['pair']
        
        if qualifiers == ('=', '='):
            Ldif = huber_loss(y1 - y2, ŷ1 - ŷ2, delta=0.5)
        else:
            # Handle inequality qualifiers (>, <)
            Ldif = indicator_based_loss(...)
        
        # Absolute affinity loss (secondary)
        Labs = huber_loss(targets['absolute'], predictions['absolute'], delta=0.5)
        
        # Binary classification loss
        Lbinary = focal_loss(predictions['logits'], targets['binary'], gamma=1)
        
        # Total loss
        Ltotal = 0.9 * Ldif + 0.1 * Labs + Lbinary
        
        return Ltotal
```

**Key Differences**:
1. Boltz-2 uses **Huber loss** (robust to outliers) instead of MSE
2. **Pairwise differences** are weighted 9× more than absolute values
3. **Focal loss** for binary classification (handles class imbalance)
4. Handles **inequality qualifiers** (>, <) in affinity measurements

---

## ✅ What We Got Right

Despite the differences, we got several things right:

1. ✅ **Pair representation approach** - Using trunk output for predictions
2. ✅ **Uncertainty quantification** - Predicting confidence bounds
3. ✅ **Per-residue contributions** - Interpretability features
4. ✅ **Attention mechanisms** - For aggregating information
5. ✅ **Data weighting** - Concept of weighting different data sources
6. ✅ **Synthetic data generation** - For testing before real data
7. ✅ **Modular architecture** - Easy to extend and modify

---

## 🔧 What Needs to Change

To match Boltz-2, we need to:

### **1. Update Dataset Loaders** (`pearl/data/multitask_datasets.py`)

```python
# Remove or deprioritize
- PDBBindDataset  # Not in Boltz-2
- SKEMPI2Dataset  # Not in Boltz-2
- BRENDADataset   # Not in Boltz-2
- ProteinGymDataset  # Not in Boltz-2

# Add
+ ChEMBLDataset
+ BindingDBDataset
+ PubChemHTSDataset
+ PubChemSmallAssaysDataset
+ CeMMFragmentsDataset
+ MIDASDataset
+ SyntheticDecoysDataset
```

### **2. Update Loss Functions** (`scripts/train_multitask_pearl.py`)

```python
# Replace MSE with Huber loss
- F.mse_loss(pred, target)
+ F.huber_loss(pred, target, delta=0.5)

# Add pairwise differences loss
+ pairwise_diff_loss = huber_loss(y1 - y2, ŷ1 - ŷ2, delta=0.5)

# Add focal loss for binary classification
+ focal_loss = -(1 - p)**gamma * log(p)

# Update loss weighting
- total_loss = mse_loss + 0.5 * nll_loss
+ total_loss = 0.9 * pairwise_diff_loss + 0.1 * absolute_loss + focal_loss
```

### **3. Update Training Strategy**

```python
# Add three-phase training
Phase 1: Train structure (trunk + denoising)
Phase 2: Train confidence module
Phase 3: Train affinity module (freeze trunk!)

# Add proper sampling weights
sampling_weights = {
    'chembl_bindingdb': 0.25,
    'pubchem_hts': 0.44,  # Highest!
    'pubchem_small_values': 0.005,
    'pubchem_small_binary': 0.02,
    'cemm_fragments': 0.03,
    'midas': 0.005,
    'synthetic_decoys': 0.25
}

# Add data curation pipeline
- Apply PAINS filters
- Filter by iptm >0.75
- Standardize to log10(µM)
- Filter HTS assays (≥100 compounds, hit rate <10%)
```

### **4. Simplify Architecture**

```python
# Remove multi-task complexity
- MultiTaskPEARL with 4 heads
+ Single affinity module with 2 heads (binary + continuous)

# Add gradient detachment
if training_phase == 'affinity':
    pair_repr = pair_repr.detach()  # Freeze trunk!
```

---

## 🎯 Recommendations

### **Option 1: Replicate Boltz-2 Exactly** (Recommended for comparison)

**Pros**:
- Direct comparison with Boltz-2 results
- Proven approach with published benchmarks
- Focused on single task (binding affinity)

**Cons**:
- Loses multi-task capabilities (kcat, fitness, protein-protein ΔΔG)
- Requires downloading and curating 6 new datasets

**Effort**: ~2-3 weeks (mostly data preparation)

### **Option 2: Hybrid Approach** (Recommended for innovation)

**Pros**:
- Keep multi-task capabilities
- Add Boltz-2 datasets as additional tasks
- Best of both worlds

**Cons**:
- More complex architecture
- Longer training time
- Harder to compare directly with Boltz-2

**Effort**: ~3-4 weeks

### **Option 3: Keep Current Implementation** (Not recommended)

**Pros**:
- Already working
- Tested on synthetic data

**Cons**:
- Based on incorrect assumptions about Boltz-2
- Can't claim to replicate Boltz-2 approach
- Missing key datasets and training strategies

---

## 📊 Summary Table

| Aspect | Our Implementation | Boltz-2 Reality | Match? |
|--------|-------------------|-----------------|--------|
| **Datasets** | PDBbind, SKEMPI2, BRENDA, ProteinGym | ChEMBL, BindingDB, PubChem, CeMM, MIDAS | ❌ |
| **Tasks** | 4 tasks (binding, ΔΔG, kcat, fitness) | 1 task (binding affinity only) | ❌ |
| **Loss** | MSE + NLL | Huber + pairwise + focal | ❌ |
| **Sampling** | Equal task weights | Dataset-specific weights (0.005-0.44) | ❌ |
| **Curation** | Basic quality weighting | Extensive 10-step pipeline | ❌ |
| **Training** | Single-phase | Three-phase (structure → confidence → affinity) | ❌ |
| **Architecture** | Multi-task with 4 heads | Single affinity module with 2 heads | ❌ |
| **Pair repr** | ✅ Trunk output | ✅ Trunk output | ✅ |
| **Uncertainty** | ✅ Confidence bounds | ✅ Confidence bounds | ✅ |
| **Interpretability** | ✅ Per-residue contrib | ✅ Attention weights | ✅ |

**Overall Match**: ~30% (architecture concepts correct, but datasets and training strategy different)

---

## 🚀 Next Steps

1. **Review this comparison** with the user
2. **Decide on approach**: Replicate Boltz-2 exactly, hybrid, or keep current?
3. **Update implementation** based on decision
4. **Download and curate datasets** (if replicating Boltz-2)
5. **Test on synthetic data** first
6. **Train on real data** and benchmark

---

**Bottom Line**: We built a solid multi-task learning framework, but it's based on different datasets and training strategies than Boltz-2. To truly replicate Boltz-2, we need to update our implementation to match their exact approach.

