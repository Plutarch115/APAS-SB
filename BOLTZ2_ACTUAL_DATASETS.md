# Boltz-2 Actual Datasets - Based on Paper Analysis
## Reference: 2025.06.14.659707v1.full.pdf

## 🎯 Executive Summary

After analyzing the actual Boltz-2 paper, here are the **exact datasets** they used for training their binding affinity prediction model:

### **Datasets Used in Boltz-2** (Table 1 from paper)

| Source | Type | Supervision | # Binders | # Decoys | # Targets | # Compounds |
|--------|------|-------------|-----------|----------|-----------|-------------|
| **ChEMBL + BindingDB** | optimization | values | 1.2M | 0 | 2k | 600k |
| **PubChem small assays** | hit-discovery | both | 10k | 50k | 250 | 20k |
| **PubChem HTS** | hit-discovery | binary | 200k | 1.8M | 300 | 400k |
| **CeMM Fragments** | hit-discovery | binary | 25k | 115k | 1.3k | 400 |
| **MIDAS Metabolites** | hit-discovery | binary | 2k | 20k | 60 | 400 |
| **Synthetic decoys** | - | binary | 0 | 1.2M | 2k | 600k |
| **TOTAL** | - | - | **1.4M** | **3.2M** | **3.5k** | **1M** |

### **Key Findings**

1. **NO kcat/enzyme kinetics data** - BRENDA, SABIO-RK not mentioned
2. **NO deep mutational scanning** - ProteinGym not mentioned
3. **NO protein-protein ΔΔG** - SKEMPI 2.0 not mentioned
4. **Focus on binding affinity only** - Both continuous (Ki, Kd, IC50) and binary (active/inactive)

---

## 📊 Detailed Dataset Descriptions

### 1. **ChEMBL (v34)** - Primary Affinity Data

**What Boltz-2 Says**:
> "For the binding affinity regression values (e.g., Ki, Kd, IC50, AC50, EC50, XC50), we gather data from PubChem, ChEMBL, and BindingDB."

**Details**:
- **Type**: Manually curated bioactive molecules
- **Size in Boltz-2**: 1.2M binders (combined with BindingDB)
- **Measurements**: Ki, Kd, IC50, AC50, EC50, XC50
- **Standardization**: All converted to log10(µM)
- **URL**: https://www.ebi.ac.uk/chembl/

**Curation Strategy**:
- Only single-protein targets
- Only biochemical or functional assays
- Exclude low-confidence or unreliable assays
- Discard assays with low affinity standard deviation
- Apply PAINS filters
- Exclude ligands with >50 heavy atoms
- Filter by structural quality (iptm >0.75)

**Sampling Weight in Training**: 0.25

---

### 2. **BindingDB** - Affinity Data

**What Boltz-2 Says**:
> "We gather data from PubChem, ChEMBL, and BindingDB... We retain only assays that target a single protein and are categorized as either biochemical or functional."

**Details**:
- **Type**: Protein-ligand binding measurements
- **Size in Boltz-2**: Combined with ChEMBL (1.2M total)
- **Measurements**: Ki, Kd, IC50, AC50, EC50, XC50
- **URL**: https://www.bindingdb.org/
- **Curation**: Same as ChEMBL

**Sampling Weight in Training**: 0.25 (combined with ChEMBL)

---

### 3. **PubChem HTS** - High-Throughput Screening

**What Boltz-2 Says**:
> "For PubChem HTS, we retain only assays that include at least 100 compounds and exhibit a hit rate below 10%, helping to filter out noisy screens."

**Details**:
- **Type**: Large-scale binary screening assays
- **Size in Boltz-2**: 200K binders, 1.8M decoys
- **Targets**: 300 protein clusters (90% sequence identity)
- **Compounds**: 400K unique
- **URL**: https://pubchem.ncbi.nlm.nih.gov/

**Curation Strategy**:
- Only assays with ≥100 compounds
- Hit rate <10%
- Cross-reference with confirmatory assays
- Check for associated quantitative measurements (Ki, Kd, XC50)
- Estimated ~40% false positive rate

**Sampling Weight in Training**: 0.44 (highest weight!)

---

### 4. **PubChem Small Assays** - Higher Quality Screens

**What Boltz-2 Says**:
> "PubChem small assays... supervised for both classification and regression"

**Details**:
- **Type**: Smaller, higher-quality screening assays
- **Size in Boltz-2**: 10K binders, 50K decoys
- **Targets**: 250 protein clusters
- **Compounds**: 20K unique
- **Supervision**: Both binary and continuous values

**Sampling Weight in Training**: 
- Values: 0.005
- Binary: 0.02

---

### 5. **CeMM Fragment Screening**

**What Boltz-2 Says**:
> "A fragment screening dataset from CeMM [Offensperger et al., 2024]"

**Details**:
- **Type**: Fragment-based drug discovery screens
- **Size in Boltz-2**: 25K binders, 115K decoys
- **Targets**: 1.3K protein clusters
- **Compounds**: 400 unique fragments
- **Reference**: Offensperger et al., 2024

**Sampling Weight in Training**: 0.03

---

### 6. **MIDAS** - Protein-Metabolite Interactome

**What Boltz-2 Says**:
> "MIDAS, a protein–metabolite interactome dataset from the University of Utah [Hicks et al., 2023]"

**Details**:
- **Type**: Protein-metabolite interactions
- **Size in Boltz-2**: 2K binders, 20K decoys
- **Targets**: 60 protein clusters
- **Compounds**: 400 unique metabolites
- **Reference**: Hicks et al., 2023 (University of Utah)

**Sampling Weight in Training**: 0.005

---

### 7. **Synthetic Decoys** - Data Augmentation

**What Boltz-2 Says**:
> "We augment the binary classification dataset by generating synthetic decoys created by shuffling binders identified in hit-to-lead screens across different targets, while mitigating low false negative rates by ensuring that each decoy has a Tanimoto similarity below 0.3 to all known binders associated with similar proteins."

**Details**:
- **Type**: Computationally generated negative examples
- **Size in Boltz-2**: 1.2M decoys
- **Source**: Shuffled from ChEMBL/BindingDB binders
- **Generation Strategy**:
  - Shuffle binders across different targets
  - Ensure Tanimoto similarity <0.3 to known binders of similar proteins
  - Reduces false negatives

**Sampling Weight in Training**: 0.25

---

## 🚫 Datasets NOT Used in Boltz-2

The following datasets were **NOT** mentioned in the Boltz-2 paper:

### ❌ **BRENDA** - Enzyme Kinetics
- **Status**: Not mentioned
- **Why relevant**: Contains kcat, Km data for enzyme catalysis
- **Potential**: Could extend model to predict catalytic activity

### ❌ **SABIO-RK** - Reaction Kinetics
- **Status**: Not mentioned
- **Why relevant**: Biochemical reaction kinetics
- **Potential**: Metabolic pathway modeling

### ❌ **ProteinGym** - Deep Mutational Scanning
- **Status**: Not mentioned
- **Why relevant**: Fitness scores from systematic mutagenesis
- **Potential**: Could predict mutation effects

### ❌ **SKEMPI 2.0** - Protein-Protein ΔΔG
- **Status**: Not mentioned
- **Why relevant**: Protein-protein interaction affinity changes
- **Potential**: Already implemented in our ΔΔG predictor

### ❌ **ProTherm** - Protein Stability
- **Status**: Not mentioned
- **Why relevant**: ΔΔG of folding/unfolding
- **Potential**: Stability prediction

---

## 🔧 Integration Strategy for PEARL

### **Phase 1: Replicate Boltz-2 Datasets** (Recommended)

Focus on the 6 datasets actually used in Boltz-2:

1. **ChEMBL (v34)** - Download and parse affinity values
2. **BindingDB** - Download and parse affinity values
3. **PubChem HTS** - Download high-throughput screening assays
4. **PubChem Small Assays** - Download smaller, higher-quality assays
5. **CeMM Fragments** - Obtain from Offensperger et al., 2024
6. **MIDAS** - Obtain from Hicks et al., 2023

**Implementation**:
```python
# Update pearl/data/multitask_datasets.py to match Boltz-2 exactly

class Boltz2AffinityDataset(Dataset):
    """Combined dataset matching Boltz-2 training data"""
    
    def __init__(self, data_dir, split='train'):
        # Load all 6 datasets
        self.chembl_data = self._load_chembl()
        self.bindingdb_data = self._load_bindingdb()
        self.pubchem_hts = self._load_pubchem_hts()
        self.pubchem_small = self._load_pubchem_small()
        self.cemm_fragments = self._load_cemm()
        self.midas = self._load_midas()
        self.synthetic_decoys = self._generate_decoys()
        
        # Apply Boltz-2 sampling weights
        self.sampling_weights = {
            'chembl_bindingdb': 0.25,
            'pubchem_small_values': 0.005,
            'pubchem_hts': 0.44,
            'pubchem_small_binary': 0.02,
            'cemm': 0.03,
            'midas': 0.005,
            'synthetic_decoys': 0.25
        }
```

### **Phase 2: Extend Beyond Boltz-2** (Optional)

After replicating Boltz-2, optionally add:

1. **BRENDA** - For kcat prediction
2. **ProteinGym** - For fitness score prediction
3. **SKEMPI 2.0** - For protein-protein ΔΔG (already implemented)

---

## 📈 Expected Performance

Based on Boltz-2 benchmarks:

| Benchmark | Metric | Boltz-2 | FEP+ | OpenFE |
|-----------|--------|---------|------|--------|
| **FEP+ (OpenFE subset)** | Pearson R | 0.62 | 0.72 | 0.63 |
| **FEP+ (4 targets)** | Pearson R | 0.66 | 0.78 | 0.66 |
| **CASP16** | Pearson R | 0.65 | - | - |
| **MF-PCBA** | AP | 0.0248 | - | - |
| **MF-PCBA** | EF@0.5% | 18.4 | - | - |

**Key Insights**:
- Boltz-2 approaches FEP accuracy (R=0.62-0.66 vs 0.72-0.78)
- 1000× faster than FEP (20 sec vs 6-12 GPU hours)
- Strong performance on binary classification (EF@0.5% = 18.4)

---

## 💰 Data Preparation Cost

| Dataset | Download Size | Processing Time | Storage |
|---------|---------------|-----------------|---------|
| ChEMBL v34 | ~10 GB | 1 day | 20 GB |
| BindingDB | ~5 GB | 1 day | 10 GB |
| PubChem HTS | ~50 GB | 3 days | 100 GB |
| PubChem Small | ~5 GB | 1 day | 10 GB |
| CeMM Fragments | ~1 GB | 4 hours | 2 GB |
| MIDAS | ~500 MB | 2 hours | 1 GB |
| **Total** | **~72 GB** | **~7 days** | **~143 GB** |

---

## 🎯 Recommendations

### **For Replicating Boltz-2**:
1. ✅ Focus on the 6 datasets they actually used
2. ✅ Apply their exact curation strategy
3. ✅ Use their sampling weights
4. ✅ Standardize to log10(µM) for affinity values
5. ✅ Apply PAINS filters and structural quality filters (iptm >0.75)

### **For Extending Beyond Boltz-2**:
1. 📋 Add BRENDA for kcat prediction (new capability)
2. 📋 Add ProteinGym for fitness scores (new capability)
3. 📋 Keep SKEMPI 2.0 for protein-protein ΔΔG (already implemented)

### **What NOT to Do**:
1. ❌ Don't assume they used datasets they didn't mention
2. ❌ Don't mix in unrelated data without careful curation
3. ❌ Don't skip the quality filters (they're critical!)

---

## 📚 References

1. **Boltz-2 Paper**: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
2. **ChEMBL v34**: Zdrazil et al., 2024 - https://www.ebi.ac.uk/chembl/
3. **BindingDB**: Liu et al., 2007 - https://www.bindingdb.org/
4. **PubChem**: Kim et al., 2023 - https://pubchem.ncbi.nlm.nih.gov/
5. **CeMM Fragments**: Offensperger et al., 2024
6. **MIDAS**: Hicks et al., 2023

---

**Next Steps**: Update `pearl/data/multitask_datasets.py` to match the actual Boltz-2 datasets and curation strategy.

