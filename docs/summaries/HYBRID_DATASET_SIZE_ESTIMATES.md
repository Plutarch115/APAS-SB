# Hybrid Dataset Size Estimates
## Combining Original Multi-Task + Boltz-2 Datasets

## 🎯 Executive Summary

The hybrid implementation combines:
1. **Original multi-task datasets** (4 datasets) for diverse biochemical properties
2. **Boltz-2 datasets** (7 datasets) for binding affinity prediction

**Total Estimated Size**: ~5.1M training examples, ~200 GB storage, ~$85M training cost

---

## 📊 Detailed Dataset Breakdown

### **PART 1: Original Multi-Task Datasets**

| Dataset | # Examples | # Targets | # Compounds | Task | Storage | Source |
|---------|-----------|-----------|-------------|------|---------|--------|
| **PDBbind** | 20,000 | 3,000 | 15,000 | Binding affinity | 5 GB | http://www.pdbbind.org.cn/ |
| **SKEMPI 2.0** | 8,000 | 200 | - | Protein-protein ΔΔG | 2 GB | https://life.bsc.es/pid/skempi2 |
| **BRENDA** | 100,000 | 50,000 | 80,000 | Enzyme kcat | 15 GB | https://www.brenda-enzymes.org/ |
| **ProteinGym** | 2,500,000 | 250 | - | Fitness scores | 30 GB | https://proteingym.org/ |
| **Subtotal** | **2,628,000** | **53,450** | **95,000** | - | **52 GB** | - |

### **PART 2: Boltz-2 Datasets**

| Dataset | # Binders | # Decoys | # Targets | # Compounds | Task | Storage | Source |
|---------|-----------|----------|-----------|-------------|------|---------|--------|
| **ChEMBL v34** | 600,000 | 0 | 1,000 | 300,000 | Affinity (continuous) | 10 GB | ChEMBL |
| **BindingDB** | 600,000 | 0 | 1,000 | 300,000 | Affinity (continuous) | 10 GB | BindingDB |
| **PubChem HTS** | 200,000 | 1,800,000 | 300 | 400,000 | Binary classification | 50 GB | PubChem |
| **PubChem Small** | 10,000 | 50,000 | 250 | 20,000 | Both | 5 GB | PubChem |
| **CeMM Fragments** | 25,000 | 115,000 | 1,300 | 400 | Binary classification | 3 GB | Offensperger 2024 |
| **MIDAS** | 2,000 | 20,000 | 60 | 400 | Binary classification | 1 GB | Hicks 2023 |
| **Synthetic Decoys** | 0 | 1,200,000 | 2,000 | 600,000 | Binary classification | 25 GB | Generated |
| **Subtotal** | **1,437,000** | **3,185,000** | **5,910** | **1,620,800** | - | **104 GB** | - |

### **TOTAL HYBRID DATASET**

| Metric | Value |
|--------|-------|
| **Total Training Examples** | **~7.25M** (2.63M multi-task + 4.62M Boltz-2) |
| **Total Unique Targets** | **~59,360** (53,450 + 5,910) |
| **Total Unique Compounds** | **~1.72M** (95,000 + 1,620,800) |
| **Total Storage (Raw)** | **~156 GB** (52 GB + 104 GB) |
| **Total Storage (Processed)** | **~200 GB** (with features, embeddings) |

---

## 💾 Storage Breakdown by Component

### **1. Raw Data** (~156 GB)

| Component | Size | Description |
|-----------|------|-------------|
| Protein sequences | 20 GB | FASTA files, ~60k unique proteins |
| Protein structures | 50 GB | PDB files, AlphaFold predictions |
| Ligand structures | 15 GB | SMILES, SDF files, ~1.7M compounds |
| Affinity measurements | 5 GB | CSV/JSON files with Ki, Kd, IC50 values |
| Binary labels | 10 GB | Active/inactive classifications |
| Metadata | 6 GB | Assay conditions, quality scores, etc. |
| **Subtotal** | **106 GB** | - |

### **2. Processed Features** (~94 GB)

| Component | Size | Description |
|-----------|------|-------------|
| Protein embeddings | 40 GB | ESM-2 or similar (1280-dim × 60k proteins) |
| Ligand embeddings | 25 GB | Morgan fingerprints or GNN embeddings |
| Pair representations | 20 GB | Precomputed trunk outputs (optional) |
| Cached structures | 9 GB | Preprocessed PDB files |
| **Subtotal** | **94 GB** | - |

### **3. Total Storage** = **200 GB**

---

## 🔢 Training Data Statistics

### **By Task Type**

| Task | # Examples | % of Total | Avg. Quality |
|------|-----------|------------|--------------|
| **Binding Affinity (continuous)** | 1,220,000 | 16.8% | High (ChEMBL, BindingDB, PDBbind) |
| **Binding Classification (binary)** | 3,185,000 | 43.9% | Medium (HTS noise) |
| **Protein-Protein ΔΔG** | 8,000 | 0.1% | Very High (SKEMPI 2.0) |
| **Enzyme kcat** | 100,000 | 1.4% | High (BRENDA) |
| **Fitness Scores** | 2,500,000 | 34.5% | High (ProteinGym DMS) |
| **Decoys** | 3,185,000 | 43.9% | N/A (negative examples) |

### **By Data Source Quality**

| Quality Tier | # Examples | % of Total | Weight | Description |
|--------------|-----------|------------|--------|-------------|
| **Tier 1 (Highest)** | 1,328,000 | 18.3% | 10.0 | ChEMBL, BindingDB, SKEMPI, BRENDA |
| **Tier 2 (High)** | 2,500,000 | 34.5% | 9.0 | ProteinGym DMS |
| **Tier 3 (Medium)** | 60,000 | 0.8% | 7.0 | PubChem Small Assays |
| **Tier 4 (Lower)** | 2,140,000 | 29.5% | 5.0 | PubChem HTS, CeMM, MIDAS |
| **Tier 5 (Synthetic)** | 1,220,000 | 16.8% | 3.0 | Synthetic decoys |

---

## 💰 Cost Estimates

### **1. Data Acquisition Costs**

| Dataset | Cost | Time | Notes |
|---------|------|------|-------|
| **ChEMBL v34** | Free | 1 day | Public download |
| **BindingDB** | Free | 1 day | Public download |
| **PubChem** | Free | 3 days | Large download (50 GB) |
| **CeMM Fragments** | Free | 4 hours | Request from authors |
| **MIDAS** | Free | 2 hours | Request from authors |
| **PDBbind** | Free | 1 day | Public download |
| **SKEMPI 2.0** | Free | 2 hours | Public download |
| **BRENDA** | Free | 1 day | Public download (requires registration) |
| **ProteinGym** | Free | 2 days | Large download (30 GB) |
| **Total** | **$0** | **~10 days** | All datasets are publicly available |

### **2. Data Processing Costs**

| Task | Compute | Time | Cost | Notes |
|------|---------|------|------|-------|
| **Download & extract** | CPU | 10 days | $0 | Manual work |
| **Parse & filter** | 32 CPU cores | 5 days | $500 | Apply Boltz-2 curation |
| **Generate embeddings** | 8× A100 GPUs | 3 days | $5,000 | ESM-2 protein embeddings |
| **Precompute features** | 16 CPU cores | 7 days | $300 | Ligand fingerprints, etc. |
| **Quality control** | Manual | 5 days | $2,000 | Human review of edge cases |
| **Total** | - | **~30 days** | **~$7,800** | One-time cost |

### **3. Training Costs** (Hybrid Model)

Assuming training on the full hybrid dataset:

| Component | Compute | Time | Cost | Notes |
|-----------|---------|------|------|-------|
| **Structure training** | 64× A100 GPUs | 30 days | $50M | Train trunk on structures |
| **Confidence training** | 32× A100 GPUs | 10 days | $10M | Train confidence module |
| **Affinity training** | 16× A100 GPUs | 15 days | $7.5M | Train affinity heads (frozen trunk) |
| **Multi-task training** | 32× A100 GPUs | 20 days | $20M | Train kcat, fitness, ΔΔG heads |
| **Total** | - | **~75 days** | **~$87.5M** | Full training from scratch |

**Cost Reduction Strategies**:
- Use pretrained PEARL trunk: **-$50M** (skip structure training)
- Use smaller model: **-50%** (reduce parameters by 2×)
- Use mixed precision: **-30%** (FP16/BF16 training)
- **Estimated cost with optimizations**: **~$15-20M**

### **4. Storage Costs**

| Storage Type | Size | Cost/Month | Annual Cost |
|--------------|------|------------|-------------|
| **Raw data** | 156 GB | $4 | $48 |
| **Processed data** | 200 GB | $5 | $60 |
| **Model checkpoints** | 50 GB | $1 | $12 |
| **Total** | **406 GB** | **$10/month** | **$120/year** |

---

## 📈 Training Time Estimates

### **Scenario 1: Full Training from Scratch**

| Phase | Epochs | Examples/Epoch | Time/Epoch | Total Time |
|-------|--------|----------------|------------|------------|
| **Structure** | 50 | 7.25M | 14 hours | 30 days |
| **Confidence** | 20 | 7.25M | 12 hours | 10 days |
| **Affinity** | 30 | 4.62M | 12 hours | 15 days |
| **Multi-task** | 40 | 2.63M | 12 hours | 20 days |
| **Total** | - | - | - | **~75 days** |

**Hardware**: 64× A100 GPUs (80 GB each)

### **Scenario 2: Using Pretrained PEARL Trunk**

| Phase | Epochs | Examples/Epoch | Time/Epoch | Total Time |
|-------|--------|----------------|------------|------------|
| **Affinity heads** | 30 | 4.62M | 8 hours | 10 days |
| **Multi-task heads** | 40 | 2.63M | 6 hours | 10 days |
| **Fine-tuning** | 10 | 7.25M | 10 hours | 4 days |
| **Total** | - | - | - | **~24 days** |

**Hardware**: 32× A100 GPUs (80 GB each)

### **Scenario 3: Incremental Training (Recommended)**

Train on subsets first, then scale up:

| Phase | Dataset Size | Time | Cost |
|-------|-------------|------|------|
| **Phase 1: Synthetic data** | 50K | 1 day | $500 |
| **Phase 2: High-quality subset** | 500K | 5 days | $5K |
| **Phase 3: Full dataset** | 7.25M | 24 days | $20M |
| **Total** | - | **~30 days** | **~$20M** |

---

## 🎯 Data Sampling Strategy

To handle the large hybrid dataset efficiently, use weighted sampling:

### **Boltz-2 Sampling Weights** (for binding affinity tasks)

| Dataset | Weight | Effective Samples/Epoch |
|---------|--------|-------------------------|
| ChEMBL + BindingDB | 0.25 | 300,000 |
| PubChem HTS | 0.44 | 880,000 |
| PubChem Small (values) | 0.005 | 300 |
| PubChem Small (binary) | 0.02 | 1,200 |
| CeMM Fragments | 0.03 | 4,200 |
| MIDAS | 0.005 | 110 |
| Synthetic Decoys | 0.25 | 300,000 |
| **Total** | **1.0** | **~1.5M/epoch** |

### **Multi-Task Sampling Weights** (for other tasks)

| Dataset | Weight | Effective Samples/Epoch |
|---------|--------|-------------------------|
| ProteinGym | 0.50 | 1,250,000 |
| BRENDA | 0.20 | 20,000 |
| PDBbind | 0.15 | 3,000 |
| SKEMPI 2.0 | 0.15 | 1,200 |
| **Total** | **1.0** | **~1.3M/epoch** |

**Combined**: ~2.8M effective samples per epoch (vs. 7.25M total)

---

## 📊 Expected Performance

Based on Boltz-2 benchmarks and multi-task learning literature:

### **Binding Affinity Prediction**

| Benchmark | Metric | Expected Performance | Boltz-2 | Improvement |
|-----------|--------|---------------------|---------|-------------|
| **FEP+ (OpenFE)** | Pearson R | 0.64-0.66 | 0.62 | +3-6% |
| **CASP16** | Pearson R | 0.66-0.68 | 0.65 | +2-5% |
| **MF-PCBA** | AP | 0.026-0.028 | 0.0248 | +5-13% |
| **MF-PCBA** | EF@0.5% | 19-21 | 18.4 | +3-14% |

**Rationale**: Hybrid model has more diverse training data, which should improve generalization.

### **Multi-Task Predictions** (New Capabilities)

| Task | Metric | Expected Performance | Baseline | Improvement |
|------|--------|---------------------|----------|-------------|
| **Protein-Protein ΔΔG** | Pearson R | 0.55-0.60 | 0.50 | +10-20% |
| **Enzyme kcat** | Pearson R | 0.45-0.50 | 0.40 | +12-25% |
| **Fitness Scores** | Spearman ρ | 0.50-0.55 | 0.45 | +11-22% |

**Rationale**: Multi-task learning provides regularization and shared representations.

---

## 🚀 Recommendations

### **Phase 1: Proof of Concept** (Week 1-2)
- Use synthetic data (50K examples)
- Test all dataset loaders
- Verify training pipeline
- **Cost**: $500, **Time**: 2 weeks

### **Phase 2: High-Quality Subset** (Week 3-6)
- ChEMBL + BindingDB (1.2M examples)
- SKEMPI 2.0 (8K examples)
- ProteinGym subset (100K examples)
- **Cost**: $5K, **Time**: 4 weeks

### **Phase 3: Full Hybrid Dataset** (Week 7-14)
- All 7.25M examples
- Full multi-task training
- Comprehensive benchmarking
- **Cost**: $20M, **Time**: 8 weeks

### **Total Timeline**: ~14 weeks (~3.5 months)
### **Total Cost**: ~$20M (with pretrained trunk)

---

## 📋 Data Preparation Checklist

### **Week 1-2: Data Acquisition**
- [ ] Download ChEMBL v34 (10 GB)
- [ ] Download BindingDB (5 GB)
- [ ] Download PubChem HTS (50 GB)
- [ ] Download PubChem Small Assays (5 GB)
- [ ] Request CeMM Fragments dataset
- [ ] Request MIDAS dataset
- [ ] Download PDBbind (5 GB)
- [ ] Download SKEMPI 2.0 (2 GB)
- [ ] Download BRENDA (15 GB)
- [ ] Download ProteinGym (30 GB)

### **Week 3-4: Data Processing**
- [ ] Parse and filter ChEMBL (apply Boltz-2 curation)
- [ ] Parse and filter BindingDB
- [ ] Parse PubChem assays (filter by quality)
- [ ] Process CeMM and MIDAS data
- [ ] Apply PAINS filters to all ligands
- [ ] Filter proteins by iptm >0.75
- [ ] Standardize all affinity values to log10(µM)
- [ ] Generate synthetic decoys (Tanimoto <0.3)

### **Week 5-6: Feature Generation**
- [ ] Generate protein embeddings (ESM-2)
- [ ] Generate ligand fingerprints
- [ ] Precompute pair representations (optional)
- [ ] Create train/val/test splits (90% sequence identity)
- [ ] Compute data statistics and quality metrics

### **Week 7+: Training**
- [ ] Train on synthetic data (sanity check)
- [ ] Train on high-quality subset
- [ ] Train on full hybrid dataset
- [ ] Benchmark on all test sets
- [ ] Compare with Boltz-2 and baselines

---

## 💡 Key Insights

1. **Data is mostly free**: All datasets are publicly available ($0 acquisition cost)
2. **Processing is expensive**: ~$8K and 30 days to prepare data
3. **Training is very expensive**: ~$20M with optimizations, ~$87M from scratch
4. **Storage is cheap**: ~$10/month for 200 GB
5. **Hybrid approach adds value**: Multi-task capabilities + Boltz-2 performance
6. **Incremental training recommended**: Start small, scale up gradually

---

**Bottom Line**: The hybrid dataset contains **~7.25M training examples** requiring **~200 GB storage** and **~$20M training cost** (with pretrained trunk). This provides both Boltz-2's binding affinity prediction capabilities AND additional multi-task predictions (kcat, fitness, protein-protein ΔΔG).

