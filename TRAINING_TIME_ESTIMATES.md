# Pearl Training Time Estimates: Full-Scale Production

## 🎯 Hardware Configuration

### NVIDIA Blackwell B200 GPU Cluster
- **GPUs:** 8× NVIDIA Blackwell B200
- **Memory per GPU:** 192 GB HBM3e
- **Total GPU Memory:** 1.5 TB
- **Compute:** ~20 PetaFLOPS FP8 (total cluster)
- **Interconnect:** NVLink 5.0 (1.8 TB/s per GPU)
- **Host:** Dual AMD EPYC or Intel Xeon (128+ cores)
- **System Memory:** 2+ TB DDR5

### Performance Characteristics
- **FP32 Performance:** ~2.5 PetaFLOPS (total)
- **TF32 Performance:** ~5 PetaFLOPS (total)
- **FP16/BF16 Performance:** ~10 PetaFLOPS (total)
- **FP8 Performance:** ~20 PetaFLOPS (total)
- **Memory Bandwidth:** 8 TB/s (per GPU)

---

## 📊 Dataset Size Estimates

### 1. PDB (Experimental Structures)

**X-ray Crystallography:**
- Total structures: ~180,000
- Protein-ligand complexes: ~25,000
- High-quality (< 2.5Å): ~15,000
- **Training set:** ~12,000 structures

**CryoEM (EMDB):**
- Total structures: ~35,000
- Protein-ligand complexes: ~2,000
- High-quality (< 3.5Å): ~1,500
- **Training set:** ~1,200 structures

**Total Experimental:** ~13,200 structures

### 2. AlphaFoldDB (Predicted Structures)

**Important Note:** AlphaFoldDB contains **predicted structures without ligands**. For Pearl training, we need:
- Protein structures from AlphaFoldDB: ~200 million
- **But:** Need to dock ligands synthetically
- **Realistic subset:** ~100,000 diverse proteins
- **With synthetic ligands:** ~100,000 × 640 = 64 million complexes

**Breakdown:**
- Human proteome: ~20,000 proteins
- Model organisms: ~50,000 proteins
- Diverse fold space: ~30,000 proteins
- **Total proteins:** ~100,000
- **Ligands per protein:** 640 (from paper)
- **Total synthetic complexes:** 64 million

### 3. Combined Training Dataset

**Experimental (high weight):**
- X-ray: 12,000 structures
- CryoEM: 1,200 structures
- **Subtotal:** 13,200 structures

**Synthetic (lower weight):**
- AlphaFoldDB-derived: 64,000,000 complexes
- **Subtotal:** 64 million structures

**Total:** ~64 million training examples

---

## ⚙️ Model and Training Configuration

### Pearl Model Size

**Architecture:**
- Trunk (Pairformer): 48 blocks
- Diffusion module: 24 blocks
- Embedding dimension: 384
- Attention heads: 12
- **Total parameters:** ~500M

**Memory Requirements:**
- Model parameters: ~2 GB (FP32) or ~1 GB (FP16)
- Optimizer states (AdamW): ~4 GB
- Gradients: ~2 GB
- Activations (batch_size=4): ~20 GB
- **Total per GPU:** ~27 GB
- **Fits comfortably in 192 GB Blackwell GPU**

### Training Configuration

**Batch Size:**
- Per GPU: 4 complexes
- Total (8 GPUs): 32 complexes
- Gradient accumulation: 4 steps
- **Effective batch size:** 128 complexes

**Curriculum Stages:**
1. Stage 1: ≤100 atoms (10% of data)
2. Stage 2: ≤200 atoms (20% of data)
3. Stage 3: ≤500 atoms (30% of data)
4. Stage 4: ≤1000 atoms (25% of data)
5. Stage 5: Unlimited (15% of data)

**Training Steps:**
- Experimental data: 3 epochs
- Synthetic data: 1 epoch
- Total steps: ~500,000

---

## ⏱️ Timing Breakdown

### Per-Example Timing

**Forward Pass:**
- Trunk (Pairformer): ~150ms (500 atoms)
- Diffusion module: ~100ms (500 atoms)
- Total forward: ~250ms

**Backward Pass:**
- Gradient computation: ~500ms
- **Total per example:** ~750ms

**With Uncertainty Weighting:**
- B-factor extraction: ~5ms (cached)
- Confidence computation: ~2ms
- Weighted loss: ~3ms
- **Overhead:** ~10ms (negligible)

**Effective Time:**
- Per example: ~750ms
- Per batch (4 examples): ~3 seconds
- Per effective batch (128 examples): ~12 seconds

### Scaling Efficiency

**Data Parallelism (8 GPUs):**
- Linear scaling efficiency: ~85%
- Communication overhead: ~15%
- **Effective speedup:** 6.8×

**Adjusted Timing:**
- Per effective batch: 12s / 6.8 = ~1.8 seconds

---

## 📈 Training Time Calculations

### Phase 1: Experimental Data (High Priority)

**Dataset:**
- 13,200 structures
- 3 epochs
- Total examples: 39,600

**Time:**
- Steps: 39,600 / 128 = 310 steps
- Time per step: 1.8 seconds
- **Total: 310 × 1.8s = 558 seconds = 9.3 minutes**

**With curriculum (5 stages):**
- **Total: 9.3 × 5 = 46.5 minutes**

### Phase 2: Synthetic Data Generation

**Generation Time:**
- 100,000 proteins
- 640 ligands per protein
- Docking time: ~10 seconds per ligand (AutoDock Vina)
- **Total: 100,000 × 640 × 10s = 640M seconds = 7,407 days**

**Parallelization:**
- 1,000 CPU cores (typical HPC cluster)
- **Parallel time: 7,407 / 1,000 = 7.4 days**

**Realistic Estimate:**
- With optimizations (GPU docking, caching): **3-5 days**

### Phase 3: Synthetic Data Training

**Dataset:**
- 64,000,000 complexes
- 1 epoch
- Total examples: 64,000,000

**Time:**
- Steps: 64,000,000 / 128 = 500,000 steps
- Time per step: 1.8 seconds
- **Total: 500,000 × 1.8s = 900,000 seconds = 10.4 days**

**With curriculum (5 stages):**
- Stage 1 (10%): 50,000 steps × 1.8s = 25 hours
- Stage 2 (20%): 100,000 steps × 1.8s = 50 hours
- Stage 3 (30%): 150,000 steps × 1.8s = 75 hours
- Stage 4 (25%): 125,000 steps × 1.8s = 62.5 hours
- Stage 5 (15%): 75,000 steps × 1.8s = 37.5 hours
- **Total: 250 hours = 10.4 days**

### Phase 4: Fine-tuning on Experimental Data

**Dataset:**
- 13,200 structures
- 10 epochs (fine-tuning)
- Total examples: 132,000

**Time:**
- Steps: 132,000 / 128 = 1,031 steps
- Time per step: 1.8 seconds
- **Total: 1,031 × 1.8s = 1,856 seconds = 31 minutes**

---

## 🎯 Total Training Time Estimate

### Conservative Estimate (with buffer)

| Phase | Time | Notes |
|-------|------|-------|
| **1. Experimental Pre-training** | 1 hour | 13K structures, 3 epochs, 5 stages |
| **2. Synthetic Data Generation** | 5 days | 64M complexes, parallel docking |
| **3. Synthetic Data Training** | 12 days | 64M complexes, 1 epoch, 5 stages |
| **4. Experimental Fine-tuning** | 1 hour | 13K structures, 10 epochs |
| **5. Validation & Checkpointing** | 6 hours | Periodic evaluation |
| **6. Buffer (failures, restarts)** | 2 days | 10% buffer |
| **TOTAL** | **~19-20 days** | **End-to-end** |

### Optimistic Estimate (everything works)

| Phase | Time | Notes |
|-------|------|-------|
| **1. Experimental Pre-training** | 1 hour | |
| **2. Synthetic Data Generation** | 3 days | Optimized docking |
| **3. Synthetic Data Training** | 10 days | |
| **4. Experimental Fine-tuning** | 1 hour | |
| **5. Validation & Checkpointing** | 4 hours | |
| **TOTAL** | **~13-14 days** | **Best case** |

### Realistic Estimate

**🎯 Expected Training Time: 15-20 days (2-3 weeks)**

---

## 💰 Cost Estimates

### Cloud GPU Costs (AWS p5.48xlarge equivalent)

**Assumptions:**
- 8× H100 (80GB): ~$30/hour (Blackwell not yet available)
- 8× Blackwell B200: ~$50/hour (estimated)
- Training time: 20 days = 480 hours

**Costs:**
- GPU compute: 480 hours × $50/hour = **$24,000**
- Storage (10 TB): $230/month × 1 month = **$230**
- Data transfer: ~$500
- **Total: ~$25,000**

### On-Premise HPC (Amortized)

**Assumptions:**
- Cluster cost: $500,000 (8× Blackwell + infrastructure)
- Amortization: 3 years
- Utilization: 80%
- Cost per hour: $500,000 / (3 × 365 × 24 × 0.8) = ~$24/hour

**Costs:**
- Compute: 480 hours × $24/hour = **$11,520**
- Power: 480 hours × 20 kW × $0.10/kWh = **$960**
- **Total: ~$12,500**

**On-premise is ~50% cheaper for sustained workloads**

---

## 🚀 Optimization Strategies

### 1. Reduce Synthetic Data (Most Impactful)

**Strategy:** Use fewer proteins or ligands per protein
- 100K proteins × 100 ligands = 10M complexes (instead of 64M)
- **Training time: 10.4 days → 1.6 days**
- **Total time: 20 days → 8 days**

**Trade-off:** Slightly lower performance on rare binding modes

### 2. Mixed Precision Training

**Strategy:** Use FP8 for forward pass, BF16 for backward
- **Speedup: 1.5-2×**
- **Training time: 20 days → 10-13 days**

**Trade-off:** Minimal (modern GPUs handle this well)

### 3. Flash Attention & Kernel Fusion

**Strategy:** Use optimized attention kernels
- **Speedup: 1.3-1.5×**
- **Training time: 20 days → 13-15 days**

**Trade-off:** None (pure optimization)

### 4. Gradient Checkpointing

**Strategy:** Trade compute for memory, increase batch size
- Batch size: 4 → 8 per GPU
- **Speedup: 1.4×**
- **Training time: 20 days → 14 days**

**Trade-off:** 20% slower per step, but 2× throughput

### 5. Cached Docking Results

**Strategy:** Pre-compute and cache docking poses
- **Synthetic generation: 5 days → 1 day**
- **Total time: 20 days → 16 days**

**Trade-off:** Storage (10 TB)

### Combined Optimizations

**All strategies combined:**
- Base: 20 days
- Reduce synthetic: 0.85× (8 days saved)
- Mixed precision: 0.7× (6 days saved)
- Flash attention: 0.85× (3 days saved)
- Gradient checkpointing: 0.7× (6 days saved)
- Cached docking: (4 days saved)

**Realistic combined speedup: 2.5-3×**

**🎯 Optimized Training Time: 7-10 days (1-1.5 weeks)**

---

## 📊 Comparison with Other Models

### AlphaFold 2
- Dataset: ~170K structures (PDB + templates)
- Training time: ~2 weeks on 128 TPUv3 cores
- Equivalent: ~16 V100 GPUs
- **Pearl (8× Blackwell) ≈ 4× faster hardware**

### AlphaFold 3
- Dataset: ~200K structures + synthetic
- Training time: ~3-4 weeks (estimated)
- Hardware: 256 TPUv4 cores
- **Pearl (8× Blackwell) ≈ similar hardware**

### RoseTTAFold Diffusion
- Dataset: ~100K structures
- Training time: ~1 week on 64 A100 GPUs
- **Pearl (8× Blackwell) ≈ 2× faster hardware**

**Pearl's 2-3 week estimate is competitive with state-of-the-art models**

---

## 🎓 Key Insights

### 1. Bottleneck is Synthetic Data Generation
- Docking: 5 days (25% of total time)
- Training: 12 days (60% of total time)
- **Solution:** Parallelize docking, use GPU-accelerated docking

### 2. Blackwell GPUs are Excellent for This
- Large memory (192 GB) → bigger batches
- High compute (FP8) → faster training
- Fast interconnect → efficient multi-GPU
- **8× Blackwell is well-suited for Pearl**

### 3. Uncertainty Weighting Adds Minimal Overhead
- B-factor extraction: cached
- Confidence computation: ~2ms
- Weighted loss: ~3ms
- **Total overhead: < 1%**

### 4. Curriculum Learning is Essential
- Without: 64M examples × 1 epoch = 10 days
- With: Progressive complexity = same 10 days but better convergence
- **Curriculum doesn't add time, improves quality**

### 5. Experimental Data is Tiny
- 13K structures train in < 1 hour
- But provides critical signal
- **High-quality data is worth 1000× synthetic data**

---

## 🎯 Final Recommendations

### Recommended Configuration

**Hardware:**
- 8× NVIDIA Blackwell B200 GPUs
- 2 TB system memory
- 10 TB NVMe storage
- 100 Gbps network

**Dataset:**
- Experimental: 13K structures (PDB + EMDB)
- Synthetic: 10M complexes (100K proteins × 100 ligands)
- **Total: ~10M training examples**

**Training:**
- Batch size: 128 (effective)
- Mixed precision: FP8/BF16
- Curriculum: 5 stages
- Epochs: 1 (synthetic) + 10 (experimental fine-tuning)

**Optimizations:**
- Flash Attention
- Gradient checkpointing
- Cached docking results
- Mixed precision training

**Expected Time:**
- Synthetic generation: 1-2 days (cached)
- Training: 5-7 days
- Fine-tuning: 1 day
- **Total: 7-10 days (1-1.5 weeks)**

**Expected Cost:**
- Cloud: ~$10,000
- On-premise: ~$5,000

### Scaling Options

**If you need faster training:**
- 16× Blackwell GPUs: **4-5 days** (~$15K)
- 32× Blackwell GPUs: **2-3 days** (~$25K)
- 64× Blackwell GPUs: **1-2 days** (~$40K)

**Diminishing returns beyond 16 GPUs due to communication overhead**

---

## 📈 Timeline Breakdown

### Week 1: Data Preparation
- Day 1-2: Download PDB/EMDB structures
- Day 3-5: Generate synthetic ligands and dock
- Day 6-7: Preprocess and extract B-factors

### Week 2: Training
- Day 8-10: Curriculum stages 1-3 (small complexes)
- Day 11-13: Curriculum stages 4-5 (large complexes)
- Day 14: Experimental fine-tuning

### Week 3: Validation & Iteration
- Day 15-17: Evaluate on test sets
- Day 18-19: Hyperparameter tuning
- Day 20-21: Final training run

**Total: 3 weeks end-to-end**

---

## 🎉 Summary

**Question:** How long to train Pearl on full AlphaFoldDB + CryoEM with 8× Blackwell GPUs?

**Answer:**

| Scenario | Time | Cost |
|----------|------|------|
| **Conservative** | 19-20 days | $25,000 |
| **Realistic** | 15-17 days | $20,000 |
| **Optimized** | 7-10 days | $10,000 |

**🎯 Best Estimate: 2-3 weeks with optimizations**

**Key Factors:**
- ✅ 8× Blackwell is excellent hardware (well-suited)
- ✅ Uncertainty weighting adds < 1% overhead
- ⚠️ Bottleneck is synthetic data generation (5 days)
- ⚠️ 64M synthetic complexes may be overkill (can reduce to 10M)
- ✅ Experimental data trains in < 1 hour (tiny but critical)

**Recommendation:** Start with 10M synthetic complexes, train for 1 week, evaluate, then scale up if needed.

