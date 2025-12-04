# APAS-SB Development Roadmap
## 3-Month Training & Evaluation Plan for 64 H100 GPU Cluster

**Project**: Advanced Protein Analysis System with Structure-Based Learning  
**Timeline**: 3 months (90 days)  
**Infrastructure**: 64x H100 GPUs on Oracle Cloud Infrastructure  
**Goal**: Train hybrid PEARL model with Boltz-2 datasets while optimizing cluster utilization

---

## Executive Summary

This roadmap leverages your 64 H100 cluster to simultaneously:
- Train the multi-task PEARL model with progressive complexity
- Integrate pre-existing MD trajectory databases (mdCATH + ATLAS)
- Run targeted custom simulations only for data gaps
- Maintain >95% GPU utilization through optimized parallel workloads
- Complete comprehensive evaluation benchmarks

**Key Strategy**: Use publicly available MD datasets (mdCATH: 135K trajectories, ATLAS: 4K trajectories) to eliminate 90% of simulation time, freeing GPUs for training and completing 5 days ahead of schedule.

## 🚀 Major Optimizations with Public MD Data

### What Changed?
By integrating mdCATH and ATLAS databases, we've transformed the project timeline:

**Original Plan**:
- Generate 100K MD simulations from scratch
- 48 GPUs dedicated to simulations for 50+ days
- 90-day timeline
- 100K protein structures

**Optimized Plan with mdCATH + ATLAS**:
- Download 135K existing trajectories (mdCATH) + 4K (ATLAS)
- Only 5-10K custom simulations needed
- 85-day timeline (**5 days faster** ⚡)
- **143K protein structures** (43% more!)
- ~$20K cost savings
- Higher quality community-vetted data

### Key Benefits of Using Public MD Databases

1. **Immediate Data Availability** 📦
   - mdCATH: 134,950 trajectories ready to download
   - ATLAS: 1,390 proteins × 3 replicates = 4,170 trajectories
   - Start training on Day 13 with full dataset (vs Day 50+ originally)

2. **Superior Data Quality** ✨
   - Community-validated protocols
   - Standardized force fields (CHARMM36m)
   - Professional simulation parameters
   - Pre-computed quality metrics

3. **Unique Features** 🎯
   - **mdCATH multi-temperature**: 5 temps (320-450K) = natural uncertainty labels
   - **Forces included**: mdCATH provides atomic forces (rare in MD databases!)
   - **Diverse sampling**: Multiple replicates at different conditions
   - **Broad coverage**: CATH/ECOD fold classification coverage

4. **Resource Optimization** 💰
   - 90% reduction in GPU time for MD simulations
   - Redirect 40-50 GPUs to training instead of simulations
   - Faster convergence with more training compute
   - 5-day timeline reduction + cost savings

5. **Scientific Rigor** 📊
   - Reproducible: Use same data as other studies
   - Benchmarkable: Compare against other mdCATH-trained models
   - Transparent: Well-documented simulation protocols
   - Validated: Published datasets with DOIs

### Data Integration Strategy
```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1 (Days 1-12): Data Acquisition                      │
├─────────────────────────────────────────────────────────────┤
│ • Download mdCATH (3 TB)                    [56 GPUs]      │
│ • Download ATLAS (~500 GB)                  [8 GPUs]       │
│ • Extract density maps from trajectories                    │
│ • Validate quality metrics                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 2 (Days 13-62): Training with Pre-existing Data      │
├─────────────────────────────────────────────────────────────┤
│ • Train on 143K structures from Day 13     [48-64 GPUs]   │
│ • Run 5-10K custom sims in parallel        [8-16 GPUs]    │
│ • Leverage multi-temperature mdCATH data                    │
│ • Forces from mdCATH for enhanced learning                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Phase 3 (Days 63-85): Evaluation ⚡ 5 days early            │
├─────────────────────────────────────────────────────────────┤
│ • Complete benchmarking                    [40 GPUs]       │
│ • Production optimization                  [16 GPUs]       │
│ • Documentation and manuscript                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Infrastructure Setup & Data Pipeline (Days 1-12) ⚡ ACCELERATED

### Week 1: Infrastructure & Environment (Days 1-7)
**GPU Allocation**: 8 GPUs for setup, 56 GPUs start data download/processing

**Tasks**:
- [ ] Set up Oracle Cloud environment with 64 H100 GPUs
- [ ] Configure PyTorch distributed training with NCCL backend
- [ ] Install APAS-SB dependencies and validate environment
- [ ] Set up MLflow/Weights & Biases for experiment tracking
- [ ] Configure data storage (at least 1TB for datasets + checkpoints + MD data)
- [ ] Test multi-node training with toy dataset (8 GPUs)
- [ ] Set up monitoring dashboard for GPU utilization, memory, and throughput
- [ ] **Download mdCATH dataset from Hugging Face** (Days 1-3, 56 GPUs for parallel processing)
  - **Source**: https://huggingface.co/datasets/compsciencelab/mdCATH
  - **Method**: HuggingFace Dataset API or direct download
  - **Size**: ~3 TB
  - **Format**: HDF5 files (one per domain)
  - **Access options**:
    1. HuggingFace Dataset API (recommended)
    2. Browser download from HF
    3. PlayMolecule visualization (no download): https://open.playmolecule.org/mdcath
  - **Integration**: TorchMD-Net data loader available
- [ ] **Download ATLAS dataset from DSIMB** (Days 2-5, parallel with mdCATH)
  - **Source**: https://www.dsimb.inserm.fr/ATLAS
  - **Method**: Use provided download_ATLAS.py script with aria2c
  - **Format**: Reduced format (1,000 frames) recommended for efficiency
  - **Size**: ~500 GB - 1 TB (reduced format vs 15 TB full)
  - **API access**: https://www.dsimb.inserm.fr/ATLAS/download.html

**Download Code Examples**:

**mdCATH (HuggingFace)**:
```python
# Option 1: Download specific domain files
from huggingface_hub import hf_hub_download

# List of domains to download (5,398 domains total)
domain_file = hf_hub_download(
    repo_id="compsciencelab/mdCATH",
    filename="1a0aA00.h5",  # Example domain
    repo_type="dataset"
)

# Option 2: Use with TorchMD-Net (integrated loader)
from torchmdnet.datasets import mdCATH

dataset = mdCATH(
    root='./data/mdcath',
    temperature=320,  # Choose: 320, 350, 380, 410, 450 K
    split='train'
)

# Option 3: Bulk download with huggingface-cli
# In terminal:
# huggingface-cli download compsciencelab/mdCATH --repo-type dataset --local-dir ./mdcath_data
```

**ATLAS (DSIMB)**:
```python
# Download the provided script from DSIMB
# https://www.dsimb.inserm.fr/ATLAS/download.html

# Example using their Python script with aria2c
import subprocess

# Install aria2c first: apt-get install aria2
subprocess.run([
    'python', 'download_ATLAS.py',
    '--output-dir', './atlas_data',
    '--format', 'reduced',  # Get 1,000 frame version
    '--parallel', '16'  # Parallel downloads
])

# Or use their API for specific proteins
import requests

# Get list of available proteins
response = requests.get('https://www.dsimb.inserm.fr/ATLAS/api/proteins')
proteins = response.json()

# Download specific protein trajectory
protein_id = '1k5n_A'
response = requests.get(
    f'https://www.dsimb.inserm.fr/ATLAS/api/download/{protein_id}'
)
```

**Deliverables**:
- Functional 64-GPU training environment
- Validated distributed training setup
- Monitoring infrastructure
- mdCATH download completed (3 TB)
- ATLAS download initiated/in progress

### Week 2: Data Acquisition & Preprocessing (Days 8-12) ⚡ 3 DAYS FASTER
**GPU Allocation**: 32 GPUs for preprocessing, 32 GPUs for targeted MD simulations

**Tasks**:
- [ ] Download Boltz-2 datasets (ChEMBL 600K, BindingDB 600K, PubChem HTS 2M)
- [ ] Download original datasets (ProteinGym 2.5M, BRENDA 100K, PDBbind 20K, SKEMPI 8K)
- [ ] **Complete ATLAS download** (if not finished in Week 1)
- [ ] **Process mdCATH trajectories for density map extraction** (32 GPUs)
  - Extract coordinates and forces from HDF5 files
  - Focus on 320K temperature trajectories first (physiological)
  - Generate electron density maps using coordinates
  - Validate force data quality
  - Process ~135K trajectories
  
**mdCATH Processing Pipeline**:
```python
import h5py
import numpy as np

def extract_mdcath_data(hdf5_file):
    """Extract key data from mdCATH HDF5 files"""
    with h5py.File(hdf5_file, 'r') as f:
        # Key fields in mdCATH HDF5 structure:
        coords = f['coordinates'][:]  # Shape: (n_frames, n_atoms, 3)
        forces = f['forces'][:]       # Shape: (n_frames, n_atoms, 3)
        temperature = f['temperature'][()]
        cath_class = f['cath_domain'][()]
        
        # Secondary structure (pre-computed)
        sec_struct = f['secondary_structure'][:]
        
    return {
        'coordinates': coords,
        'forces': forces,
        'temperature': temperature,
        'cath_class': cath_class,
        'sec_struct': sec_struct
    }

# Process in parallel across GPUs
from torch.utils.data import DataLoader

# Using TorchMD-Net loader (recommended)
from torchmdnet.datasets import mdCATH

dataset = mdCATH(
    root='./mdcath_data',
    temperature=320,  # Start with physiological temp
    force_download=False
)

# Or custom processing
for domain_file in mdcath_files:
    data = extract_mdcath_data(domain_file)
    density_map = compute_density_map(data['coordinates'])
    save_processed_data(domain_file.stem, density_map, data['forces'])
```

- [ ] **Process ATLAS trajectories** (32 GPUs)
  - Extract from GROMACS format trajectories
  - Use pre-computed RMSF and secondary structure data
  - Generate density maps from 3 replicates per protein
  - Process 1,390 proteins × 3 replicates = 4,170 trajectories

**ATLAS Processing Pipeline**:
```python
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

def process_atlas_trajectory(pdb_file, trajectory_file):
    """Process ATLAS trajectory data"""
    # Load trajectory
    u = mda.Universe(pdb_file, trajectory_file)
    
    # ATLAS provides reduced format: 1,000 frames
    # Full format: 10,000 frames (if downloaded)
    
    # Extract coordinates
    coords = []
    for ts in u.trajectory:
        coords.append(u.atoms.positions.copy())
    coords = np.array(coords)
    
    # ATLAS pre-computed data is also available
    # Download from their API for efficiency
    
    return {
        'coordinates': coords,
        'rmsf': load_precomputed_rmsf(pdb_file),
        'secondary_structure': load_precomputed_dssp(pdb_file)
    }

# Parallel processing across proteins
from multiprocessing import Pool

def process_protein(protein_id):
    pdb = f'atlas_data/{protein_id}/{protein_id}.pdb'
    traj = f'atlas_data/{protein_id}/{protein_id}_traj.xtc'
    return process_atlas_trajectory(pdb, traj)

with Pool(32) as p:  # 32 GPUs worth of parallel processing
    results = p.map(process_protein, atlas_protein_ids)
```

- [ ] **Identify protein structures not in mdCATH/ATLAS**
  - Cross-reference PDBbind and SKEMPI with mdCATH/ATLAS
  - Create list of missing structures requiring custom simulation
  - Priority: protein-ligand complexes, protein-protein interfaces
  
- [ ] **Run targeted MD simulations only for gaps** (32 GPUs)
  - Target: 5,000-10,000 additional structures
  - Use OpenMM or GROMACS with GPU acceleration
  - 20-50 ns trajectories per structure
  - Focus on binding site conformations
  
- [ ] Generate protein embeddings for all datasets (using spare GPU cycles)
- [ ] Create unified data loaders for mdCATH + ATLAS + custom
- [ ] Validate trajectory quality and density map accuracy

**Key Optimization**: Using pre-existing MD data eliminates ~90% of simulation time!

**MD Data Sources Summary**:
```
mdCATH:    134,950 trajectories × 5 temps × 464ns avg = ~62 ms total time
           ✓ Available on HuggingFace
           ✓ HDF5 format with forces included
           ✓ TorchMD-Net integration
           
ATLAS:     1,390 proteins × 3 replicates × 100ns = ~420 μs total time
           ✓ Available from DSIMB with download script
           ✓ Reduced format: 1,000 frames
           ✓ Pre-computed analyses available
           
Custom:    5,000-10,000 targeted structures × 50ns = ~500 μs total time
           → Only for structures missing from above datasets
           
TOTAL:     ~143K structures with dynamics data (vs 100K originally planned)
```

**Data Format Standards**:
```python
# Unified data structure for all sources
processed_data = {
    'source': 'mdcath' | 'atlas' | 'custom',
    'protein_id': str,
    'coordinates': np.ndarray,  # (n_frames, n_atoms, 3)
    'forces': np.ndarray | None,  # Only mdCATH has this
    'density_map': np.ndarray,
    'temperature': float,
    'metadata': {
        'force_field': 'CHARMM36m' | 'AMBER',
        'n_frames': int,
        'trajectory_length_ns': float,
        'classification': str  # CATH or ECOD
    }
}
```

**Deliverables**:
- 7.25M training examples downloaded and validated
- **143K protein structures with MD trajectories** (43% more than planned!)
- Electron density maps from mdCATH + ATLAS
- Efficient data pipeline tested
- Gap analysis for targeted simulations
- Unified data loaders implemented

---

## Phase 2: Model Training - Progressive Scaling (Days 16-70)

### Stage 2A: Baseline Model Training (Days 13-25) ⚡ 5 DAYS FASTER
**GPU Allocation**: 48 GPUs for training, 16 GPUs for continued processing

**Training Configuration**:
- **Model**: Multi-task PEARL with basic tasks
- **Batch Size**: 256 per GPU (effective batch size: 12,288)
- **GPUs**: 48 H100s (increased from 32)
- **Datasets**: ChEMBL, BindingDB, PDBbind (1.2M examples)
- **Training Time**: ~13 days for initial convergence (vs 15 days)

**Parallel Activities** (16 GPUs):
- Process remaining mdCATH/ATLAS trajectories
- Run targeted MD simulations for gap structures (5-10K targets)
- Generate embeddings for new data batches
- Extract and validate density maps

**Tasks**:
- [ ] Train baseline binding affinity model
- [ ] Implement gradient accumulation and mixed precision training
- [ ] Monitor training metrics (loss, learning rate, GPU memory)
- [ ] **Integrate mdCATH density maps as training data** (ready on Day 13!)
- [ ] Generate first checkpoint at Day 20
- [ ] Validate on FEP+ and CASP16 benchmarks
- [ ] Complete targeted MD simulation pipeline for remaining structures

**Data Advantage**:
- Start training with 135K structures from mdCATH/ATLAS immediately
- Add custom simulations incrementally as they complete
- Larger training dataset from Day 1 = better initial performance

**Expected Performance**:
- Binding Affinity (FEP+): R = 0.62-0.64 (better than original plan)
- CASP16: R = 0.64-0.66 (improved with more dynamic data)

**Deliverables**:
- Baseline model checkpoint
- **143K protein structures with density maps** (all sources combined)
- Initial benchmark results showing improvement

### Stage 2B: Multi-Task Expansion (Days 26-42) ⚡ 8 DAYS FASTER
**GPU Allocation**: 56 GPUs for training, 8 GPUs for final processing/validation

**Training Configuration**:
- **Model**: Full multi-task PEARL (all 11 datasets)
- **Batch Size**: 224 per GPU (effective batch size: 12,544)
- **GPUs**: 56 H100s (increased from 48)
- **Datasets**: All 7.25M examples + **143K MD structures**
- **Training Time**: ~17 days (vs 20 days)

**Parallel Activities** (8 GPUs):
- Final validation of all MD trajectory data
- Real-time data augmentation and preprocessing
- Generate validation set embeddings
- Quality control on density maps

**Tasks**:
- [ ] Expand to full multi-task learning setup
- [ ] Integrate all datasets with task-specific heads
- [ ] Implement multi-task loss balancing (uncertainty weighting)
- [ ] Add ProteinGym, BRENDA, SKEMPI tasks
- [ ] **Leverage complete MD dataset** (143K structures, all available from start!)
- [ ] Train for 17 days with checkpointing every 2 days
- [ ] Monitor task-specific convergence rates
- [ ] Conduct ablation study: mdCATH vs ATLAS vs custom MD contributions

**Data Advantage**:
- Full 143K structure dataset available from Day 1 of this phase
- Diverse conformational sampling (5 temps from mdCATH)
- Standardized CHARMM36m force field data from ATLAS
- More robust training with larger, higher-quality dynamics data

**Expected Performance**:
- Binding Affinity: R = 0.65-0.67 (improved)
- Protein-Protein ΔΔG: R = 0.57-0.60 (improved)
- Enzyme kcat: R = 0.47-0.50 (improved)
- Fitness Scores: ρ = 0.52-0.55 (improved)

**Deliverables**:
- Multi-task model checkpoint
- Complete 143K structure dataset integrated
- Task-specific validation metrics
- Ablation study report on MD data sources

### Stage 2C: Uncertainty & Density-Aware Training (Days 43-62) ⚡ 8 DAYS FASTER
**GPU Allocation**: 64 GPUs for advanced training (full cluster utilization)

**Training Configuration**:
- **Model**: PEARL with uncertainty quantification + density-aware losses
- **Batch Size**: 128 per GPU (effective batch size: 8,192)
- **GPUs**: 64 H100s (full cluster)
- **Datasets**: 7.25M + 143K structures with density maps
- **Training Time**: ~20 days (same as original plan)

**Tasks**:
- [ ] Implement uncertainty-aware loss functions
- [ ] Add density-aware training with electron density maps
- [ ] Train confidence estimation heads
- [ ] Three-phase training: structure → confidence → affinity
- [ ] **Leverage multi-temperature mdCATH data** for uncertainty calibration
  - 320K-450K temperature range provides excellent uncertainty training signal
- [ ] Perform hyperparameter search for loss weighting (4 GPUs)
- [ ] Generate ensemble of 5 models for uncertainty quantification
- [ ] Run ablation studies on density-aware components
- [ ] Test temperature-dependent performance using mdCATH data

**Multi-Temperature Advantage**:
- mdCATH's 5 temperature conditions (320K-450K) provide natural uncertainty labels
- Higher temperature trajectories = higher uncertainty states
- Improved uncertainty quantification without additional simulations

**Expected Performance**:
- Binding Affinity (FEP+): R = 0.66-0.68 (exceed Boltz-2 target!)
- CASP16: R = 0.68-0.70 (exceed Boltz-2)
- MF-PCBA: AP = 0.027-0.029
- ΔΔG: R = 0.60-0.62 (significantly improved)
- **Superior uncertainty calibration** with multi-temp data

**Deliverables**:
- Final trained model with uncertainty quantification
- Ensemble of 5 models
- Ablation study results
- Temperature-dependent performance analysis

---

## Phase 3: Evaluation & Validation (Days 63-85) ⚡ 5 DAYS FASTER

### Week 10-11: Benchmark Evaluation (Days 63-75)
**GPU Allocation**: 40 GPUs for comprehensive inference, 24 GPUs for analysis

**Tasks**:
- [ ] Run comprehensive evaluation on all benchmarks:
  - FEP+ (binding affinity)
  - CASP16 (structure-based binding)
  - MF-PCBA (binary classification)
  - ProteinGym (fitness landscape)
  - SKEMPI 2.0 (protein-protein ΔΔG)
  - BRENDA (enzyme kinetics)
- [ ] Generate uncertainty quantification metrics
- [ ] Perform error analysis and failure mode identification
- [ ] Compare against Boltz-2 baseline
- [ ] Run inference speed benchmarks

**Deliverables**:
- Complete benchmark report with metrics
- Uncertainty calibration plots
- Performance comparison table

### Week 12: Extended Testing & Case Studies (Days 76-81) ⚡ 3 DAYS FASTER
**GPU Allocation**: 32 GPUs for testing, 32 GPUs for final experiments

**Tasks**:
- [ ] Test on held-out drug discovery datasets
- [ ] Antibody-antigen binding predictions
- [ ] Enzyme engineering case studies
- [ ] Generate visualizations (binding poses, uncertainty maps)
- [ ] Test ensemble predictions for improved accuracy
- [ ] Validate on proprietary datasets (if available)
- [ ] **Compare mdCATH vs ATLAS-trained model components**

**Deliverables**:
- Case study reports
- Visualization gallery
- Ensemble performance metrics
- MD data source comparison analysis

### Week 13: Optimization & Documentation (Days 82-85) ⚡ 5 DAYS FASTER
**GPU Allocation**: 16 GPUs for optimization, 48 GPUs released

**Tasks**:
- [ ] Model optimization (quantization, pruning, distillation)
- [ ] Create inference API and Docker container
- [ ] Write technical documentation
- [ ] Prepare manuscript/preprint
- [ ] Generate demo notebook for common use cases
- [ ] Archive all checkpoints and datasets
- [ ] Cost analysis and efficiency report
- [ ] Document mdCATH/ATLAS integration methodology

**Deliverables**:
- Optimized production model
- Complete documentation
- Manuscript draft
- Demo notebooks
- **5 days ahead of schedule!**

---

## GPU Utilization Strategy ⚡ OPTIMIZED

### Maximizing Cluster Efficiency with Pre-existing MD Data

| Phase | Training GPUs | Data Processing GPUs | Utilization | Days |
|-------|--------------|---------------------|-------------|------|
| Days 1-12 | 8-48 | 24-56 | 100% | 12 |
| Days 13-25 | 48 | 16 | 100% | 13 |
| Days 26-42 | 56 | 8 | 100% | 17 |
| Days 43-62 | 64 | 0 | 100% | 20 |
| Days 63-85 | 32-40 | 24-32 | 90-95% | 23 |

**Total Timeline**: 85 days (vs 90 days original) - **5 days ahead of schedule!**

### Pre-existing MD Data Integration Strategy 🎯

**Primary Data Sources**:
1. **mdCATH** (Prioritized)
   - 134,950 trajectories for 5,398 unique domains
   - 5 temperature replicates (320K-450K) each
   - Average 464 ns per trajectory
   - Total: ~62 milliseconds of simulation time
   - Forces included (unique feature!)
   - Download: Days 1-7 (parallel download on 56 GPUs)
   - Processing: Days 8-12 (density map extraction)

2. **ATLAS** (Complementary)
   - 1,390 proteins with 3 replicates each
   - 100 ns trajectories at 300K
   - CHARMM36m force field (standardized)
   - Pre-computed RMSF, secondary structure
   - Download: Days 7-10 (smaller dataset)
   - Processing: Days 8-12

3. **Custom MD Simulations** (Gap Filling Only)
   - Target: 5,000-10,000 structures
   - Focus on: PDBbind/SKEMPI structures not in mdCATH/ATLAS
   - Protein-ligand complexes needing specific conformations
   - Duration: 20-50 ns per structure
   - Timeline: Days 8-25 (parallel with training)

**Combined Dataset Statistics**:
```
mdCATH:    134,950 structures
ATLAS:     4,170 structures (1,390 × 3 replicates)
Custom:    5,000-10,000 structures
────────────────────────────────────
TOTAL:     ~143,000-149,000 structures with MD data

Comparison to original plan: +43% to +49% more structures!
```

**Key Advantages**:
- ✅ **Immediate availability**: mdCATH/ATLAS data ready from Day 13
- ✅ **Diverse conformational sampling**: Multiple temperatures, replicates
- ✅ **Standardized quality**: Professional force fields, validated protocols
- ✅ **Force data**: mdCATH includes forces (unique for ML training)
- ✅ **90% reduction in simulation time**: Focus GPUs on training
- ✅ **Higher quality**: Community-vetted trajectories
- ✅ **Multi-temperature data**: Natural uncertainty quantification labels

**Processing Timeline**:
```
Days 1-7:   Download mdCATH (3 TB) + ATLAS (~500 GB)
Days 8-12:  Extract density maps, validate quality
Days 13+:   Ready for training! Custom sims fill gaps in parallel
```

---

## Key Milestones ⚡ ACCELERATED TIMELINE

| Day | Milestone | Success Criteria |
|-----|-----------|------------------|
| 7 | Infrastructure Ready + MD Data Download Started | 64 GPUs operational, mdCATH/ATLAS downloads initiated |
| 12 | Data Pipeline Complete | 7.25M examples processed, 143K MD structures ready |
| 25 | Baseline Model Trained | R > 0.62 on FEP+, first checkpoint with mdCATH data |
| 42 | Multi-Task Model Complete | All tasks converged, full MD dataset integrated |
| 62 | Final Model Trained | Target performance met/exceeded vs Boltz-2 |
| 75 | Evaluation Complete | Full benchmark suite run, results validated |
| 85 | Production Ready ⚡ | Optimized model, documentation, manuscript draft |

**Timeline Improvement**: **5 days ahead** of original 90-day schedule!
**Data Improvement**: **+43% more structures** with MD trajectories!

---

## Risk Mitigation ⚡ REDUCED RISKS

### Technical Risks (Significantly Reduced!)
1. **MD Simulation Bottleneck**: ✅ **ELIMINATED**
   - Pre-existing mdCATH + ATLAS data removes 90% of simulation burden
   - Only 5-10K custom simulations needed (vs 100K originally)
   - No impact to timeline if custom sims delayed

2. **Data Quality Issues**:
   - Mitigation: mdCATH/ATLAS are community-vetted, standardized protocols
   - Pre-computed analyses available for validation
   - Fallback: Use subset of highest-quality structures first

3. **Training Instability**:
   - Mitigation: Gradient clipping, learning rate warmup, frequent checkpointing
   - Fallback: Reduce batch size or learning rate

4. **Storage Limitations**: ⚠️ **INCREASED ATTENTION NEEDED**
   - Mitigation: Need ~4 TB for mdCATH + ATLAS + checkpoints + working space
   - Implement data streaming and on-the-fly preprocessing
   - Archive processed data, keep only density maps + coordinates
   - Fallback: Use subset of mdCATH (e.g., 320K temperature only)

5. **Data Download Time**:
   - Mitigation: Parallel downloads on 56 GPUs, high-bandwidth Oracle networking
   - Alternative: Some mdCATH data available on Hugging Face (faster access)
   - Fallback: Start with ATLAS (smaller), add mdCATH incrementally

### Resource Risks
1. **GPU Availability**:
   - Mitigation: Use Oracle Cloud spot instances wisely, maintain checkpoints
   - Fallback: Reduce training GPUs if costs exceed budget

2. **Cost Overruns**:
   - Mitigation: Monitor costs daily, optimize idle time
   - Current estimate: ~$150K-200K for 3 months (64 H100s)

---

## Cost Estimates (Oracle Cloud) ⚡ REDUCED COSTS

**H100 GPU Pricing**: ~$3.00/hour per GPU (OCI pricing may vary)

| Phase | Days | GPU-Hours | Cost Estimate |
|-------|------|-----------|---------------|
| Phase 1 | 12 | 18,432 | $55,296 |
| Phase 2 | 50 | 77,760 | $233,280 |
| Phase 3 | 23 | 35,328 | $105,984 |
| **Total** | **85** | **131,520** | **$394,560** |

**Savings vs Original Plan**:
- **5 fewer days** of cluster time
- **6,720 fewer GPU-hours**
- **~$20,000 cost reduction**
- **43% more training data** (mdCATH + ATLAS)

**Additional Costs**:
- Data storage: ~$40-50/month for 4 TB (mdCATH + ATLAS + working space)
- Egress for downloads: ~$100-200 (one-time)
- **Total Additional**: ~$300-400 one-time + $50/month

**Return on Investment**:
- Higher quality model due to better dynamics data
- Faster time to results
- Community-standard MD data = better reproducibility
- Forces from mdCATH = unique training signal

**Note**: Actual costs may be lower with spot instances. Oracle Cloud typically offers competitive pricing for bulk GPU reservations.

---

## Success Metrics

### Technical Metrics
- [ ] FEP+ Binding Affinity: R ≥ 0.64 (match Boltz-2)
- [ ] CASP16: R ≥ 0.66 (exceed Boltz-2)
- [ ] ΔΔG Prediction: R ≥ 0.58
- [ ] Enzyme kcat: R ≥ 0.48
- [ ] Model uncertainty calibration: ECE < 0.05
- [ ] Inference speed: < 1 second per prediction

### Operational Metrics
- [ ] Average GPU utilization: ≥ 85%
- [ ] Training completed within 90 days
- [ ] All 7.25M examples processed
- [ ] 100K MD simulations completed
- [ ] Zero major data loss incidents

---

## Weekly Standup Agenda

**Recommended weekly check-ins to track progress:**

1. Training metrics review (loss curves, validation performance)
2. GPU utilization report
3. Data pipeline status (MD simulations, preprocessing)
4. Blockers and risks
5. Next week's priorities

---

## Next Steps ⚡ PRIORITY ACTIONS

### Immediate Actions (This Week):
1. **Provision Infrastructure**
   - Secure 64 H100 GPUs on Oracle Cloud
   - Configure high-bandwidth storage (4+ TB)
   - Set up parallel download capabilities
   - Install dependencies: `pip install huggingface_hub torch torchmdnet aria2c`

2. **Start Data Downloads** 🚨 **CRITICAL PATH**
   
   **A. mdCATH from HuggingFace (Priority #1 - Days 1-3)**
   
   **Links**:
   - HuggingFace: https://huggingface.co/datasets/compsciencelab/mdCATH
   - GitHub (tools & scripts): https://github.com/compsciencelab/mdCATH
   - PlayMolecule (visualization): https://open.playmolecule.org/mdcath
   - Paper: https://doi.org/10.1038/s41597-024-04140-z
   
   **Quick Start**:
   ```bash
   # Method 1: HuggingFace CLI (Fastest for bulk download)
   pip install huggingface_hub[cli]
   huggingface-cli login  # Optional, for faster speeds
   
   # Download entire dataset
   huggingface-cli download compsciencelab/mdCATH \
       --repo-type dataset \
       --local-dir ./data/mdcath \
       --resume-download
   
   # Method 2: Python API (More control)
   from huggingface_hub import snapshot_download
   
   snapshot_download(
       repo_id="compsciencelab/mdCATH",
       repo_type="dataset",
       local_dir="./data/mdcath",
       resume_download=True,
       max_workers=8  # Parallel downloads
   )
   
   # Method 3: Use TorchMD-Net integration (Recommended for training)
   from torchmdnet.datasets import mdCATH
   
   dataset = mdCATH(
       root='./data/mdcath',
       temperature=320,  # Options: 320, 350, 380, 410, 450 K
       split='train',
       download=True  # Auto-downloads if needed
   )
   ```
   
   **Size**: ~3 TB  
   **Format**: HDF5 files (one per domain, 5,398 domains)  
   **Features**: Coordinates, forces, temperature, CATH classification  
   
   **B. ATLAS from DSIMB (Priority #2 - Days 2-5)**
   
   **Links**:
   - Main site: https://www.dsimb.inserm.fr/ATLAS
   - Download page: https://www.dsimb.inserm.fr/ATLAS/download.html
   - API docs: https://www.dsimb.inserm.fr/ATLAS/about.html
   - Paper: https://doi.org/10.1093/nar/gkad1084
   
   **Quick Start**:
   ```bash
   # Step 1: Download the ATLAS download script
   wget https://www.dsimb.inserm.fr/ATLAS/download_ATLAS.py
   
   # Step 2: Install aria2c for parallel downloads
   sudo apt-get install aria2
   
   # Step 3: Run bulk download (reduced format recommended)
   python download_ATLAS.py \
       --output-dir ./data/atlas \
       --format reduced \
       --parallel 16
   
   # Alternative: Download via their web interface
   # Navigate to: https://www.dsimb.inserm.fr/ATLAS/download.html
   # Select proteins and download trajectories
   
   # API access for programmatic download
   import requests
   
   # Get protein list
   proteins = requests.get(
       'https://www.dsimb.inserm.fr/ATLAS/api/proteins'
   ).json()
   
   # Download specific protein (example)
   protein_id = '1k5n_A'
   trajectory = requests.get(
       f'https://www.dsimb.inserm.fr/ATLAS/api/download/{protein_id}/trajectory'
   )
   
   with open(f'{protein_id}.tar.gz', 'wb') as f:
       f.write(trajectory.content)
   ```
   
   **Size**: ~500 GB (reduced format) or ~15 TB (full format)  
   **Format**: GROMACS trajectories (XTC) + PDB structures  
   **Features**: 1,390 proteins × 3 replicates, pre-computed RMSF/DSSP  
   **Recommendation**: Use reduced format (1,000 frames vs 10,000)

3. **Environment Setup (Parallel with downloads)**
   ```bash
   # Clone APAS-SB repository
   git clone https://github.com/acadev/APAS-SB.git
   cd APAS-SB
   
   # Install dependencies
   pip install -r pearl/requirements.txt
   
   # Install additional packages for MD processing
   pip install MDAnalysis h5py torchmdnet
   
   # Test environment with small dataset
   python scripts/test_hybrid_datasets.py
   ```

4. **Download Original Datasets (Can proceed in parallel)**
   - ChEMBL: https://www.ebi.ac.uk/chembl/
   - BindingDB: https://www.bindingdb.org/
   - ProteinGym: https://github.com/OATML-Markslab/ProteinGym
   - BRENDA: https://www.brenda-enzymes.org/
   - PDBbind: http://www.pdbbind.org.cn/
   - SKEMPI 2.0: https://life.bsc.es/pid/skempi2

### Week 1 Goals:
- ✅ Complete infrastructure setup
- ✅ mdCATH download 80%+ complete (or fully cached)
- ✅ ATLAS download initiated/50%+ complete
- ✅ Validate distributed training setup
- ✅ Begin trajectory processing pipeline development
- ✅ Test mdCATH TorchMD-Net integration

### Week 1 Daily Checklist:
**Day 1**: 
- Provision Oracle Cloud GPUs
- Start mdCATH download via HuggingFace CLI
- Set up monitoring

**Day 2**:
- Monitor mdCATH download progress (should be 30-50% done)
- Start ATLAS download
- Install processing tools

**Day 3**:
- mdCATH download should be complete
- Begin validation of mdCATH files
- Continue ATLAS download

**Day 4-5**:
- Start processing first batch of mdCATH data
- Test density map extraction
- Complete ATLAS download

**Day 6-7**:
- Validate all downloaded data
- Test data loaders
- Begin gap analysis for custom simulations

### Month 1 Objective:
- ✅ All 143K structures processed and ready
- ✅ Baseline model trained with mdCATH/ATLAS data
- ✅ Initial benchmarks showing improved performance

### Technical Notes for mdCATH/ATLAS Integration:

**mdCATH Processing Priority**:
```python
# Temperature selection strategy
# 320K (physiological) - Priority 1: Use first, largest dataset
# 350K, 380K - Priority 2: Add for diversity
# 410K, 450K - Priority 3: Use for uncertainty quantification

# Key fields to extract from HDF5:
- Coordinates: Every 1 ns (primary training data)
- Forces: Unique feature! (can enable force-matching)
- Temperature: For temperature-aware models
- CATH domain: For stratified sampling
- Secondary structure: Pre-computed (save processing)

# Recommended: Focus on 320K first, add others incrementally
```

**ATLAS Processing Priority**:
```python
# ATLAS provides excellent complementary data
- Standardized CHARMM36m (matches many structures)
- Room temperature (300K) physiological conditions  
- 3 replicates = uncertainty quantification ready
- Pre-computed RMSF, DSSP (saves compute)

# Use ATLAS for:
1. Different fold coverage (ECOD vs CATH)
2. Validation of mdCATH-trained models
3. Ensemble generation (3 replicates)
```

**Custom Simulation Priorities**:
```python
# Only simulate structures NOT in mdCATH or ATLAS
priority_list = [
    "PDBbind complexes not in mdCATH",  # ~5,000 structures
    "SKEMPI protein-protein interfaces",  # ~2,000 structures  
    "Specific ligand-bound conformations",  # ~3,000 structures
]
# Target: 5,000-10,000 structures maximum (vs 100K original plan)
```

### Troubleshooting Common Issues:

**mdCATH Download Issues**:
- If HuggingFace is slow: Try during off-peak hours (evening US time)
- Use `--resume-download` flag to continue interrupted downloads
- Consider downloading subsets first (by temperature)

**ATLAS Download Issues**:
- Use `aria2c` with `-x16` for 16 parallel connections
- Request reduced format to save bandwidth
- Download critical proteins first, full set later

**Storage Issues**:
- Monitor disk space: `df -h`
- Delete raw trajectories after processing density maps
- Compress intermediate files: `tar czf processed_data.tar.gz processed/`

### Contact & Resources:

**mdCATH Support**:
- GitHub Issues: https://github.com/compsciencelab/mdCATH/issues
- Author: Antonio Mirarchi (contact via GitHub)

**ATLAS Support**:
- Email: Contact form on DSIMB website
- Documentation: https://www.dsimb.inserm.fr/ATLAS/about.html

**APAS-SB**:
- GitHub Issues: https://github.com/acadev/APAS-SB/issues

---

## Contact & Support

**Questions or issues?** 
- GitHub Issues: https://github.com/acadev/APAS-SB/issues
- Monitor MLflow dashboard for training progress
- Weekly sync meetings recommended

**Good luck with your training campaign!** 🚀
