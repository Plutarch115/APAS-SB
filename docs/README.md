# APAS-SB Documentation

This directory contains organized documentation for the APAS-SB (Advanced Protein Analysis System with Structure-Based Learning) project.

## 📁 Documentation Structure

### Root Directory
Essential project documentation:
- **[README.md](../README.md)**: Main project overview and quick start
- **[APAS-SB_Development_Roadmap.md](../APAS-SB_Development_Roadmap.md)**: 85-day training plan for Oracle Cloud (64 H100 GPUs)
- **[IMPLEMENTATION_COMPLETE_SUMMARY.md](../IMPLEMENTATION_COMPLETE_SUMMARY.md)**: Summary of completed implementation (Steps 1-5)

### 📖 Guides (`guides/`)
User-facing guides for getting started and deploying the system:
- **QUICKSTART.md**: Quick start guide for basic usage
- **QUICK_START_DDG.md**: Guide for ΔΔG prediction tasks
- **QUICK_START_UNCERTAINTY.md**: Guide for uncertainty-aware training
- **SUPERCOMPUTER_DEPLOYMENT_GUIDE.md**: Deployment guide for HPC systems
- **UNIFIED_PEARL_TRAINING_GUIDE.md**: Comprehensive training guide

### 🏗️ Architecture (`architecture/`)
Technical architecture and design documents:
- **DENSITY_AWARE_PEARL_ARCHITECTURE.md**: Electron density-aware architecture
- **PEARL_DDG_PREDICTION_EXTENSION.md**: ΔΔG prediction extension design
- **BOLTZ2_ACTUAL_DATASETS.md**: Boltz-2 dataset specifications
- **DATA_PIPELINE.md**: Data processing pipeline architecture
- **MD_SIMULATION_INTEGRATION.md**: MD trajectory integration design

### 📊 Summaries (`summaries/`)
Cost analysis, scaling studies, and reference tables:
- **EXECUTIVE_SUMMARY_COSTS.md**: Cost analysis for different training strategies
- **ENSEMBLE_PEARL_SCALING_ANALYSIS.md**: Scaling analysis for ensemble training
- **EXTREME_SCALE_TRAINING_ANALYSIS.md**: Analysis of extreme-scale training (10K+ GPUs)
- **DENSITY_MD_UNIFIED_COST_ANALYSIS.md**: Unified cost analysis for MD-enhanced training
- **HYBRID_DATASET_SIZE_ESTIMATES.md**: Dataset size estimates for hybrid approach
- **TRAINING_TIME_ESTIMATES.md**: Training time estimates for different configurations
- **SCALING_SUMMARY_TABLE.md**: Quick reference table for scaling
- **QUICK_REFERENCE_TABLE.md**: Quick reference for key metrics
- **TRAINING_TIME_QUICK_REFERENCE.md**: Quick reference for training times

### 📦 Archive (`archive/`)
Historical documents and old summaries (kept for reference):
- Implementation summaries from earlier development phases
- Experiment results and demonstrations
- Old planning documents

## 🚀 Quick Navigation

### For New Users
1. Start with [README.md](../README.md)
2. Follow [guides/QUICKSTART.md](guides/QUICKSTART.md)
3. Review [APAS-SB_Development_Roadmap.md](../APAS-SB_Development_Roadmap.md)

### For Developers
1. Review [architecture/](architecture/) for technical design
2. Check [IMPLEMENTATION_COMPLETE_SUMMARY.md](../IMPLEMENTATION_COMPLETE_SUMMARY.md) for current status
3. See [summaries/](summaries/) for cost and scaling analysis

### For Deployment
1. Read [guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md](guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md)
2. Review [APAS-SB_Development_Roadmap.md](../APAS-SB_Development_Roadmap.md)
3. Check [summaries/EXECUTIVE_SUMMARY_COSTS.md](summaries/EXECUTIVE_SUMMARY_COSTS.md)

## 📝 Key Documents by Topic

### Training & Deployment
- [APAS-SB_Development_Roadmap.md](../APAS-SB_Development_Roadmap.md) - 85-day training plan
- [guides/UNIFIED_PEARL_TRAINING_GUIDE.md](guides/UNIFIED_PEARL_TRAINING_GUIDE.md) - Comprehensive training guide
- [guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md](guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md) - HPC deployment

### Datasets & Data Processing
- [architecture/BOLTZ2_ACTUAL_DATASETS.md](architecture/BOLTZ2_ACTUAL_DATASETS.md) - Boltz-2 datasets
- [architecture/DATA_PIPELINE.md](architecture/DATA_PIPELINE.md) - Data pipeline
- [summaries/HYBRID_DATASET_SIZE_ESTIMATES.md](summaries/HYBRID_DATASET_SIZE_ESTIMATES.md) - Dataset sizes

### Cost & Scaling Analysis
- [summaries/EXECUTIVE_SUMMARY_COSTS.md](summaries/EXECUTIVE_SUMMARY_COSTS.md) - Cost analysis
- [summaries/ENSEMBLE_PEARL_SCALING_ANALYSIS.md](summaries/ENSEMBLE_PEARL_SCALING_ANALYSIS.md) - Scaling analysis
- [summaries/TRAINING_TIME_ESTIMATES.md](summaries/TRAINING_TIME_ESTIMATES.md) - Time estimates

### Architecture & Design
- [architecture/DENSITY_AWARE_PEARL_ARCHITECTURE.md](architecture/DENSITY_AWARE_PEARL_ARCHITECTURE.md) - Density-aware design
- [architecture/PEARL_DDG_PREDICTION_EXTENSION.md](architecture/PEARL_DDG_PREDICTION_EXTENSION.md) - ΔΔG prediction
- [architecture/MD_SIMULATION_INTEGRATION.md](architecture/MD_SIMULATION_INTEGRATION.md) - MD integration

## 🔄 Recent Updates

### Latest Implementation (Steps 1-5)
See [IMPLEMENTATION_COMPLETE_SUMMARY.md](../IMPLEMENTATION_COMPLETE_SUMMARY.md) for:
- ✅ 7 Boltz-2 datasets implemented
- ✅ 3 Boltz-2 loss functions implemented
- ✅ Download infrastructure for all datasets
- ✅ mdCATH and ATLAS MD trajectory loaders
- ✅ Oracle Cloud training scripts (64 H100 GPUs)

## 📧 Contributing

When adding new documentation:
1. Place user guides in `guides/`
2. Place technical architecture in `architecture/`
3. Place analysis and summaries in `summaries/`
4. Update this README with links to new documents

## 📄 License

See the main [README.md](../README.md) for license information.

