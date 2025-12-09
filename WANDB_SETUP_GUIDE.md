# Weights & Biases Integration Guide

Complete guide for setting up and using W&B experiment tracking with APAS-SB.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Local Testing](#local-testing)
5. [Full Training](#full-training)
6. [Oracle Cloud Deployment](#oracle-cloud-deployment)
7. [Monitoring & Visualization](#monitoring--visualization)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install W&B and other dependencies
pip install wandb pyyaml h5py

# Or install all requirements
pip install -r pearl/requirements.txt
```

### 2. Login to W&B

```bash
# Login to your W&B account
wandb login

# Or set API key as environment variable
export WANDB_API_KEY=your_api_key_here
```

### 3. Run Quick Test

```bash
# Test W&B integration (takes ~2 minutes)
python scripts/test_wandb_integration.py
```

This will:
- ✅ Initialize W&B
- ✅ Load synthetic datasets
- ✅ Create and log model architecture
- ✅ Run 3 training epochs
- ✅ Save and upload checkpoint
- ✅ Provide W&B dashboard URL

---

## 📦 Installation

### Option 1: Using pip

```bash
pip install wandb>=0.16.0
```

### Option 2: Using conda

```bash
conda install -c conda-forge wandb
```

### Verify Installation

```bash
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"
```

---

## ⚙️ Configuration

### 1. Edit W&B Config

Edit `scripts/wandb_config.yaml`:

```yaml
wandb:
  project: "apas-sb"  # Your project name
  entity: "your-username"  # Your W&B username or team
  tags:
    - "pearl"
    - "binding-affinity"
```

### 2. Set Data Paths

```yaml
data_root: "./data"  # Path to your data directory
use_synthetic: true  # Set to false when using real data
```

### 3. Configure Training Phases

The config includes 3 training phases aligned with the development roadmap:

- **Phase 2A**: Baseline (48 GPUs, 13 days)
- **Phase 2B**: Multi-task (56 GPUs, 17 days)
- **Phase 2C**: Uncertainty-aware (64 GPUs, 20 days)

---

## 🧪 Local Testing

### Test 1: Quick Integration Test

```bash
# Run all integration tests (~2 minutes)
python scripts/test_wandb_integration.py
```

**What it tests:**
- W&B initialization
- Dataset loading and statistics
- Model creation and architecture logging
- Training loop with metrics
- Checkpoint saving and artifacts

### Test 2: Small Training Run

```bash
# Train for 10 epochs with synthetic data
python scripts/train_with_wandb.py \
    --config scripts/wandb_config.yaml \
    --phase 2a \
    --test
```

**What it does:**
- Loads small synthetic dataset (1000 samples)
- Trains for 10 epochs
- Logs metrics every 10 batches
- Saves checkpoints
- Takes ~5-10 minutes on GPU

### Test 3: Single GPU Training

```bash
# Full training on single GPU
python scripts/train_with_wandb.py \
    --config scripts/wandb_config.yaml \
    --phase 2a
```

---

## 🎯 Full Training

### Single Node (8 GPUs)

```bash
# Phase 2A: Baseline training
torchrun --nproc_per_node=8 \
    scripts/train_with_wandb.py \
    --config scripts/wandb_config.yaml \
    --phase 2a
```

### Multi-Node (64 GPUs)

```bash
# On each node, run:
torchrun \
    --nproc_per_node=8 \
    --nnodes=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_with_wandb.py \
    --config scripts/wandb_config.yaml \
    --phase 2c
```

**Environment Variables:**
- `NODE_RANK`: 0-7 (node index)
- `MASTER_ADDR`: IP of rank 0 node
- `MASTER_PORT`: Port for communication (e.g., 29500)

---

## ☁️ Oracle Cloud Deployment

### 1. Setup OCI Instance

```bash
# SSH into your OCI instance
ssh -i ~/.ssh/oci_key ubuntu@<instance-ip>

# Clone repository
git clone https://github.com/acadev/APAS-SB.git
cd APAS-SB
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r pearl/requirements.txt

# Login to W&B
wandb login
```

### 3. Configure for OCI

Edit `scripts/wandb_config.yaml`:

```yaml
data_root: "/mnt/data"  # OCI block storage mount point
checkpoint_dir: "/mnt/checkpoints"
use_synthetic: false  # Use real data
```

### 4. Launch Training

```bash
# Single node test
python scripts/train_with_wandb.py \
    --config scripts/wandb_config.yaml \
    --phase 2a

# Multi-node (use OCI job submission system)
# See docs/guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md
```

---

## 📊 Monitoring & Visualization

### W&B Dashboard Features

Once training starts, view your dashboard at: `https://wandb.ai/<username>/apas-sb`

**Metrics Tracked:**

1. **Training Metrics**
   - Loss (total, huber, ranking, focal)
   - Learning rate
   - Gradient norms
   - Epoch progress

2. **System Metrics**
   - GPU memory usage
   - GPU utilization
   - Training speed (samples/sec)

3. **Model Metrics**
   - Parameter count
   - Model size
   - Architecture details

4. **Dataset Statistics**
   - Dataset sizes
   - Sample distributions
   - Task weights

### Key Visualizations

- **Loss Curves**: Track training progress
- **Learning Rate Schedule**: Verify warmup and decay
- **Gradient Norms**: Monitor training stability
- **GPU Memory**: Optimize batch sizes
- **System Metrics**: Identify bottlenecks

### Custom Plots

Create custom plots in W&B dashboard:
- Loss by dataset
- Performance by task type
- Training efficiency metrics

---

## 🔧 Troubleshooting

### Issue: W&B login fails

```bash
# Set API key manually
export WANDB_API_KEY=your_key_here

# Or use offline mode
export WANDB_MODE=offline
```

### Issue: Out of memory

Reduce batch size in config:

```yaml
training:
  phase_2a:
    batch_size_per_gpu: 2  # Reduce from 4
```

### Issue: Slow logging

Increase log interval:

```yaml
log_interval: 50  # Log every 50 batches instead of 10
```

---

## 📝 Next Steps

1. ✅ Run `test_wandb_integration.py` to verify setup
2. ✅ Test with `--test` flag for quick validation
3. ✅ Run Phase 2A training locally
4. ✅ Deploy to Oracle Cloud for full-scale training
5. ✅ Monitor progress on W&B dashboard

For more details, see:
- [Development Roadmap](APAS-SB_Development_Roadmap.md)
- [Deployment Guide](docs/guides/SUPERCOMPUTER_DEPLOYMENT_GUIDE.md)
- [Quick Start](docs/guides/QUICKSTART.md)

