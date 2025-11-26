#!/bin/bash

# Run Density-Aware Pearl Experiment
# This script sets up the environment and runs the comparison experiment

set -e  # Exit on error

echo "=========================================="
echo "Density-Aware Pearl Experiment"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Install dependencies if needed
echo ""
echo "Installing dependencies..."
pip install -q torch numpy biopython mrcfile 2>/dev/null || echo "Dependencies already installed"

# Check if data exists
echo ""
echo "Checking data..."
if [ ! -d "data/cryoem_maps" ]; then
    echo "ERROR: CryoEM maps not found in data/cryoem_maps/"
    exit 1
fi

if [ ! -d "data/cryoem_pdb_files" ]; then
    echo "ERROR: PDB files not found in data/cryoem_pdb_files/"
    exit 1
fi

echo "✓ Data found"

# Create results directory
mkdir -p results/density_aware_experiment

# Run experiment
echo ""
echo "=========================================="
echo "Running experiment..."
echo "=========================================="
echo ""

python scripts/train_density_aware_comparison.py

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/density_aware_experiment/"

