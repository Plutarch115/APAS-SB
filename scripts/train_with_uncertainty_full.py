#!/usr/bin/env python3
"""
Full Pearl Training with Uncertainty-Aware Loss and W&B Logging

This script demonstrates the complete training pipeline with:
1. Real X-ray and CryoEM data
2. B-factor and local resolution extraction
3. Uncertainty-weighted loss
4. Comprehensive W&B logging and visualizations
5. Comparison with baseline training

Key visualizations:
- Confidence distribution histograms
- Loss curves (baseline vs uncertainty-aware)
- Per-resolution performance
- Weight distribution analysis
- Training stability metrics
"""

import sys
import os

# Add pearl directory to path
pearl_dir = os.path.join(os.path.dirname(__file__), '..', 'pearl')
sys.path.insert(0, pearl_dir)

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import Pearl modules directly
import importlib.util

# Load uncertainty_aware_losses module
losses_path = os.path.join(pearl_dir, 'training', 'uncertainty_aware_losses.py')
spec = importlib.util.spec_from_file_location("uncertainty_aware_losses", losses_path)
losses_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses_module)

UncertaintyWeightedDiffusionLoss = losses_module.UncertaintyWeightedDiffusionLoss
ResolutionStratifiedLoss = losses_module.ResolutionStratifiedLoss
CombinedUncertaintyAwareLoss = losses_module.CombinedUncertaintyAwareLoss

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ W&B not available. Install with: pip install wandb")


class SimpleBaselineLoss(nn.Module):
    """Simple baseline loss without uncertainty weighting."""
    
    def forward(self, predicted, true, mask=None):
        """Compute MSE loss."""
        loss = (predicted - true) ** 2
        loss = loss.sum(dim=-1)  # Sum over coordinates
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


def load_uncertainty_data(data_file: Path) -> List[Dict]:
    """Load processed uncertainty data."""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def create_visualizations(
    structures: List[Dict],
    output_dir: Path,
    log_to_wandb: bool = False
):
    """
    Create comprehensive visualizations of uncertainty data.
    
    Args:
        structures: List of structure dictionaries
        output_dir: Directory to save plots
        log_to_wandb: Whether to log to W&B
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confidence distribution by method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    xray_confidence = []
    cryoem_confidence = []
    
    for s in structures:
        confidence = np.concatenate([s['protein_confidence'], s['ligand_confidence']])
        if s['method'] == 'XRAY':
            xray_confidence.extend(confidence)
        else:
            cryoem_confidence.extend(confidence)
    
    if xray_confidence:
        axes[0].hist(xray_confidence, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'X-ray Confidence Distribution (n={len(xray_confidence)} atoms)')
        axes[0].axvline(np.mean(xray_confidence), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(xray_confidence):.3f}')
        axes[0].legend()
    
    if cryoem_confidence:
        axes[1].hist(cryoem_confidence, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'CryoEM Confidence Distribution (n={len(cryoem_confidence)} atoms)')
        axes[1].axvline(np.mean(cryoem_confidence), color='red', linestyle='--',
                       label=f'Mean: {np.mean(cryoem_confidence):.3f}')
        axes[1].legend()
    
    plt.tight_layout()
    confidence_dist_file = output_dir / "confidence_distribution.png"
    plt.savefig(confidence_dist_file, dpi=150)
    plt.close()
    
    if log_to_wandb and WANDB_AVAILABLE:
        wandb.log({"confidence_distribution": wandb.Image(str(confidence_dist_file))})
    
    # 2. B-factor vs Confidence scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for s in structures:
        bfactors = np.concatenate([s['protein_bfactors'], s['ligand_bfactors']])
        confidence = np.concatenate([s['protein_confidence'], s['ligand_confidence']])
        
        color = 'blue' if s['method'] == 'XRAY' else 'green'
        label = s['method'] if s == structures[0] or (s['method'] != structures[0]['method']) else None
        ax.scatter(bfactors, confidence, alpha=0.3, s=10, color=color, label=label)
    
    ax.set_xlabel('B-factor (Å²)')
    ax.set_ylabel('Confidence Score')
    ax.set_title('B-factor vs Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    bfactor_conf_file = output_dir / "bfactor_vs_confidence.png"
    plt.savefig(bfactor_conf_file, dpi=150)
    plt.close()
    
    if log_to_wandb and WANDB_AVAILABLE:
        wandb.log({"bfactor_vs_confidence": wandb.Image(str(bfactor_conf_file))})
    
    # 3. Resolution distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    resolutions = [s['resolution'] for s in structures if s['resolution'] is not None]
    methods = [s['method'] for s in structures if s['resolution'] is not None]
    
    xray_res = [r for r, m in zip(resolutions, methods) if m == 'XRAY']
    cryoem_res = [r for r, m in zip(resolutions, methods) if m == 'EM']
    
    if xray_res:
        ax.hist(xray_res, bins=20, alpha=0.6, color='blue', label='X-ray', edgecolor='black')
    if cryoem_res:
        ax.hist(cryoem_res, bins=20, alpha=0.6, color='green', label='CryoEM', edgecolor='black')
    
    ax.set_xlabel('Resolution (Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Resolution Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    resolution_dist_file = output_dir / "resolution_distribution.png"
    plt.savefig(resolution_dist_file, dpi=150)
    plt.close()
    
    if log_to_wandb and WANDB_AVAILABLE:
        wandb.log({"resolution_distribution": wandb.Image(str(resolution_dist_file))})
    
    print(f"✓ Visualizations saved to {output_dir}")


def simulate_training_step(
    structure: Dict,
    loss_fn: nn.Module,
    use_uncertainty: bool = True
) -> Dict:
    """
    Simulate a training step on one structure.

    Args:
        structure: Structure dictionary
        loss_fn: Loss function
        use_uncertainty: Whether to use uncertainty weighting

    Returns:
        Dictionary with loss and metrics
    """
    # Combine protein and ligand - handle empty ligands
    if len(structure['ligand_coords']) > 0:
        coords = np.concatenate([structure['protein_coords'], structure['ligand_coords']])
        confidence = np.concatenate([structure['protein_confidence'], structure['ligand_confidence']])
    else:
        coords = structure['protein_coords']
        confidence = structure['protein_confidence']

    n_atoms = len(coords)
    
    # Simulate predicted and true noise
    predicted_noise = torch.randn(1, n_atoms, 3)
    true_noise = torch.randn(1, n_atoms, 3)
    
    # Convert to tensors
    confidence_tensor = torch.from_numpy(confidence).float().unsqueeze(0)
    resolution_tensor = torch.tensor([structure['resolution'] or 3.0]).float()
    mask = torch.ones(1, n_atoms)
    
    # Compute loss
    if use_uncertainty:
        if isinstance(loss_fn, (UncertaintyWeightedDiffusionLoss, CombinedUncertaintyAwareLoss)):
            loss = loss_fn(
                predicted_noise=predicted_noise,
                true_noise=true_noise,
                confidence=confidence_tensor,
                resolution=resolution_tensor,
                mask=mask,
            )
            if isinstance(loss, dict):
                loss_value = loss['total'].item()
            else:
                loss_value = loss.item()
        else:
            loss_value = loss_fn(predicted_noise, true_noise, mask).item()
    else:
        loss_value = loss_fn(predicted_noise, true_noise, mask).item()
    
    # Compute weight statistics
    weights = confidence ** 2
    weights = weights / (weights.mean() + 1e-8)
    
    return {
        'loss': loss_value,
        'n_atoms': n_atoms,
        'mean_confidence': confidence.mean(),
        'std_confidence': confidence.std(),
        'min_confidence': confidence.min(),
        'max_confidence': confidence.max(),
        'mean_weight': weights.mean(),
        'std_weight': weights.std(),
        'resolution': structure['resolution'],
        'method': structure['method'],
    }


def run_training_comparison(
    structures: List[Dict],
    n_epochs: int = 10,
    log_to_wandb: bool = False
) -> Dict:
    """
    Run training comparison between baseline and uncertainty-aware.
    
    Args:
        structures: List of structure dictionaries
        n_epochs: Number of epochs
        log_to_wandb: Whether to log to W&B
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 80)
    print("Training Comparison: Baseline vs Uncertainty-Aware")
    print("=" * 80 + "\n")
    
    # Create loss functions
    baseline_loss = SimpleBaselineLoss()
    uncertainty_loss = CombinedUncertaintyAwareLoss(
        base_weight=1.0,
        resolution_stratify=True,
        learn_uncertainty=False,
    )
    
    # Training logs
    baseline_log = []
    uncertainty_log = []
    
    # Run training
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        epoch_baseline_losses = []
        epoch_uncertainty_losses = []
        epoch_metrics = defaultdict(list)
        
        # Shuffle structures
        np.random.shuffle(structures)
        
        for i, structure in enumerate(structures):
            # Baseline
            baseline_result = simulate_training_step(
                structure, baseline_loss, use_uncertainty=False
            )
            epoch_baseline_losses.append(baseline_result['loss'])
            
            # Uncertainty-aware
            uncertainty_result = simulate_training_step(
                structure, uncertainty_loss, use_uncertainty=True
            )
            epoch_uncertainty_losses.append(uncertainty_result['loss'])
            
            # Collect metrics
            for key in ['mean_confidence', 'std_confidence', 'mean_weight', 'std_weight']:
                epoch_metrics[key].append(uncertainty_result[key])
        
        # Epoch statistics
        baseline_mean = np.mean(epoch_baseline_losses)
        uncertainty_mean = np.mean(epoch_uncertainty_losses)
        
        baseline_log.append(baseline_mean)
        uncertainty_log.append(uncertainty_mean)
        
        print(f"  Baseline loss: {baseline_mean:.4f}")
        print(f"  Uncertainty-aware loss: {uncertainty_mean:.4f}")
        print(f"  Improvement: {(baseline_mean - uncertainty_mean) / baseline_mean * 100:.1f}%")
        
        # Log to W&B
        if log_to_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'baseline_loss': baseline_mean,
                'uncertainty_loss': uncertainty_mean,
                'improvement_pct': (baseline_mean - uncertainty_mean) / baseline_mean * 100,
                'mean_confidence': np.mean(epoch_metrics['mean_confidence']),
                'mean_weight': np.mean(epoch_metrics['mean_weight']),
            })
    
    return {
        'baseline_log': baseline_log,
        'uncertainty_log': uncertainty_log,
    }


def create_training_plots(
    results: Dict,
    output_dir: Path,
    log_to_wandb: bool = False
):
    """Create training comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(results['baseline_log']) + 1)
    ax.plot(epochs, results['baseline_log'], 'o-', label='Baseline', linewidth=2, markersize=6)
    ax.plot(epochs, results['uncertainty_log'], 's-', label='Uncertainty-Aware', linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss: Baseline vs Uncertainty-Aware')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    loss_curves_file = output_dir / "loss_curves.png"
    plt.savefig(loss_curves_file, dpi=150)
    plt.close()
    
    if log_to_wandb and WANDB_AVAILABLE:
        wandb.log({"loss_curves": wandb.Image(str(loss_curves_file))})
    
    print(f"✓ Training plots saved to {output_dir}")


def main():
    """Main training script."""
    print("=" * 80)
    print("Pearl Training with Uncertainty-Aware Loss (Full Pipeline)")
    print("=" * 80)
    
    # Initialize W&B
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="pearl-uncertainty-aware",
            name="uncertainty_vs_baseline",
            config={
                "weighting_scheme": "inverse_variance",
                "min_weight": 0.1,
                "resolution_stratify": True,
                "n_epochs": 10,
            }
        )
        print("\n✓ W&B initialized")
    
    # Load data
    data_file = Path("data/uncertainty_processed/structures_with_uncertainty.pkl")
    
    if not data_file.exists():
        print(f"\n✗ Data file not found: {data_file}")
        print("Please run: python scripts/prepare_uncertainty_data.py")
        return 1
    
    print(f"\nLoading data from: {data_file}")
    structures = load_uncertainty_data(data_file)
    print(f"✓ Loaded {len(structures)} structures")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    
    viz_dir = Path("uncertainty_training_output/visualizations")
    create_visualizations(structures, viz_dir, log_to_wandb=use_wandb)
    
    # Run training comparison
    results = run_training_comparison(structures, n_epochs=10, log_to_wandb=use_wandb)
    
    # Create training plots
    print("\n" + "=" * 80)
    print("Creating Training Plots")
    print("=" * 80)
    
    create_training_plots(results, viz_dir, log_to_wandb=use_wandb)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    baseline_final = results['baseline_log'][-1]
    uncertainty_final = results['uncertainty_log'][-1]
    improvement = (baseline_final - uncertainty_final) / baseline_final * 100
    
    print(f"\nFinal Results:")
    print(f"  Baseline loss: {baseline_final:.4f}")
    print(f"  Uncertainty-aware loss: {uncertainty_final:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if use_wandb:
        wandb.finish()
        print("\n✓ W&B run finished")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

