"""
Visualization script for ΔΔG prediction results.

Generates Boltz-2 style plots for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr

from pearl.models.mock_pearl import MockPearl
from pearl.models.ddg_predictor import PearlWithDDG
from pearl.data.ddg_dataset import SyntheticDDGDataset, collate_ddg_batch
from torch.utils.data import DataLoader

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_ddg_correlation(
    pred_ddg: np.ndarray,
    exp_ddg: np.ndarray,
    confidence: np.ndarray,
    save_path: str
):
    """Generate correlation plot"""
    # Compute metrics
    r_pearson, p_pearson = pearsonr(pred_ddg, exp_ddg)
    r_spearman, p_spearman = spearmanr(pred_ddg, exp_ddg)
    rmse = np.sqrt(np.mean((pred_ddg - exp_ddg) ** 2))
    mae = np.mean(np.abs(pred_ddg - exp_ddg))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot with confidence as alpha
    scatter = ax.scatter(exp_ddg, pred_ddg, 
                        c=confidence, cmap='viridis',
                        alpha=0.6, s=50, edgecolors='navy', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(exp_ddg.min(), pred_ddg.min())
    max_val = max(exp_ddg.max(), pred_ddg.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', lw=2, label='Perfect prediction', zorder=10)
    
    # ±1 kcal/mol error bands
    ax.fill_between([min_val, max_val], 
                    [min_val - 1, max_val - 1],
                    [min_val + 1, max_val + 1],
                    alpha=0.2, color='gray', label='±1 kcal/mol')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence (kcal/mol)', fontsize=14)
    
    # Labels and title
    ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=16, fontweight='bold')
    
    title = 'ΔΔG Prediction Performance\n'
    title += f'Pearson R = {r_pearson:.3f} (p < {p_pearson:.1e})\n'
    title += f'Spearman ρ = {r_spearman:.3f}\n'
    title += f'RMSE = {rmse:.2f} kcal/mol, MAE = {mae:.2f} kcal/mol'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Legend
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    # Grid
    ax.grid(alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_training_history(history: dict, save_path: str):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE (kcal/mol)', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Pearson R
    axes[1, 0].plot(epochs, history['val_pearson_r'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Pearson R', fontsize=12)
    axes[1, 0].set_title('Validation Correlation', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # RMSE
    axes[1, 1].plot(epochs, history['val_rmse'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('RMSE (kcal/mol)', fontsize=12)
    axes[1, 1].set_title('Validation RMSE', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_confidence_calibration(
    pred_ddg: np.ndarray,
    exp_ddg: np.ndarray,
    confidence: np.ndarray,
    save_path: str,
    n_bins: int = 10
):
    """Plot confidence calibration"""
    # Compute actual errors
    errors = np.abs(pred_ddg - exp_ddg)
    
    # Bin by confidence
    confidence_bins = np.percentile(confidence, np.linspace(0, 100, n_bins + 1))
    
    mean_confidence = []
    mean_error = []
    std_error = []
    
    for i in range(n_bins):
        mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
        if mask.sum() > 0:
            mean_confidence.append(confidence[mask].mean())
            mean_error.append(errors[mask].mean())
            std_error.append(errors[mask].std())
    
    mean_confidence = np.array(mean_confidence)
    mean_error = np.array(mean_error)
    std_error = np.array(std_error)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Calibration curve
    ax1.errorbar(mean_confidence, mean_error, yerr=std_error, 
                fmt='o-', markersize=8, linewidth=2, capsize=5,
                label='Observed', color='steelblue')
    
    # Perfect calibration line
    max_val = max(mean_confidence.max(), mean_error.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect calibration')
    
    ax1.set_xlabel('Predicted Confidence (kcal/mol)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual Error (kcal/mol)', fontsize=14, fontweight='bold')
    ax1.set_title('Confidence Calibration', fontsize=16, pad=15)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Confidence distribution
    ax2.hist(confidence, bins=50, alpha=0.7, color='steelblue', edgecolor='navy')
    ax2.axvline(x=confidence.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {confidence.mean():.2f}')
    ax2.axvline(x=errors.mean(), color='green', linestyle='--', 
               linewidth=2, label=f'Actual MAE = {errors.mean():.2f}')
    
    ax2.set_xlabel('Predicted Confidence (kcal/mol)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=16, pad=15)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def evaluate_model(model, data_loader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_trues = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = model(
                wt_protein_features=batch['wt_protein_features'],
                wt_ligand_features=batch['wt_ligand_features'],
                mut_protein_features=batch['mut_protein_features'],
                mut_ligand_features=batch['mut_ligand_features'],
                protein_mask=batch['protein_mask'],
                ligand_mask=batch['ligand_mask']
            )
            
            all_preds.extend(outputs['ddg'].cpu().numpy())
            all_trues.extend(batch['ddg_true'].cpu().numpy())
            all_confidences.extend(outputs['ddg_confidence'].cpu().numpy())
    
    return np.array(all_preds), np.array(all_trues), np.array(all_confidences)


def main():
    """Main visualization function"""
    print("=" * 80)
    print("ΔΔG Prediction Visualization")
    print("=" * 80)
    
    # Configuration
    results_dir = Path('./results/ddg_training')
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load training history
    print("\nLoading training history...")
    with open(results_dir / 'training_history.json', 'r') as f:
        history = json.load(f)
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, viz_dir / 'training_history.png')
    
    # Load model
    print("\nLoading trained model...")
    base_pearl = MockPearl(
        protein_feature_dim=64,
        ligand_feature_dim=64,
        pair_dim=128,
        trunk_blocks=4,
        trunk_heads=8
    )
    model = PearlWithDDG(base_pearl, freeze_pearl=True)
    
    checkpoint = torch.load(results_dir / 'best_ddg_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = SyntheticDDGDataset(num_samples=500, seed=123)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_ddg_batch)
    
    # Evaluate
    print("\nEvaluating model...")
    pred_ddg, exp_ddg, confidence = evaluate_model(model, test_loader, device)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_ddg_correlation(pred_ddg, exp_ddg, confidence, viz_dir / 'ddg_correlation.png')
    plot_confidence_calibration(pred_ddg, exp_ddg, confidence, viz_dir / 'confidence_calibration.png')
    
    # Print final metrics
    print("\n" + "=" * 80)
    print("Final Test Set Metrics:")
    print("=" * 80)
    r_pearson, _ = pearsonr(pred_ddg, exp_ddg)
    r_spearman, _ = spearmanr(pred_ddg, exp_ddg)
    rmse = np.sqrt(np.mean((pred_ddg - exp_ddg) ** 2))
    mae = np.mean(np.abs(pred_ddg - exp_ddg))
    
    print(f"  Pearson R: {r_pearson:.4f}")
    print(f"  Spearman ρ: {r_spearman:.4f}")
    print(f"  RMSE: {rmse:.4f} kcal/mol")
    print(f"  MAE: {mae:.4f} kcal/mol")
    print(f"  Mean Confidence: {confidence.mean():.4f} kcal/mol")
    print("=" * 80)
    
    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    main()

