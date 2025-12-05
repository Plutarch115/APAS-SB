"""
Visualization tools for ΔΔG prediction analysis.

Generates plots similar to those in the Boltz-2 paper for comparing
predicted vs experimental ΔΔG values, analyzing errors, and visualizing
per-residue contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple, Optional
import pandas as pd
from matplotlib.patches import Rectangle


# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300


# ============================================================================
# 1. CORRELATION PLOTS (Main Figure)
# ============================================================================

def plot_ddg_correlation(
    pred_ddg: np.ndarray,
    exp_ddg: np.ndarray,
    dataset_name: str = "Test Set",
    confidence: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate scatter plot of predicted vs experimental ΔΔG.
    
    This is the main figure showing model performance, similar to
    Figure 3 in the Boltz-2 paper.
    
    Args:
        pred_ddg: Predicted ΔΔG values (kcal/mol)
        exp_ddg: Experimental ΔΔG values (kcal/mol)
        dataset_name: Name of the dataset
        confidence: Optional confidence intervals for error bars
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Compute metrics
    r_pearson, p_pearson = pearsonr(pred_ddg, exp_ddg)
    r_spearman, p_spearman = spearmanr(pred_ddg, exp_ddg)
    rmse = np.sqrt(np.mean((pred_ddg - exp_ddg) ** 2))
    mae = np.mean(np.abs(pred_ddg - exp_ddg))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot with optional error bars
    if confidence is not None:
        ax.errorbar(exp_ddg, pred_ddg, yerr=confidence, 
                   fmt='o', alpha=0.5, markersize=5, 
                   elinewidth=1, capsize=2, label='Predictions')
    else:
        ax.scatter(exp_ddg, pred_ddg, alpha=0.6, s=30, 
                  c='steelblue', edgecolors='navy', linewidth=0.5)
    
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
    
    # Labels and title
    ax.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=16, fontweight='bold')
    
    # Title with metrics
    title = f'{dataset_name}\n'
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
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# 2. ERROR DISTRIBUTION BY MUTATION TYPE
# ============================================================================

def categorize_mutation(mutation: str) -> str:
    """
    Categorize mutation by amino acid properties.
    
    Args:
        mutation: Mutation string (e.g., "L99A")
        
    Returns:
        Category string
    """
    # Amino acid categories
    hydrophobic = set('AILMFVPW')
    polar = set('STNQ')
    charged_pos = set('KR')
    charged_neg = set('DE')
    aromatic = set('FYW')
    small = set('AGST')
    
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    
    # Determine categories
    def get_category(aa):
        if aa in hydrophobic:
            return 'Hydrophobic'
        elif aa in polar:
            return 'Polar'
        elif aa in charged_pos:
            return 'Positive'
        elif aa in charged_neg:
            return 'Negative'
        elif aa in aromatic:
            return 'Aromatic'
        elif aa in small:
            return 'Small'
        else:
            return 'Other'
    
    wt_cat = get_category(wt_aa)
    mut_cat = get_category(mut_aa)
    
    # Special cases
    if wt_aa == mut_aa:
        return 'Wild-type'
    elif wt_cat == mut_cat:
        return f'{wt_cat} → {mut_cat}'
    else:
        return f'{wt_cat} → {mut_cat}'


def plot_error_by_mutation_type(
    mutations: List[str],
    pred_ddg: np.ndarray,
    exp_ddg: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Box plot of prediction errors by mutation type.
    
    Similar to Boltz-2 supplementary figures showing performance
    breakdown by mutation characteristics.
    
    Args:
        mutations: List of mutation strings
        pred_ddg: Predicted ΔΔG values
        exp_ddg: Experimental ΔΔG values
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Create dataframe
    df = pd.DataFrame({
        'mutation': mutations,
        'pred_ddg': pred_ddg,
        'exp_ddg': exp_ddg,
        'error': pred_ddg - exp_ddg,
        'abs_error': np.abs(pred_ddg - exp_ddg)
    })
    
    # Categorize mutations
    df['mutation_type'] = df['mutation'].apply(categorize_mutation)
    
    # Sort by median error
    mutation_order = df.groupby('mutation_type')['abs_error'].median().sort_values().index
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute error
    sns.boxplot(data=df, x='mutation_type', y='abs_error', 
               order=mutation_order, ax=ax1, palette='Set2')
    ax1.set_xlabel('Mutation Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Absolute Error (kcal/mol)', fontsize=14, fontweight='bold')
    ax1.set_title('Prediction Error by Mutation Type', fontsize=16, pad=15)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add sample counts
    for i, mut_type in enumerate(mutation_order):
        count = len(df[df['mutation_type'] == mut_type])
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', va='top', fontsize=10)
    
    # Plot 2: Signed error (bias)
    sns.violinplot(data=df, x='mutation_type', y='error', 
                  order=mutation_order, ax=ax2, palette='Set2')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Mutation Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Signed Error (kcal/mol)', fontsize=14, fontweight='bold')
    ax2.set_title('Prediction Bias by Mutation Type', fontsize=16, pad=15)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# 3. PER-RESIDUE CONTRIBUTION PLOT
# ============================================================================

def plot_residue_contributions(
    structure_id: str,
    residue_contrib: np.ndarray,
    sequence: str,
    mutation_site: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar plot of per-residue contributions to ΔΔG.
    
    Shows which residues contribute most to the binding affinity change.
    
    Args:
        structure_id: Structure identifier
        residue_contrib: Per-residue ΔΔG contributions (kcal/mol)
        sequence: Protein sequence
        mutation_site: Index of mutation site (to highlight)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_residues = len(sequence)
    residue_numbers = np.arange(n_residues)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 5))
    
    # Color by contribution (red = destabilizing, blue = stabilizing)
    colors = ['red' if c > 0 else 'blue' for c in residue_contrib]
    alphas = np.abs(residue_contrib) / (np.abs(residue_contrib).max() + 1e-6)
    
    # Bar plot
    bars = ax.bar(residue_numbers, residue_contrib, color=colors, alpha=0.7)
    
    # Highlight mutation site
    if mutation_site is not None:
        bars[mutation_site].set_edgecolor('gold')
        bars[mutation_site].set_linewidth(3)
        ax.axvline(x=mutation_site, color='gold', linestyle='--', 
                  linewidth=2, alpha=0.5, label='Mutation site')
    
    # Zero line
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    
    # Labels
    ax.set_xlabel('Residue Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('ΔΔG Contribution (kcal/mol)', fontsize=14, fontweight='bold')
    ax.set_title(f'Per-Residue ΔΔG Contributions: {structure_id}', 
                fontsize=16, pad=15)
    
    # X-axis labels (every 10th residue)
    tick_positions = residue_numbers[::10]
    tick_labels = [f'{sequence[i]}{i+1}' for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=10)
    
    # Legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Destabilizing')
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='Stabilizing')
    ax.legend(handles=[red_patch, blue_patch], fontsize=12, loc='upper right')
    
    # Grid
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# 4. CONFIDENCE CALIBRATION PLOT
# ============================================================================

def plot_confidence_calibration(
    pred_ddg: np.ndarray,
    exp_ddg: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot showing how well confidence estimates match actual errors.
    
    A well-calibrated model should have predicted confidence equal to
    actual error magnitude.
    
    Args:
        pred_ddg: Predicted ΔΔG values
        exp_ddg: Experimental ΔΔG values
        confidence: Predicted confidence intervals
        n_bins: Number of bins for calibration curve
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Compute actual errors
    errors = np.abs(pred_ddg - exp_ddg)
    
    # Bin by confidence
    confidence_bins = np.percentile(confidence, np.linspace(0, 100, n_bins + 1))
    
    mean_confidence = []
    mean_error = []
    std_error = []
    counts = []
    
    for i in range(n_bins):
        mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
        if mask.sum() > 0:
            mean_confidence.append(confidence[mask].mean())
            mean_error.append(errors[mask].mean())
            std_error.append(errors[mask].std())
            counts.append(mask.sum())
    
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
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# 5. COMPARISON WITH BASELINE METHODS
# ============================================================================

def plot_method_comparison(
    methods: List[str],
    pearson_r: List[float],
    rmse: List[float],
    speed: List[float],  # GPU-hours per prediction
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare Ensemble PEARL with baseline methods.
    
    Args:
        methods: List of method names
        pearson_r: Pearson correlation for each method
        rmse: RMSE for each method (kcal/mol)
        speed: Computational cost (GPU-hours)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    # Plot 1: Pearson R
    bars1 = ax1.barh(methods, pearson_r, color=colors)
    ax1.set_xlabel('Pearson R', fontsize=14, fontweight='bold')
    ax1.set_title('Correlation with Experiment', fontsize=16, pad=15)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # Add values on bars
    for i, (method, r) in enumerate(zip(methods, pearson_r)):
        ax1.text(r + 0.02, i, f'{r:.3f}', va='center', fontsize=11)
    
    # Plot 2: RMSE
    bars2 = ax2.barh(methods, rmse, color=colors)
    ax2.set_xlabel('RMSE (kcal/mol)', fontsize=14, fontweight='bold')
    ax2.set_title('Prediction Error', fontsize=16, pad=15)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_xaxis()  # Lower is better
    
    # Add values on bars
    for i, (method, r) in enumerate(zip(methods, rmse)):
        ax2.text(r - 0.1, i, f'{r:.2f}', va='center', ha='right', fontsize=11)
    
    # Plot 3: Speed (log scale)
    bars3 = ax3.barh(methods, speed, color=colors)
    ax3.set_xlabel('Computational Cost (GPU-hours)', fontsize=14, fontweight='bold')
    ax3.set_title('Speed', fontsize=16, pad=15)
    ax3.set_xscale('log')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_xaxis()  # Lower is better
    
    # Add values on bars
    for i, (method, s) in enumerate(zip(methods, speed)):
        if s < 1:
            label = f'{s*3600:.0f}s'
        else:
            label = f'{s:.1f}h'
        ax3.text(s * 0.7, i, label, va='center', ha='right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# 6. GENERATE ALL PLOTS (Main Function)
# ============================================================================

def generate_all_plots(
    results_dict: Dict,
    output_dir: str = './plots'
) -> None:
    """
    Generate all plots for ΔΔG prediction analysis.
    
    Args:
        results_dict: Dictionary with all results
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating ΔΔG prediction plots...")
    
    # Plot 1: Main correlation plot
    print("  1. Correlation plot...")
    fig1 = plot_ddg_correlation(
        pred_ddg=results_dict['pred_ddg'],
        exp_ddg=results_dict['exp_ddg'],
        dataset_name=results_dict.get('dataset_name', 'Test Set'),
        confidence=results_dict.get('confidence'),
        save_path=f'{output_dir}/ddg_correlation.png'
    )
    plt.close(fig1)
    
    # Plot 2: Error by mutation type
    print("  2. Error by mutation type...")
    fig2 = plot_error_by_mutation_type(
        mutations=results_dict['mutations'],
        pred_ddg=results_dict['pred_ddg'],
        exp_ddg=results_dict['exp_ddg'],
        save_path=f'{output_dir}/error_by_mutation_type.png'
    )
    plt.close(fig2)
    
    # Plot 3: Confidence calibration
    if 'confidence' in results_dict:
        print("  3. Confidence calibration...")
        fig3 = plot_confidence_calibration(
            pred_ddg=results_dict['pred_ddg'],
            exp_ddg=results_dict['exp_ddg'],
            confidence=results_dict['confidence'],
            save_path=f'{output_dir}/confidence_calibration.png'
        )
        plt.close(fig3)
    
    print(f"All plots saved to {output_dir}/")

