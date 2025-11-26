"""
Simple Example: Using Pearl for Protein-Ligand Cofolding

This example demonstrates how to use Pearl for both unconditional
and conditional cofolding modes.
"""

import torch
import sys
sys.path.append('..')

from pearl.models.pearl import Pearl
from pearl.models.templating import Template
from pearl.inference.unconditional import unconditional_cofolding
from pearl.inference.conditional import conditional_cofolding
from pearl.evaluation.metrics import compute_ligand_rmsd, MetricsCalculator


def create_dummy_data():
    """Create dummy data for demonstration."""
    # Protein features (e.g., from sequence encoding)
    n_protein_atoms = 100
    protein_features = torch.randn(1, n_protein_atoms, 64)
    
    # Ligand features (e.g., from molecular graph)
    n_ligand_atoms = 20
    ligand_features = torch.randn(1, n_ligand_atoms, 64)
    
    # Ground truth coordinates (for evaluation)
    true_protein_coords = torch.randn(n_protein_atoms, 3) * 10
    true_ligand_coords = torch.randn(n_ligand_atoms, 3) * 5
    
    return protein_features, ligand_features, true_protein_coords, true_ligand_coords


def example_unconditional_cofolding():
    """
    Example: Unconditional cofolding mode
    
    Predicts structure from sequence and topology alone.
    """
    print("=" * 60)
    print("Example 1: Unconditional Cofolding")
    print("=" * 60)
    
    # Initialize model
    model = Pearl(
        protein_feature_dim=64,
        ligand_feature_dim=64,
        pair_dim=128,
        trunk_blocks=2,  # Reduced for demo
        diffusion_blocks=4,  # Reduced for demo
        num_diffusion_steps=50  # Reduced for demo
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    protein_features, ligand_features, true_protein_coords, true_ligand_coords = create_dummy_data()
    
    print(f"Protein atoms: {protein_features.shape[1]}")
    print(f"Ligand atoms: {ligand_features.shape[1]}")
    
    # Perform unconditional cofolding
    print("\nGenerating structures (unconditional mode)...")
    structures = unconditional_cofolding(
        model=model,
        protein_features=protein_features,
        ligand_features=ligand_features,
        num_samples=5,  # Generate 5 samples
        device='cpu'
    )
    
    print(f"Generated {structures.shape[0]} structures")
    print(f"Structure shape: {structures.shape}")
    
    # Evaluate (using dummy ground truth)
    print("\nEvaluating structures...")
    n_protein = protein_features.shape[1]
    
    for i, structure in enumerate(structures):
        pred_protein = structure[0, :n_protein]
        pred_ligand = structure[0, n_protein:]
        
        rmsd = compute_ligand_rmsd(
            pred_ligand, true_ligand_coords,
            pred_protein, true_protein_coords
        )
        print(f"Sample {i+1} - Ligand RMSD: {rmsd:.2f} Å")
    
    print("\n✓ Unconditional cofolding completed successfully!")
    return model


def example_conditional_cofolding(model):
    """
    Example: Conditional cofolding mode
    
    Uses structural priors (pocket information) to guide prediction.
    """
    print("\n" + "=" * 60)
    print("Example 2: Conditional Cofolding (Pocket-Aware)")
    print("=" * 60)
    
    # Create dummy data
    protein_features, ligand_features, true_protein_coords, true_ligand_coords = create_dummy_data()
    
    # Create a pocket template (e.g., from known binding pocket)
    n_pocket_residues = 20
    pocket_coords = torch.randn(n_pocket_residues, 3) * 10
    pocket_features = torch.randn(n_pocket_residues, 64)
    
    pocket_template = Template(
        protein_coords=pocket_coords,
        protein_features=pocket_features,
        confidence=0.9
    )
    
    print(f"Using pocket template with {n_pocket_residues} residues")
    
    # Perform conditional cofolding
    print("\nGenerating structures (conditional mode)...")
    structures = conditional_cofolding(
        model=model,
        protein_features=protein_features,
        ligand_features=ligand_features,
        pocket_template=pocket_template,
        num_samples=5,
        device='cpu'
    )
    
    print(f"Generated {structures.shape[0]} structures")
    
    # Evaluate
    print("\nEvaluating structures...")
    n_protein = protein_features.shape[1]
    
    for i, structure in enumerate(structures):
        pred_protein = structure[0, :n_protein]
        pred_ligand = structure[0, n_protein:]
        
        rmsd = compute_ligand_rmsd(
            pred_ligand, true_ligand_coords,
            pred_protein, true_protein_coords
        )
        print(f"Sample {i+1} - Ligand RMSD: {rmsd:.2f} Å")
    
    print("\n✓ Conditional cofolding completed successfully!")


def example_metrics_calculation():
    """
    Example: Computing evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Example 3: Metrics Calculation")
    print("=" * 60)
    
    # Create dummy predictions and ground truth
    pred_ligand = torch.randn(20, 3)
    true_ligand = torch.randn(20, 3)
    pred_protein = torch.randn(100, 3)
    true_protein = torch.randn(100, 3)
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(
        rmsd_thresholds=[1.0, 2.0],
        compute_lddt=True
    )
    
    # Compute metrics
    metrics = calculator.compute_all_metrics(
        pred_ligand, true_ligand,
        pred_protein, true_protein
    )
    
    print("\nMetrics for single prediction:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Simulate multiple predictions
    all_metrics = []
    for _ in range(10):
        pred_ligand = torch.randn(20, 3)
        pred_protein = torch.randn(100, 3)
        
        metrics = calculator.compute_all_metrics(
            pred_ligand, true_ligand,
            pred_protein, true_protein
        )
        all_metrics.append(metrics)
    
    # Aggregate
    aggregated = calculator.aggregate_metrics(all_metrics)
    
    print("\nAggregated metrics over 10 predictions:")
    for key, value in aggregated.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✓ Metrics calculation completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Pearl: Placing Every Atom in the Right Location")
    print("Simple Examples")
    print("=" * 60)
    
    # Example 1: Unconditional cofolding
    model = example_unconditional_cofolding()
    
    # Example 2: Conditional cofolding
    example_conditional_cofolding(model)
    
    # Example 3: Metrics calculation
    example_metrics_calculation()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Unconditional cofolding (sequence + topology only)")
    print("  ✓ Conditional cofolding (with pocket information)")
    print("  ✓ Multiple sample generation")
    print("  ✓ Evaluation metrics (RMSD, lDDT-PLI)")
    print("\nFor more advanced usage, see the documentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

