#!/usr/bin/env python3
"""
Generate synthetic protein-ligand complexes for Pearl training.

This script uses the downloaded PDB structures to generate synthetic
training data using virtual ligands and physics-based docking.
"""

import sys
import os
sys.path.insert(0, 'pearl')

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings

# Import Pearl data modules
from data import (
    PDBDataset,
    VirtualLigandLibrary,
    PhysicsBasedDocker,
    SyntheticDataGenerator,
)


def main():
    """Generate synthetic training data."""
    print("=" * 80)
    print("Pearl Synthetic Data Generator")
    print("=" * 80)
    
    # Configuration
    pdb_dir = Path("data/pdb_files")
    output_dir = Path("data/synthetic")
    n_ligands_per_protein = 20  # Small number for testing (paper uses 640)
    
    print(f"\n📁 PDB directory: {pdb_dir.absolute()}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"🧪 Ligands per protein: {n_ligands_per_protein}")
    
    # Check if PDB directory exists
    if not pdb_dir.exists():
        print(f"\n❌ Error: PDB directory not found: {pdb_dir}")
        print("   Please run: python scripts/download_pdb_subset.py")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load PDB structures
    print("\n" + "=" * 80)
    print("Step 1: Loading PDB Structures")
    print("=" * 80)
    
    try:
        dataset = PDBDataset(
            pdb_dir=str(pdb_dir),
            max_protein_atoms=10000,
            max_ligand_atoms=200,
        )
        print(f"✓ Loaded {len(dataset)} PDB structures")
    except Exception as e:
        print(f"❌ Error loading PDB structures: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if len(dataset) == 0:
        print("❌ No valid structures found")
        return 1
    
    # Show sample
    sample = dataset[0]
    print(f"\n📊 Sample structure: {sample['pdb_id']}")
    print(f"   Protein atoms: {len(sample['protein_coords'])}")
    print(f"   Ligand atoms: {len(sample['ligand_coords'])}")
    
    # Step 2: Create virtual ligand library
    print("\n" + "=" * 80)
    print("Step 2: Creating Virtual Ligand Library")
    print("=" * 80)
    
    try:
        # Try to use RDKit if available
        ligand_library = VirtualLigandLibrary(
            min_heavy_atoms=10,
            max_heavy_atoms=50,
            diversity_threshold=0.3,
        )
        print(f"✓ Created virtual ligand library")
        print(f"   Library size: {len(ligand_library.ligands)} ligands")
        
        if len(ligand_library.ligands) == 0:
            print("⚠️  Warning: No ligands in library, will use fallback method")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not create full ligand library: {e}")
        print("   This is expected if RDKit is not installed")
        print("   Will use simplified ligand generation")
        ligand_library = VirtualLigandLibrary()
    
    # Step 3: Setup docking engine
    print("\n" + "=" * 80)
    print("Step 3: Setting Up Docking Engine")
    print("=" * 80)
    
    docker = PhysicsBasedDocker(method='random_placement')
    print("✓ Docking engine ready (method: random_placement)")
    print("   Note: Using fast random placement for testing")
    print("   For production, consider using AutoDock Vina")
    
    # Step 4: Generate synthetic dataset
    print("\n" + "=" * 80)
    print("Step 4: Generating Synthetic Complexes")
    print("=" * 80)
    
    # Convert dataset to list of structures
    protein_structures = []
    for i in range(len(dataset)):
        structure = dataset[i]
        # Compute pocket center from ligand
        if len(structure['ligand_coords']) > 0:
            pocket_center = np.mean(structure['ligand_coords'], axis=0)
        else:
            pocket_center = np.mean(structure['protein_coords'], axis=0)
        
        structure['pocket_center'] = pocket_center
        protein_structures.append(structure)
    
    print(f"✓ Prepared {len(protein_structures)} protein structures")
    
    # Create generator
    generator = SyntheticDataGenerator(
        protein_structures=protein_structures,
        ligand_library=ligand_library,
        docker=docker,
        n_ligands_per_protein=n_ligands_per_protein,
    )
    
    # Generate dataset
    print(f"\n⚙️  Generating synthetic complexes...")
    print(f"   Target: {len(protein_structures)} proteins × {n_ligands_per_protein} ligands")
    print(f"   = {len(protein_structures) * n_ligands_per_protein} total structures")
    print()
    
    try:
        synthetic_data = generator.generate_dataset(
            output_dir=str(output_dir),
            max_structures=len(protein_structures) * n_ligands_per_protein,
        )
        
        print(f"\n✓ Generated {len(synthetic_data)} synthetic structures")
        
    except Exception as e:
        print(f"\n❌ Error generating synthetic data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Save summary statistics
    print("\n" + "=" * 80)
    print("Step 5: Saving Statistics")
    print("=" * 80)
    
    # Compute statistics
    docking_scores = [s['docking_score'] for s in synthetic_data]
    protein_sizes = [len(s['protein_coords']) for s in synthetic_data]
    ligand_sizes = [len(s['ligand_coords']) for s in synthetic_data]
    
    stats = {
        'n_structures': len(synthetic_data),
        'n_proteins': len(protein_structures),
        'n_ligands_per_protein': n_ligands_per_protein,
        'docking_scores': {
            'mean': float(np.mean(docking_scores)),
            'std': float(np.std(docking_scores)),
            'min': float(np.min(docking_scores)),
            'max': float(np.max(docking_scores)),
        },
        'protein_atoms': {
            'mean': float(np.mean(protein_sizes)),
            'std': float(np.std(protein_sizes)),
            'min': int(np.min(protein_sizes)),
            'max': int(np.max(protein_sizes)),
        },
        'ligand_atoms': {
            'mean': float(np.mean(ligand_sizes)),
            'std': float(np.std(ligand_sizes)),
            'min': int(np.min(ligand_sizes)),
            'max': int(np.max(ligand_sizes)),
        },
    }
    
    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to: {stats_file}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total structures: {stats['n_structures']}")
    print(f"   Proteins: {stats['n_proteins']}")
    print(f"   Ligands per protein: {stats['n_ligands_per_protein']}")
    print(f"   Avg docking score: {stats['docking_scores']['mean']:.2f} ± {stats['docking_scores']['std']:.2f}")
    print(f"   Avg protein atoms: {stats['protein_atoms']['mean']:.0f} ± {stats['protein_atoms']['std']:.0f}")
    print(f"   Avg ligand atoms: {stats['ligand_atoms']['mean']:.0f} ± {stats['ligand_atoms']['std']:.0f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Synthetic Data Generation Complete!")
    print("=" * 80)
    print(f"\n📁 Synthetic data saved to: {output_dir.absolute()}")
    print(f"📊 Generated {len(synthetic_data)} synthetic structures")
    print("\nNext steps:")
    print("  1. Run: python scripts/prepare_training_data.py")
    print("  2. Run: python scripts/train_pearl.py")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

