#!/usr/bin/env python3
"""
Prepare training data for Pearl by organizing and preprocessing structures.

This script:
1. Loads PDB and synthetic data
2. Preprocesses and featurizes structures
3. Organizes data for curriculum training
4. Saves preprocessed data for efficient training
"""

import sys
import os
sys.path.insert(0, 'pearl')

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings

# Import Pearl data modules
from data import (
    PDBDataset,
    ProteinFeaturizer,
    LigandFeaturizer,
    ComplexPreprocessor,
    CroppingStrategy,
)


def load_synthetic_data(synthetic_dir: Path) -> List[Dict]:
    """Load synthetic data from directory."""
    metadata_file = synthetic_dir / "metadata.json"
    
    if not metadata_file.exists():
        print(f"⚠️  Warning: No metadata file found at {metadata_file}")
        return []
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata.get('structures', [])


def preprocess_dataset(
    structures: List[Dict],
    preprocessor: ComplexPreprocessor,
    dataset_name: str,
) -> List[Dict]:
    """Preprocess a dataset of structures."""
    print(f"\n⚙️  Preprocessing {dataset_name} dataset...")
    print(f"   Total structures: {len(structures)}")
    
    preprocessed = []
    failed = 0
    
    for i, structure in enumerate(structures):
        try:
            # Skip structures with no atoms
            if (len(structure.get('protein_coords', [])) == 0 or 
                len(structure.get('ligand_coords', [])) == 0):
                failed += 1
                continue
            
            # Preprocess
            processed = preprocessor.preprocess(structure)
            processed['dataset_source'] = dataset_name
            preprocessed.append(processed)
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(structures)}...")
                
        except Exception as e:
            failed += 1
            if failed <= 3:  # Only show first few errors
                print(f"   ⚠️  Failed to preprocess structure {i}: {e}")
    
    print(f"   ✓ Successfully preprocessed: {len(preprocessed)}/{len(structures)}")
    if failed > 0:
        print(f"   ⚠️  Failed: {failed}")
    
    return preprocessed


def main():
    """Prepare training data."""
    print("=" * 80)
    print("Pearl Training Data Preparation")
    print("=" * 80)
    
    # Configuration
    pdb_dir = Path("data/pdb_files")
    synthetic_dir = Path("data/synthetic")
    output_dir = Path("data/processed")
    
    print(f"\n📁 PDB directory: {pdb_dir.absolute()}")
    print(f"📁 Synthetic directory: {synthetic_dir.absolute()}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load PDB data
    print("\n" + "=" * 80)
    print("Step 1: Loading PDB Data")
    print("=" * 80)
    
    try:
        pdb_dataset = PDBDataset(
            pdb_dir=str(pdb_dir),
            max_protein_atoms=10000,
            max_ligand_atoms=200,
        )
        print(f"✓ Loaded {len(pdb_dataset)} PDB structures")
        
        # Convert to list
        pdb_structures = []
        for i in range(len(pdb_dataset)):
            structure = pdb_dataset[i]
            # Only keep structures with both protein and ligand
            if (len(structure.get('protein_coords', [])) > 0 and 
                len(structure.get('ligand_coords', [])) > 0):
                pdb_structures.append(structure)
        
        print(f"✓ Valid PDB structures: {len(pdb_structures)}")
        
    except Exception as e:
        print(f"❌ Error loading PDB data: {e}")
        pdb_structures = []
    
    # Step 2: Load synthetic data
    print("\n" + "=" * 80)
    print("Step 2: Loading Synthetic Data")
    print("=" * 80)
    
    try:
        synthetic_structures = load_synthetic_data(synthetic_dir)
        print(f"✓ Loaded {len(synthetic_structures)} synthetic structures")
    except Exception as e:
        print(f"❌ Error loading synthetic data: {e}")
        synthetic_structures = []
    
    # Check if we have any data
    if len(pdb_structures) == 0 and len(synthetic_structures) == 0:
        print("\n❌ Error: No data available for training")
        return 1
    
    # Step 3: Setup preprocessing
    print("\n" + "=" * 80)
    print("Step 3: Setting Up Preprocessing")
    print("=" * 80)
    
    protein_featurizer = ProteinFeaturizer(feature_dim=64)
    ligand_featurizer = LigandFeaturizer(feature_dim=64)
    
    # Create preprocessors for different curriculum stages
    preprocessors = {
        'stage1': ComplexPreprocessor(
            protein_featurizer=protein_featurizer,
            ligand_featurizer=ligand_featurizer,
            cropping_strategy=CroppingStrategy(max_atoms=100, strategy='pocket_centered'),
            normalize_coords=True,
            augment=False,  # No augmentation for validation
        ),
        'stage3': ComplexPreprocessor(
            protein_featurizer=protein_featurizer,
            ligand_featurizer=ligand_featurizer,
            cropping_strategy=CroppingStrategy(max_atoms=200, strategy='pocket_centered'),
            normalize_coords=True,
            augment=False,
        ),
        'stage5': ComplexPreprocessor(
            protein_featurizer=protein_featurizer,
            ligand_featurizer=ligand_featurizer,
            cropping_strategy=CroppingStrategy(max_atoms=1000, strategy='pocket_centered'),
            normalize_coords=True,
            augment=False,
        ),
    }
    
    print("✓ Created preprocessors for curriculum stages")
    
    # Step 4: Preprocess data
    print("\n" + "=" * 80)
    print("Step 4: Preprocessing Data")
    print("=" * 80)
    
    # Preprocess each dataset for each stage
    processed_data = {}
    
    for stage_name, preprocessor in preprocessors.items():
        print(f"\n--- {stage_name.upper()} ---")
        
        stage_data = {}
        
        if len(pdb_structures) > 0:
            stage_data['pdb'] = preprocess_dataset(
                pdb_structures, preprocessor, 'pdb'
            )
        
        if len(synthetic_structures) > 0:
            stage_data['synthetic'] = preprocess_dataset(
                synthetic_structures, preprocessor, 'synthetic'
            )
        
        processed_data[stage_name] = stage_data
    
    # Step 5: Save preprocessed data
    print("\n" + "=" * 80)
    print("Step 5: Saving Preprocessed Data")
    print("=" * 80)
    
    for stage_name, stage_data in processed_data.items():
        stage_dir = output_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        
        for dataset_name, structures in stage_data.items():
            if len(structures) > 0:
                output_file = stage_dir / f"{dataset_name}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(structures, f)
                print(f"✓ Saved {len(structures)} {dataset_name} structures to {output_file}")
    
    # Step 6: Create training manifest
    print("\n" + "=" * 80)
    print("Step 6: Creating Training Manifest")
    print("=" * 80)
    
    manifest = {
        'stages': {},
        'total_structures': 0,
    }
    
    for stage_name, stage_data in processed_data.items():
        stage_info = {
            'datasets': {},
            'total': 0,
        }
        
        for dataset_name, structures in stage_data.items():
            stage_info['datasets'][dataset_name] = len(structures)
            stage_info['total'] += len(structures)
        
        manifest['stages'][stage_name] = stage_info
        manifest['total_structures'] += stage_info['total']
    
    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Training manifest saved to {manifest_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for stage_name, stage_info in manifest['stages'].items():
        print(f"\n{stage_name.upper()}:")
        for dataset_name, count in stage_info['datasets'].items():
            print(f"  {dataset_name}: {count} structures")
        print(f"  Total: {stage_info['total']} structures")
    
    print(f"\nTotal preprocessed structures: {manifest['total_structures']}")
    
    # Final message
    print("\n" + "=" * 80)
    print("✅ Training Data Preparation Complete!")
    print("=" * 80)
    print(f"\n📁 Preprocessed data saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python scripts/train_pearl.py")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

