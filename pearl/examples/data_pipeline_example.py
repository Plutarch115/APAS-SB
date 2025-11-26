"""
Example script demonstrating Pearl's data loading and synthetic data generation pipeline.

This script shows how to:
1. Load protein-ligand complexes from PDB files
2. Generate synthetic training data
3. Preprocess and featurize structures
4. Use curriculum-based sampling for training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

# Import Pearl data modules
from data.pdb_loader import PDBDataset, PDBParser
from data.synthetic_generator import (
    VirtualLigandLibrary,
    PhysicsBasedDocker,
    SyntheticDataGenerator
)
from data.preprocessing import (
    ProteinFeaturizer,
    LigandFeaturizer,
    ComplexPreprocessor,
    CroppingStrategy
)
from data.curriculum_sampler import (
    CurriculumSampler,
    CurriculumStage,
    DEFAULT_CURRICULUM
)


def example_1_load_pdb_data():
    """Example 1: Load protein-ligand complexes from PDB files."""
    print("=" * 80)
    print("Example 1: Loading PDB Data")
    print("=" * 80)
    
    # Setup paths (adjust to your PDB directory)
    pdb_dir = "./data/pdb_files"  # Directory containing PDB files
    
    # Check if directory exists
    if not Path(pdb_dir).exists():
        print(f"⚠️  PDB directory not found: {pdb_dir}")
        print("   Please create this directory and add PDB files, or adjust the path.")
        print("   You can download PDB files from: https://www.rcsb.org/")
        return None
    
    # Create dataset
    print(f"\n📂 Loading PDB structures from: {pdb_dir}")
    dataset = PDBDataset(
        pdb_dir=pdb_dir,
        max_protein_atoms=5000,
        max_ligand_atoms=100,
        release_date_cutoff="2021-09-30",  # Pearl's training cutoff
    )
    
    print(f"✅ Found {len(dataset)} PDB structures")
    
    # Load a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n📊 Sample structure: {sample['pdb_id']}")
        print(f"   Protein atoms: {len(sample['protein_coords'])}")
        print(f"   Ligand atoms: {len(sample['ligand_coords'])}")
        print(f"   Protein sequence length: {len(sample['protein_sequence'])}")
        
        return dataset
    else:
        print("⚠️  No structures loaded")
        return None


def example_2_generate_synthetic_data():
    """Example 2: Generate synthetic protein-ligand complexes."""
    print("\n" + "=" * 80)
    print("Example 2: Generating Synthetic Data")
    print("=" * 80)
    
    # Create virtual ligand library
    print("\n🧪 Creating virtual ligand library...")
    ligand_library = VirtualLigandLibrary(
        min_heavy_atoms=10,
        max_heavy_atoms=50,
    )
    print(f"✅ Generated {len(ligand_library.ligands)} virtual ligands")
    
    # Create docking engine
    print("\n🎯 Initializing docking engine...")
    docker = PhysicsBasedDocker(method='random_placement')
    print("✅ Docking engine ready")
    
    # Create some example protein structures
    print("\n🧬 Creating example protein structures...")
    protein_structures = []
    for i in range(3):  # Generate 3 example proteins
        n_atoms = np.random.randint(500, 1000)
        protein = {
            'pdb_id': f'EXAMPLE_{i}',
            'protein_coords': np.random.randn(n_atoms, 3).astype(np.float32) * 20,
            'protein_atoms': ['C'] * n_atoms,
            'protein_sequence': 'A' * (n_atoms // 10),
            'pocket_center': np.array([0.0, 0.0, 0.0]),
        }
        protein_structures.append(protein)
    print(f"✅ Created {len(protein_structures)} example proteins")
    
    # Generate synthetic dataset
    print("\n⚙️  Generating synthetic complexes...")
    generator = SyntheticDataGenerator(
        protein_structures=protein_structures,
        ligand_library=ligand_library,
        docker=docker,
        n_ligands_per_protein=10,  # Small number for demo
    )
    
    synthetic_data = generator.generate_dataset(
        output_dir="./data/synthetic",
        max_structures=30,  # Generate 30 structures for demo
    )
    
    print(f"\n✅ Generated {len(synthetic_data)} synthetic structures")
    print(f"   Average docking score: {np.mean([s['docking_score'] for s in synthetic_data]):.2f}")
    
    return synthetic_data


def example_3_preprocess_data():
    """Example 3: Preprocess and featurize structures."""
    print("\n" + "=" * 80)
    print("Example 3: Preprocessing and Featurization")
    print("=" * 80)
    
    # Create example complex
    print("\n📦 Creating example complex...")
    complex_data = {
        'pdb_id': 'EXAMPLE',
        'protein_coords': np.random.randn(500, 3).astype(np.float32) * 20,
        'protein_atoms': ['C', 'N', 'O', 'S'] * 125,
        'protein_residues': np.repeat(np.arange(50), 10),
        'protein_sequence': 'ACDEFGHIKLMNPQRSTVWY' * 2 + 'ACDEFGHIKL',
        'ligand_coords': np.random.randn(30, 3).astype(np.float32) * 5,
        'ligand_atoms': ['C', 'N', 'O'] * 10,
        'ligand_bonds': [],
    }
    
    # Create preprocessor
    print("\n⚙️  Setting up preprocessor...")
    protein_featurizer = ProteinFeaturizer(feature_dim=64)
    ligand_featurizer = LigandFeaturizer(feature_dim=64)
    cropping_strategy = CroppingStrategy(max_atoms=200, strategy='pocket_centered')
    
    preprocessor = ComplexPreprocessor(
        protein_featurizer=protein_featurizer,
        ligand_featurizer=ligand_featurizer,
        cropping_strategy=cropping_strategy,
        normalize_coords=True,
        augment=True,
    )
    
    # Preprocess
    print("\n🔄 Preprocessing complex...")
    processed = preprocessor.preprocess(complex_data)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Protein atoms: {len(processed['protein'].coords)}")
    print(f"   Protein atom types: {processed['protein'].atom_types.shape}")
    print(f"   Protein residues: {len(processed['protein'].residue_types)}")
    print(f"   Ligand atoms: {len(processed['ligand'].coords)}")
    print(f"   Ligand bonds: {len(processed['ligand'].bonds)}")
    
    return processed


def example_4_curriculum_sampling():
    """Example 4: Curriculum-based data sampling."""
    print("\n" + "=" * 80)
    print("Example 4: Curriculum Training")
    print("=" * 80)
    
    # Create dummy datasets
    print("\n📚 Creating dummy datasets...")
    pdb_data = [{'id': f'pdb_{i}', 'type': 'pdb'} for i in range(100)]
    synthetic_data = [{'id': f'syn_{i}', 'type': 'synthetic'} for i in range(500)]
    distillation_data = [{'id': f'dist_{i}', 'type': 'distillation'} for i in range(50)]
    
    # Create curriculum sampler
    print("\n⚙️  Setting up curriculum sampler...")
    sampler = CurriculumSampler(
        pdb_dataset=pdb_data,
        distillation_dataset=distillation_data,
        synthetic_dataset=synthetic_data,
        curriculum=DEFAULT_CURRICULUM,
    )
    
    print(f"✅ Curriculum sampler ready with {len(DEFAULT_CURRICULUM)} stages")
    
    # Simulate training through curriculum stages
    print("\n🎓 Simulating curriculum training...")
    print("-" * 80)
    
    for stage_idx in range(len(DEFAULT_CURRICULUM)):
        stats = sampler.get_statistics()
        print(f"\n📊 Stage {stats['stage_idx']}/{stats['total_stages']}: {stats['stage']}")
        print(f"   Max atoms: {stats['max_atoms']}")
        print(f"   Synthetic ratio: {stats['synthetic_ratio']:.1%}")
        print(f"   Use templates: {stats['use_templates']}")
        print(f"   Template complexity: {stats['template_complexity']}")
        
        # Sample a few batches
        for step in range(3):
            batch = sampler.sample_batch(batch_size=4)
            if step == 0:
                sources = [s['dataset_source'] for s in batch]
                print(f"   Sample batch sources: {sources}")
        
        # Advance to next stage
        if not sampler.advance_stage():
            break
    
    print("\n✅ Curriculum training simulation complete!")


def example_5_full_pipeline():
    """Example 5: Complete data pipeline."""
    print("\n" + "=" * 80)
    print("Example 5: Complete Data Pipeline")
    print("=" * 80)
    
    print("\n🔄 Running complete data pipeline...")
    
    # 1. Load PDB data (if available)
    print("\n[1/5] Loading PDB data...")
    pdb_dataset = example_1_load_pdb_data()
    
    # 2. Generate synthetic data
    print("\n[2/5] Generating synthetic data...")
    synthetic_data = example_2_generate_synthetic_data()
    
    # 3. Preprocess data
    print("\n[3/5] Preprocessing data...")
    processed = example_3_preprocess_data()
    
    # 4. Setup curriculum sampling
    print("\n[4/5] Setting up curriculum sampling...")
    if pdb_dataset and len(pdb_dataset) > 0:
        pdb_data = [pdb_dataset[i] for i in range(min(10, len(pdb_dataset)))]
    else:
        pdb_data = []
    
    sampler = CurriculumSampler(
        pdb_dataset=pdb_data,
        synthetic_dataset=synthetic_data[:50] if synthetic_data else [],
        curriculum=DEFAULT_CURRICULUM,
    )
    
    # 5. Sample training batch
    print("\n[5/5] Sampling training batch...")
    batch = sampler.sample_batch(batch_size=8)
    print(f"✅ Sampled batch of {len(batch)} complexes")
    
    stats = sampler.get_statistics()
    print(f"\n📊 Current curriculum stage: {stats['stage']}")
    print(f"   Progress: {stats['progress']:.1%}")
    
    print("\n" + "=" * 80)
    print("✅ Complete pipeline demonstration finished!")
    print("=" * 80)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Pearl Data Pipeline Examples")
    print("=" * 80)
    print("\nThis script demonstrates Pearl's data loading and processing pipeline.")
    print("Based on the Pearl technical report (October 2025)")
    print("=" * 80)
    
    # Run examples
    try:
        # Individual examples
        example_1_load_pdb_data()
        example_2_generate_synthetic_data()
        example_3_preprocess_data()
        example_4_curriculum_sampling()
        
        # Full pipeline
        example_5_full_pipeline()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
    print("\n💡 Next steps:")
    print("   1. Set up your PDB data directory")
    print("   2. Generate a larger synthetic dataset")
    print("   3. Integrate with Pearl training pipeline")
    print("   4. Run curriculum training")
    print("\n📚 See pearl/data/ for implementation details")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

