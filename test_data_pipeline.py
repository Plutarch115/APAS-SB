#!/usr/bin/env python3
"""
Quick test script for Pearl data pipeline.

Tests basic functionality without requiring external data files.
"""

import sys
import os
sys.path.insert(0, 'pearl')

import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from data import (
            PDBDataset,
            SyntheticDataGenerator,
            VirtualLigandLibrary,
            PhysicsBasedDocker,
            ProteinFeaturizer,
            LigandFeaturizer,
            ComplexPreprocessor,
            CroppingStrategy,
            CurriculumSampler,
            CurriculumStage,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_featurizers():
    """Test protein and ligand featurizers."""
    print("\nTesting featurizers...")
    try:
        from data.preprocessing import ProteinFeaturizer, LigandFeaturizer
        
        # Test protein featurizer
        protein_featurizer = ProteinFeaturizer(feature_dim=64)
        protein_coords = np.random.randn(100, 3).astype(np.float32)
        protein_atoms = ['C', 'N', 'O', 'S'] * 25
        protein_residues = np.repeat(np.arange(10), 10)
        protein_sequence = 'ACDEFGHIKL'
        
        protein_features = protein_featurizer.featurize(
            protein_coords, protein_atoms, protein_residues, protein_sequence
        )
        
        assert len(protein_features.coords) == 100
        assert len(protein_features.atom_types) == 100
        print("✅ Protein featurizer works")
        
        # Test ligand featurizer
        ligand_featurizer = LigandFeaturizer(feature_dim=64)
        ligand_coords = np.random.randn(30, 3).astype(np.float32)
        ligand_atoms = ['C', 'N', 'O'] * 10
        
        ligand_features = ligand_featurizer.featurize(
            ligand_coords, ligand_atoms
        )
        
        assert len(ligand_features.coords) == 30
        assert len(ligand_features.atom_types) == 30
        print("✅ Ligand featurizer works")
        
        return True
    except Exception as e:
        print(f"❌ Featurizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cropping():
    """Test cropping strategy."""
    print("\nTesting cropping strategy...")
    try:
        from data.preprocessing import CroppingStrategy
        
        cropper = CroppingStrategy(max_atoms=50, strategy='pocket_centered')
        
        protein_coords = np.random.randn(200, 3).astype(np.float32)
        ligand_coords = np.random.randn(20, 3).astype(np.float32)
        protein_atoms = np.arange(200)
        protein_residues = np.repeat(np.arange(20), 10)
        
        cropped_coords, cropped_atoms, cropped_residues = cropper.crop_protein(
            protein_coords, ligand_coords, protein_atoms, protein_residues
        )
        
        assert len(cropped_coords) == 50
        assert len(cropped_atoms) == 50
        print("✅ Cropping strategy works")
        
        return True
    except Exception as e:
        print(f"❌ Cropping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_generation():
    """Test synthetic data generation (without RDKit)."""
    print("\nTesting synthetic data generation...")
    try:
        from data.synthetic_generator import PhysicsBasedDocker
        
        docker = PhysicsBasedDocker(method='random_placement')
        
        protein_coords = np.random.randn(500, 3).astype(np.float32) * 20
        ligand_coords = np.random.randn(30, 3).astype(np.float32) * 5
        pocket_center = np.array([0.0, 0.0, 0.0])
        
        docked_coords, score = docker.dock_ligand(
            protein_coords, ligand_coords, pocket_center
        )
        
        assert docked_coords.shape == ligand_coords.shape
        assert isinstance(score, float)
        print("✅ Docking works")
        
        return True
    except Exception as e:
        print(f"❌ Synthetic generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum():
    """Test curriculum sampler."""
    print("\nTesting curriculum sampler...")
    try:
        from data.curriculum_sampler import CurriculumSampler, DEFAULT_CURRICULUM
        
        # Create dummy datasets
        pdb_data = [{'id': f'pdb_{i}', 'type': 'pdb'} for i in range(10)]
        synthetic_data = [{'id': f'syn_{i}', 'type': 'synthetic'} for i in range(50)]
        
        sampler = CurriculumSampler(
            pdb_dataset=pdb_data,
            synthetic_dataset=synthetic_data,
            curriculum=DEFAULT_CURRICULUM,
        )
        
        # Test sampling
        batch = sampler.sample_batch(batch_size=4)
        assert len(batch) == 4
        
        # Test statistics
        stats = sampler.get_statistics()
        assert 'stage' in stats
        assert 'progress' in stats
        
        print("✅ Curriculum sampler works")
        return True
    except Exception as e:
        print(f"❌ Curriculum test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test complete preprocessing pipeline."""
    print("\nTesting preprocessing pipeline...")
    try:
        from data.preprocessing import (
            ProteinFeaturizer,
            LigandFeaturizer,
            ComplexPreprocessor,
            CroppingStrategy,
        )
        
        # Create preprocessor
        preprocessor = ComplexPreprocessor(
            protein_featurizer=ProteinFeaturizer(feature_dim=64),
            ligand_featurizer=LigandFeaturizer(feature_dim=64),
            cropping_strategy=CroppingStrategy(max_atoms=100),
            normalize_coords=True,
            augment=False,  # Disable for deterministic test
        )
        
        # Create test complex
        complex_data = {
            'pdb_id': 'TEST',
            'protein_coords': np.random.randn(200, 3).astype(np.float32),
            'protein_atoms': ['C', 'N', 'O'] * 67,
            'protein_residues': np.repeat(np.arange(20), 10),
            'protein_sequence': 'ACDEFGHIKLMNPQRSTVWY',
            'ligand_coords': np.random.randn(30, 3).astype(np.float32),
            'ligand_atoms': ['C', 'N', 'O'] * 10,
            'ligand_bonds': [],
        }
        
        # Preprocess
        processed = preprocessor.preprocess(complex_data)
        
        assert 'protein' in processed
        assert 'ligand' in processed
        assert len(processed['protein'].coords) <= 100  # Cropped
        
        print("✅ Preprocessing pipeline works")
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Pearl Data Pipeline Tests")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Featurizers", test_featurizers),
        ("Cropping", test_cropping),
        ("Synthetic Generation", test_synthetic_generation),
        ("Curriculum", test_curriculum),
        ("Preprocessing", test_preprocessing),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

