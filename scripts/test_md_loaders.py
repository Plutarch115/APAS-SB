"""
Test script for MD trajectory loaders (mdCATH and ATLAS).

Tests:
1. mdCATH dataset loading
2. ATLAS dataset loading
3. Density map computation
4. Dataset statistics
"""

import sys
sys.path.append('.')

import torch
from pearl.data.mdcath_loader import mdCATHDataset
from pearl.data.atlas_loader import ATLASDataset


def test_mdcath():
    """Test mdCATH dataset loader"""
    print("\n" + "=" * 60)
    print("Testing mdCATH Dataset Loader")
    print("=" * 60)
    
    # Test with synthetic data
    dataset = mdCATHDataset(
        data_dir='./data/mdcath',
        temperature=320,
        split='train',
        max_frames=100,
        stride=10,
        include_forces=True,
        compute_density=True,
        use_synthetic=True
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  Number of trajectories: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\n  Dataset Statistics:")
    print(f"    Total frames: {stats['total_frames']:,}")
    print(f"    Total atoms: {stats['total_atoms']:,}")
    print(f"    Avg frames/trajectory: {stats['avg_frames_per_trajectory']:.1f}")
    print(f"    Avg atoms/trajectory: {stats['avg_atoms_per_trajectory']:.1f}")
    print(f"    Temperature: {stats['temperature']}K")
    
    # Test loading a sample
    print(f"\n  Testing sample loading...")
    sample = dataset[0]
    
    print(f"  ✓ Sample loaded successfully")
    print(f"    Keys: {list(sample.keys())}")
    print(f"    Coords shape: {sample['coords'].shape}")
    print(f"    Atom types shape: {sample['atom_types'].shape}")
    print(f"    Energies shape: {sample['energies'].shape}")
    
    if 'forces' in sample:
        print(f"    Forces shape: {sample['forces'].shape}")
    
    if 'density_map' in sample:
        print(f"    Density map shape: {sample['density_map'].shape}")
    
    print(f"    Trajectory ID: {sample['trajectory_id']}")
    print(f"    CATH ID: {sample['cath_id']}")
    print(f"    Temperature: {sample['temperature'].item()}K")
    
    # Test different temperatures
    print(f"\n  Testing different temperatures...")
    for temp in [320, 350, 380, 410, 450]:
        try:
            ds = mdCATHDataset(
                data_dir='./data/mdcath',
                temperature=temp,
                use_synthetic=True
            )
            print(f"    ✓ Temperature {temp}K: {len(ds)} trajectories")
        except Exception as e:
            print(f"    ✗ Temperature {temp}K failed: {e}")


def test_atlas():
    """Test ATLAS dataset loader"""
    print("\n" + "=" * 60)
    print("Testing ATLAS Dataset Loader")
    print("=" * 60)
    
    # Test with synthetic data
    dataset = ATLASDataset(
        data_dir='./data/atlas',
        split='train',
        max_frames=100,
        stride=10,
        compute_density=True,
        use_synthetic=True
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  Number of trajectories: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\n  Dataset Statistics:")
    print(f"    Total frames: {stats['total_frames']:,}")
    print(f"    Total atoms: {stats['total_atoms']:,}")
    print(f"    Total residues: {stats['total_residues']:,}")
    print(f"    Avg frames/trajectory: {stats['avg_frames_per_trajectory']:.1f}")
    print(f"    Avg atoms/trajectory: {stats['avg_atoms_per_trajectory']:.1f}")
    print(f"    Avg residues/trajectory: {stats['avg_residues_per_trajectory']:.1f}")
    
    # Test loading a sample
    print(f"\n  Testing sample loading...")
    sample = dataset[0]
    
    print(f"  ✓ Sample loaded successfully")
    print(f"    Keys: {list(sample.keys())}")
    print(f"    Coords shape: {sample['coords'].shape}")
    print(f"    Atom types shape: {sample['atom_types'].shape}")
    print(f"    RMSF shape: {sample['rmsf'].shape}")
    
    if 'density_map' in sample:
        print(f"    Density map shape: {sample['density_map'].shape}")
    
    print(f"    Protein ID: {sample['protein_id']}")
    print(f"    UniProt ID: {sample['uniprot_id']}")
    print(f"    Replicate: {sample['replicate']}")


def test_combined_usage():
    """Test using both datasets together"""
    print("\n" + "=" * 60)
    print("Testing Combined MD Dataset Usage")
    print("=" * 60)
    
    # Load both datasets
    mdcath = mdCATHDataset(
        data_dir='./data/mdcath',
        temperature=320,
        max_frames=50,
        use_synthetic=True
    )
    
    atlas = ATLASDataset(
        data_dir='./data/atlas',
        max_frames=50,
        use_synthetic=True
    )
    
    print(f"✓ Both datasets loaded")
    print(f"  mdCATH: {len(mdcath)} trajectories")
    print(f"  ATLAS: {len(atlas)} trajectories")
    print(f"  Total: {len(mdcath) + len(atlas)} trajectories")
    
    # Combine using ConcatDataset
    from torch.utils.data import ConcatDataset, DataLoader
    
    combined = ConcatDataset([mdcath, atlas])
    print(f"\n✓ Combined dataset created: {len(combined)} trajectories")
    
    # Test with DataLoader (without batching due to variable sizes)
    print(f"\n  Testing DataLoader...")
    loader = DataLoader(combined, batch_size=1, shuffle=True)
    
    batch = next(iter(loader))
    print(f"  ✓ DataLoader working")
    print(f"    Batch keys: {list(batch.keys())}")


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("MD TRAJECTORY LOADERS TEST SUITE")
    print("Testing mdCATH and ATLAS dataset loaders")
    print("🧪" * 40)
    
    test_mdcath()
    test_atlas()
    test_combined_usage()
    
    print("\n" + "=" * 60)
    print("✅ ALL MD LOADER TESTS COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download real mdCATH data: python scripts/download_datasets.py --datasets mdcath")
    print("2. Download real ATLAS data: python scripts/download_datasets.py --datasets atlas")
    print("3. Test with real data by setting use_synthetic=False")

