"""
Test script for hybrid dataset implementation.

Verifies that all dataset loaders work correctly and can be combined.
"""

import sys
sys.path.append('.')

import torch
from pearl.data.multitask_datasets import (
    PDBBindDataset,
    SKEMPI2Dataset,
    BRENDADataset,
    ProteinGymDataset,
    ChEMBLDataset,
    BindingDBDataset,
    create_multitask_dataset
)

def test_individual_datasets():
    """Test each dataset individually"""
    print("=" * 80)
    print("Testing Individual Datasets")
    print("=" * 80)
    
    datasets = {
        'PDBbind': PDBBindDataset(data_dir='data/pdbind', use_synthetic=True),
        'SKEMPI2': SKEMPI2Dataset(data_dir='data/skempi2', use_synthetic=True),
        'BRENDA': BRENDADataset(data_dir='data/brenda', use_synthetic=True),
        'ProteinGym': ProteinGymDataset(data_dir='data/proteingym', use_synthetic=True),
        'ChEMBL': ChEMBLDataset(data_dir='data/chembl', use_synthetic=True),
        'BindingDB': BindingDBDataset(data_dir='data/bindingdb', use_synthetic=True),
    }
    
    for name, dataset in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Size: {len(dataset)}")
        
        # Get first item
        item = dataset[0]
        print(f"  Keys: {list(item.keys())}")
        print(f"  Task: {item['task']}")
        print(f"  Data source: {item['data_source']}")
        print(f"  Weight: {item['weight'].item():.1f}")
        
        if 'protein_features' in item:
            print(f"  Protein features shape: {item['protein_features'].shape}")
        if 'ligand_features' in item:
            print(f"  Ligand features shape: {item['ligand_features'].shape}")
        if 'target' in item:
            print(f"  Target shape: {item['target'].shape}")
            print(f"  Target value: {item['target'].item():.3f}")
        
        print(f"  ✓ {name} dataset works correctly")
    
    print("\n" + "=" * 80)
    print("✓ All individual datasets passed!")
    print("=" * 80)


def test_combined_dataset():
    """Test combined hybrid dataset"""
    print("\n" + "=" * 80)
    print("Testing Combined Hybrid Dataset")
    print("=" * 80)
    
    # Create combined dataset with all sources
    data_dirs = {
        'pdbind': 'data/pdbind',
        'skempi2': 'data/skempi2',
        'brenda': 'data/brenda',
        'proteingym': 'data/proteingym',
        'chembl': 'data/chembl',
        'bindingdb': 'data/bindingdb',
    }
    
    combined_dataset = create_multitask_dataset(
        data_dirs=data_dirs,
        split='train',
        use_synthetic=True
    )
    
    print(f"\nCombined Dataset:")
    print(f"  Total size: {len(combined_dataset)}")
    
    # Count examples by task
    task_counts = {}
    source_counts = {}
    
    print(f"\n  Sampling 100 examples to analyze distribution...")
    for i in range(min(100, len(combined_dataset))):
        item = combined_dataset[i]
        task = item['task']
        source = item['data_source']
        
        task_counts[task] = task_counts.get(task, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n  Task distribution (first 100 examples):")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")
    
    print(f"\n  Data source distribution (first 100 examples):")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count}")
    
    print("\n" + "=" * 80)
    print("✓ Combined dataset works correctly!")
    print("=" * 80)


def test_dataloader():
    """Test PyTorch DataLoader with hybrid dataset"""
    print("\n" + "=" * 80)
    print("Testing PyTorch DataLoader")
    print("=" * 80)
    
    from torch.utils.data import DataLoader
    
    # Create combined dataset
    data_dirs = {
        'chembl': 'data/chembl',
        'bindingdb': 'data/bindingdb',
        'skempi2': 'data/skempi2',
    }
    
    dataset = create_multitask_dataset(
        data_dirs=data_dirs,
        split='train',
        use_synthetic=True
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"\nDataLoader:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: 8")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Test first batch
    print(f"\n  Testing first batch...")
    batch = next(iter(dataloader))
    
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch sizes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"    {key}: list of {len(value)} items")
    
    print("\n" + "=" * 80)
    print("✓ DataLoader works correctly!")
    print("=" * 80)


def print_dataset_statistics():
    """Print comprehensive dataset statistics"""
    print("\n" + "=" * 80)
    print("Hybrid Dataset Statistics")
    print("=" * 80)
    
    # Synthetic data sizes (for testing)
    synthetic_sizes = {
        'PDBbind': 1000,
        'SKEMPI2': 800,
        'BRENDA': 1200,
        'ProteinGym': 2000,
        'ChEMBL': 10000,
        'BindingDB': 8000,
    }
    
    # Real data sizes (from Boltz-2 paper and other sources)
    real_sizes = {
        'PDBbind': 20000,
        'SKEMPI2': 8000,
        'BRENDA': 100000,
        'ProteinGym': 2500000,
        'ChEMBL': 600000,
        'BindingDB': 600000,
        'PubChem HTS': 2000000,
        'PubChem Small': 60000,
        'CeMM Fragments': 140000,
        'MIDAS': 22000,
        'Synthetic Decoys': 1200000,
    }
    
    print("\n📊 SYNTHETIC DATA (for testing):")
    print("-" * 80)
    total_synthetic = 0
    for name, size in synthetic_sizes.items():
        print(f"  {name:20s}: {size:>10,} examples")
        total_synthetic += size
    print(f"  {'TOTAL':20s}: {total_synthetic:>10,} examples")
    
    print("\n📊 REAL DATA (when fully implemented):")
    print("-" * 80)
    
    print("\n  ORIGINAL MULTI-TASK DATASETS:")
    original_total = 0
    for name in ['PDBbind', 'SKEMPI2', 'BRENDA', 'ProteinGym']:
        size = real_sizes[name]
        print(f"    {name:20s}: {size:>10,} examples")
        original_total += size
    print(f"    {'Subtotal':20s}: {original_total:>10,} examples")
    
    print("\n  BOLTZ-2 DATASETS:")
    boltz2_total = 0
    for name in ['ChEMBL', 'BindingDB', 'PubChem HTS', 'PubChem Small', 
                 'CeMM Fragments', 'MIDAS', 'Synthetic Decoys']:
        size = real_sizes[name]
        print(f"    {name:20s}: {size:>10,} examples")
        boltz2_total += size
    print(f"    {'Subtotal':20s}: {boltz2_total:>10,} examples")
    
    print(f"\n  {'GRAND TOTAL':20s}: {original_total + boltz2_total:>10,} examples")
    
    print("\n💾 STORAGE ESTIMATES:")
    print("-" * 80)
    print(f"  Raw data:              ~156 GB")
    print(f"  Processed features:     ~94 GB")
    print(f"  Total storage:         ~200 GB")
    
    print("\n💰 COST ESTIMATES:")
    print("-" * 80)
    print(f"  Data acquisition:          $0 (all public)")
    print(f"  Data processing:      ~$8,000 (one-time)")
    print(f"  Training (from scratch): ~$87M")
    print(f"  Training (pretrained):   ~$20M")
    print(f"  Storage:              $10/month")
    
    print("\n⏱️  TIME ESTIMATES:")
    print("-" * 80)
    print(f"  Data download:        ~10 days")
    print(f"  Data processing:      ~30 days")
    print(f"  Training (full):      ~75 days (64× A100)")
    print(f"  Training (pretrained): ~24 days (32× A100)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("HYBRID DATASET IMPLEMENTATION TEST")
    print("=" * 80)
    
    try:
        # Test individual datasets
        test_individual_datasets()
        
        # Test combined dataset
        test_combined_dataset()
        
        # Test DataLoader
        test_dataloader()
        
        # Print statistics
        print_dataset_statistics()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe hybrid dataset implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Download real datasets (see HYBRID_DATASET_SIZE_ESTIMATES.md)")
        print("  2. Implement remaining Boltz-2 datasets (PubChem, CeMM, MIDAS)")
        print("  3. Implement Boltz-2 loss functions (Huber, pairwise, focal)")
        print("  4. Train on high-quality subset first")
        print("  5. Scale up to full hybrid dataset")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

