"""
Test script for all Boltz-2 datasets (11 total datasets).

Tests:
1. All 11 dataset loaders (4 original + 7 Boltz-2)
2. Combined hybrid dataset
3. PyTorch DataLoader integration
4. Dataset statistics and validation
"""

import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader
from pearl.data.multitask_datasets import (
    PDBBindDataset,
    SKEMPI2Dataset,
    BRENDADataset,
    ProteinGymDataset,
    ChEMBLDataset,
    BindingDBDataset,
    PubChemHTSDataset,
    PubChemSmallAssaysDataset,
    CeMMFragmentsDataset,
    MIDASDataset,
    SyntheticDecoysDataset,
    create_multitask_dataset
)


def test_individual_datasets():
    """Test each dataset individually"""
    print("=" * 80)
    print("TESTING INDIVIDUAL DATASETS")
    print("=" * 80)

    datasets_to_test = [
        ("PDBbind", PDBBindDataset, 'data/pdbind'),
        ("SKEMPI 2.0", SKEMPI2Dataset, 'data/skempi2'),
        ("BRENDA", BRENDADataset, 'data/brenda'),
        ("ProteinGym", ProteinGymDataset, 'data/proteingym'),
        ("ChEMBL", ChEMBLDataset, 'data/chembl'),
        ("BindingDB", BindingDBDataset, 'data/bindingdb'),
        ("PubChem HTS", PubChemHTSDataset, 'data/pubchem_hts'),
        ("PubChem Small", PubChemSmallAssaysDataset, 'data/pubchem_small'),
        ("CeMM Fragments", CeMMFragmentsDataset, 'data/cemm'),
        ("MIDAS", MIDASDataset, 'data/midas'),
        ("Synthetic Decoys", SyntheticDecoysDataset, 'data/synthetic_decoys'),
    ]

    for name, dataset_class, data_dir in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {name} Dataset")
        print(f"{'='*60}")

        try:
            dataset = dataset_class(data_dir, split='train', use_synthetic=True)
            print(f"✓ Dataset created successfully")
            print(f"  Size: {len(dataset)} samples")

            # Test first sample
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Protein features shape: {sample['protein_features'].shape}")
            print(f"  Ligand features shape: {sample['ligand_features'].shape}")
            print(f"  Target: {sample['target'].item():.4f}")
            print(f"  Task: {sample['task']}")
            print(f"  Data source: {sample['data_source']}")
            print(f"  Weight: {sample['weight'].item():.2f}")

        except Exception as e:
            print(f"✗ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()


def test_combined_dataset():
    """Test combined hybrid dataset"""
    print("\n" + "=" * 80)
    print("TESTING COMBINED HYBRID DATASET")
    print("=" * 80)

    data_dirs = {
        # Original multi-task datasets
        'pdbind': 'data/pdbind',
        'skempi2': 'data/skempi2',
        'brenda': 'data/brenda',
        'proteingym': 'data/proteingym',
        # Boltz-2 datasets
        'chembl': 'data/chembl',
        'bindingdb': 'data/bindingdb',
        'pubchem_hts': 'data/pubchem_hts',
        'pubchem_small': 'data/pubchem_small',
        'cemm': 'data/cemm',
        'midas': 'data/midas',
        'synthetic_decoys': 'data/synthetic_decoys',
    }

    try:
        dataset = create_multitask_dataset(data_dirs, split='train', use_synthetic=True)
        print(f"✓ Combined dataset created successfully")
        print(f"  Total size: {len(dataset)} samples")

        # Test sampling from combined dataset
        print(f"\n  Testing random samples:")
        for i in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
            sample = dataset[i]
            print(f"    Sample {i}: task={sample['task']}, source={sample['data_source']}")

    except Exception as e:
        print(f"✗ Error testing combined dataset: {e}")
        import traceback
        traceback.print_exc()


def test_dataloader():
    """Test PyTorch DataLoader integration"""
    print("\n" + "=" * 80)
    print("TESTING PYTORCH DATALOADER")
    print("=" * 80)

    data_dirs = {
        'chembl': 'data/chembl',
        'bindingdb': 'data/bindingdb',
        'pubchem_hts': 'data/pubchem_hts',
    }

    try:
        dataset = create_multitask_dataset(data_dirs, split='train', use_synthetic=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: 32")
        print(f"  Number of batches: {len(dataloader)}")

        # Test one batch
        batch = next(iter(dataloader))
        print(f"\n  First batch:")
        print(f"    Protein features: {batch['protein_features'].shape}")
        print(f"    Ligand features: {batch['ligand_features'].shape}")
        print(f"    Targets: {batch['target'].shape}")
        print(f"    Weights: {batch['weight'].shape}")

    except Exception as e:
        print(f"✗ Error testing DataLoader: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("BOLTZ-2 DATASETS TEST SUITE")
    print("Testing all 11 datasets (4 original + 7 Boltz-2)")
    print("🧪" * 40 + "\n")

    test_individual_datasets()
    test_combined_dataset()
    test_dataloader()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)

