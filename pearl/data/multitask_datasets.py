"""
Multi-task datasets for PEARL training - HYBRID IMPLEMENTATION.

Combines our original multi-task datasets with Boltz-2 binding affinity datasets:

ORIGINAL DATASETS (Multi-task capabilities):
- PDBbind: Protein-ligand binding affinity
- SKEMPI 2.0: Protein-protein interaction ΔΔG
- BRENDA: Enzyme kinetic parameters (kcat, Km)
- ProteinGym: Deep mutational scanning fitness scores

BOLTZ-2 DATASETS (Binding affinity focus):
- ChEMBL (v34): Bioactivity data (1.2M binders with BindingDB)
- BindingDB: Protein-ligand binding measurements
- PubChem HTS: High-throughput screening (200K binders, 1.8M decoys)
- PubChem Small Assays: Higher-quality screens (10K binders, 50K decoys)
- CeMM Fragments: Fragment screening (25K binders, 115K decoys)
- MIDAS: Protein-metabolite interactions (2K binders, 20K decoys)
- Synthetic Decoys: Generated negative examples (1.2M decoys)

Total Hybrid Dataset: ~1.4M binders + ~3.2M decoys + original multi-task data
"""

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class PDBBindDataset(Dataset):
    """
    PDBbind database: Protein-ligand binding affinity.

    Data format:
    - PDB structures for protein-ligand complexes
    - Binding affinity values (Kd, Ki, IC50)
    - ~20,000 complexes
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic PDBbind-like data for testing"""
        np.random.seed(42)
        data = []

        for i in range(num_samples):
            # Simulate binding affinity (pKd: 4-12)
            true_affinity = np.random.uniform(4.0, 12.0)

            data.append({
                'protein_id': f'protein_{i}',
                'ligand_id': f'ligand_{i}',
                'binding_affinity': true_affinity,
                'affinity_type': np.random.choice(['Kd', 'Ki', 'IC50']),
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(20, 64).astype(np.float32),
                'weight': 10.0,
                'data_source': 'pdbind',
                'task': 'binding_affinity'
            })

        return data

    def _load_real_data(self):
        """Load real PDBbind data"""
        # TODO: Implement real data loading
        # Expected format: CSV or JSON with columns:
        # - pdb_id, protein_pdb_path, ligand_mol2_path, affinity_value, affinity_type
        raise NotImplementedError("Real PDBbind data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['binding_affinity'], dtype=torch.float32),
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class SKEMPI2Dataset(Dataset):
    """
    SKEMPI 2.0 database: Protein-protein interaction ΔΔG.

    Data format:
    - Wild-type and mutant protein structures
    - ΔΔG upon mutation (kcal/mol)
    - ~8,000 mutations
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 800):
        """Generate synthetic SKEMPI-like data for testing"""
        np.random.seed(43)
        data = []

        for i in range(num_samples):
            # Simulate ΔΔG (-5 to +5 kcal/mol)
            true_ddg = np.random.uniform(-5.0, 5.0)

            # Wild-type features
            wt_features = np.random.randn(100, 64).astype(np.float32)

            # Mutant features (perturb at mutation site)
            mut_features = wt_features.copy()
            mut_position = np.random.randint(0, 100)
            mut_features[mut_position] += np.random.randn(64) * 0.5

            data.append({
                'protein_id': f'complex_{i}',
                'mutation': f'A:{mut_position}R',
                'ddg_exp': true_ddg,
                'ddg_error': np.random.uniform(0.1, 0.5),
                'wt_protein_features': wt_features,
                'mut_protein_features': mut_features,
                'weight': 10.0,
                'data_source': 'skempi2',
                'task': 'ddg_ppi'
            })

        return data

    def _load_real_data(self):
        """Load real SKEMPI 2.0 data"""
        # TODO: Implement real data loading
        raise NotImplementedError("Real SKEMPI 2.0 data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'wt_protein_features': torch.from_numpy(item['wt_protein_features']),
            'mut_protein_features': torch.from_numpy(item['mut_protein_features']),
            'target': torch.tensor(item['ddg_exp'], dtype=torch.float32),
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class BRENDADataset(Dataset):
    """
    BRENDA database: Enzyme kinetic parameters.

    Data format:
    - Enzyme structures
    - kcat (turnover number, s^-1)
    - Km (Michaelis constant, M)
    - ~50,000 enzymes
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 500):
        """Generate synthetic BRENDA-like data for testing"""
        np.random.seed(44)
        data = []

        for i in range(num_samples):
            # Simulate kcat (log scale: 10^-2 to 10^6 s^-1)
            log_kcat = np.random.uniform(-2.0, 6.0)
            true_kcat = 10 ** log_kcat

            data.append({
                'enzyme_id': f'enzyme_{i}',
                'substrate_id': f'substrate_{i}',
                'kcat': true_kcat,
                'log_kcat': log_kcat,
                'km': 10 ** np.random.uniform(-6.0, -3.0),  # μM to mM range
                'enzyme_features': np.random.randn(100, 64).astype(np.float32),
                'substrate_features': np.random.randn(20, 64).astype(np.float32),
                'weight': 8.0,
                'data_source': 'brenda',
                'task': 'kcat'
            })

        return data

    def _load_real_data(self):
        """Load real BRENDA data"""
        # TODO: Implement real data loading
        raise NotImplementedError("Real BRENDA data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'enzyme_features': torch.from_numpy(item['enzyme_features']),
            'substrate_features': torch.from_numpy(item['substrate_features']),
            'target': torch.tensor(item['log_kcat'], dtype=torch.float32),  # Predict log(kcat)
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class ProteinGymDataset(Dataset):
    """
    ProteinGym database: Deep mutational scanning fitness scores.

    Data format:
    - Wild-type and mutant protein sequences
    - Fitness scores (normalized)
    - ~2.5M variants across 250+ proteins
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic ProteinGym-like data for testing"""
        np.random.seed(45)
        data = []

        for i in range(num_samples):
            # Simulate fitness score (-3 to +3, normalized)
            true_fitness = np.random.normal(0.0, 1.0)

            # Wild-type features
            wt_features = np.random.randn(100, 64).astype(np.float32)

            # Mutant features (can have multiple mutations)
            mut_features = wt_features.copy()
            num_mutations = np.random.randint(1, 4)
            for _ in range(num_mutations):
                mut_position = np.random.randint(0, 100)
                mut_features[mut_position] += np.random.randn(64) * 0.3

            data.append({
                'protein_id': f'protein_{i}',
                'mutations': f'{num_mutations}_mutations',
                'fitness_score': true_fitness,
                'wt_protein_features': wt_features,
                'mut_protein_features': mut_features,
                'weight': 9.0,
                'data_source': 'proteingym',
                'task': 'fitness'
            })

        return data

    def _load_real_data(self):
        """Load real ProteinGym data"""
        # TODO: Implement real data loading
        raise NotImplementedError("Real ProteinGym data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'wt_protein_features': torch.from_numpy(item['wt_protein_features']),
            'mut_protein_features': torch.from_numpy(item['mut_protein_features']),
            'target': torch.tensor(item['fitness_score'], dtype=torch.float32),
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


# ============================================================================
# BOLTZ-2 DATASETS - Binding Affinity Focus
# ============================================================================

class ChEMBLDataset(Dataset):
    """
    ChEMBL v34: Manually curated bioactive molecules.

    Boltz-2 usage:
    - 1.2M binders (combined with BindingDB)
    - Continuous affinity values (Ki, Kd, IC50, AC50, EC50, XC50)
    - Standardized to log10(µM)
    - Sampling weight: 0.25
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 10000):
        """Generate synthetic ChEMBL-like data for testing"""
        np.random.seed(43)
        data = []

        for i in range(num_samples):
            # Simulate protein-ligand binding affinity
            affinity_types = ['Ki', 'Kd', 'IC50', 'AC50', 'EC50', 'XC50']
            affinity_type = np.random.choice(affinity_types)

            # log10(µM) typically ranges from -3 to 2 (nM to mM)
            affinity_value = np.random.uniform(-3, 2)

            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(30, 64).astype(np.float32),
                'affinity_value': affinity_value,
                'affinity_type': affinity_type,
                'weight': 10.0,  # High-quality curated data
                'data_source': 'chembl',
                'task': 'binding_affinity'
            })

        return data

    def _load_real_data(self):
        """Load real ChEMBL data"""
        # TODO: Implement ChEMBL data loading
        # - Parse ChEMBL database
        # - Filter: single-protein targets only
        # - Filter: biochemical or functional assays only
        # - Apply PAINS filters
        # - Filter by iptm >0.75
        # - Standardize to log10(µM)
        raise NotImplementedError("Real ChEMBL data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['affinity_value'], dtype=torch.float32),
            'affinity_type': item['affinity_type'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class BindingDBDataset(Dataset):
    """
    BindingDB: Protein-ligand binding measurements.

    Boltz-2 usage:
    - Combined with ChEMBL (1.2M total binders)
    - Continuous affinity values (Ki, Kd, IC50, AC50, EC50, XC50)
    - Standardized to log10(µM)
    - Sampling weight: 0.25 (combined with ChEMBL)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 8000):
        """Generate synthetic BindingDB-like data for testing"""
        np.random.seed(44)
        data = []

        for i in range(num_samples):
            affinity_types = ['Ki', 'Kd', 'IC50', 'AC50', 'EC50', 'XC50']
            affinity_type = np.random.choice(affinity_types)
            affinity_value = np.random.uniform(-3, 2)

            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(30, 64).astype(np.float32),
                'affinity_value': affinity_value,
                'affinity_type': affinity_type,
                'weight': 10.0,
                'data_source': 'bindingdb',
                'task': 'binding_affinity'
            })

        return data

    def _load_real_data(self):
        """Load real BindingDB data"""
        # TODO: Implement BindingDB data loading
        raise NotImplementedError("Real BindingDB data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['affinity_value'], dtype=torch.float32),
            'affinity_type': item['affinity_type'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class PubChemHTSDataset(Dataset):
    """
    PubChem High-Throughput Screening: Binary classification for hit discovery.

    Boltz-2 usage:
    - 200K binders + 1.8M decoys
    - Binary classification (hit/non-hit)
    - Sampling weight: 0.25
    - Focus: High-throughput screening campaigns
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 5000):
        """Generate synthetic PubChem HTS-like data for testing"""
        np.random.seed(45)
        data = []

        # 10% binders, 90% decoys (realistic HTS ratio)
        num_binders = int(num_samples * 0.1)
        num_decoys = num_samples - num_binders

        for i in range(num_binders):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(25, 64).astype(np.float32),
                'is_binder': 1.0,
                'assay_id': f'AID_{np.random.randint(1000, 9999)}',
                'weight': 5.0,  # Lower weight for HTS data
                'data_source': 'pubchem_hts',
                'task': 'binary_classification'
            })

        for i in range(num_decoys):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(25, 64).astype(np.float32),
                'is_binder': 0.0,
                'assay_id': f'AID_{np.random.randint(1000, 9999)}',
                'weight': 5.0,
                'data_source': 'pubchem_hts',
                'task': 'binary_classification'
            })

        return data

    def _load_real_data(self):
        """Load real PubChem HTS data"""
        # TODO: Implement PubChem HTS data loading
        # - Download from PubChem BioAssay database
        # - Filter for protein targets
        # - Extract active/inactive compounds
        raise NotImplementedError("Real PubChem HTS data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['is_binder'], dtype=torch.float32),
            'assay_id': item['assay_id'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class PubChemSmallAssaysDataset(Dataset):
    """
    PubChem Small Assays: Higher-quality screening data.

    Boltz-2 usage:
    - 10K binders + 50K decoys
    - Both binary and continuous affinity values
    - Sampling weight: 0.10
    - Focus: Smaller, higher-quality assays
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 2000):
        """Generate synthetic PubChem Small Assays data for testing"""
        np.random.seed(46)
        data = []

        # 17% binders, 83% decoys (better ratio than HTS)
        num_binders = int(num_samples * 0.17)
        num_decoys = num_samples - num_binders

        for i in range(num_binders):
            # Some have continuous values, some binary
            has_affinity = np.random.rand() > 0.5

            item = {
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(25, 64).astype(np.float32),
                'is_binder': 1.0,
                'assay_id': f'AID_{np.random.randint(100, 999)}',
                'weight': 7.0,  # Higher quality than HTS
                'data_source': 'pubchem_small',
                'task': 'binding_affinity' if has_affinity else 'binary_classification'
            }

            if has_affinity:
                item['affinity_value'] = np.random.uniform(-2, 2)
                item['affinity_type'] = np.random.choice(['IC50', 'EC50', 'Ki'])

            data.append(item)

        for i in range(num_decoys):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(25, 64).astype(np.float32),
                'is_binder': 0.0,
                'assay_id': f'AID_{np.random.randint(100, 999)}',
                'weight': 7.0,
                'data_source': 'pubchem_small',
                'task': 'binary_classification'
            })

        return data

    def _load_real_data(self):
        """Load real PubChem Small Assays data"""
        # TODO: Implement PubChem Small Assays data loading
        raise NotImplementedError("Real PubChem Small Assays data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        result = {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item.get('affinity_value', item['is_binder']), dtype=torch.float32),
            'assay_id': item['assay_id'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }

        if 'affinity_type' in item:
            result['affinity_type'] = item['affinity_type']

        return result


class CeMMFragmentsDataset(Dataset):
    """
    CeMM Fragments: Fragment screening data.

    Boltz-2 usage:
    - 25K binders + 115K decoys
    - Binary classification
    - Sampling weight: 0.10
    - Focus: Fragment-based drug discovery
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 3000):
        """Generate synthetic CeMM fragment data for testing"""
        np.random.seed(47)
        data = []

        # 18% binders, 82% decoys (fragment screening ratio)
        num_binders = int(num_samples * 0.18)
        num_decoys = num_samples - num_binders

        for i in range(num_binders):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(15, 64).astype(np.float32),  # Smaller fragments
                'is_binder': 1.0,
                'fragment_id': f'FRAG_{i:05d}',
                'weight': 7.0,
                'data_source': 'cemm_fragments',
                'task': 'binary_classification'
            })

        for i in range(num_decoys):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(15, 64).astype(np.float32),
                'is_binder': 0.0,
                'fragment_id': f'FRAG_{i+num_binders:05d}',
                'weight': 7.0,
                'data_source': 'cemm_fragments',
                'task': 'binary_classification'
            })

        return data

    def _load_real_data(self):
        """Load real CeMM fragment data"""
        # TODO: Implement CeMM fragment data loading
        raise NotImplementedError("Real CeMM fragment data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['is_binder'], dtype=torch.float32),
            'fragment_id': item['fragment_id'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class MIDASDataset(Dataset):
    """
    MIDAS: Protein-metabolite interactions.

    Boltz-2 usage:
    - 2K binders + 20K decoys
    - Binary classification
    - Sampling weight: 0.05
    - Focus: Metabolite binding
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic MIDAS data for testing"""
        np.random.seed(48)
        data = []

        # 9% binders, 91% decoys (metabolite screening ratio)
        num_binders = int(num_samples * 0.09)
        num_decoys = num_samples - num_binders

        for i in range(num_binders):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(20, 64).astype(np.float32),  # Metabolites
                'is_binder': 1.0,
                'metabolite_id': f'MET_{i:04d}',
                'weight': 8.0,  # Higher quality metabolite data
                'data_source': 'midas',
                'task': 'binary_classification'
            })

        for i in range(num_decoys):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(20, 64).astype(np.float32),
                'is_binder': 0.0,
                'metabolite_id': f'MET_{i+num_binders:04d}',
                'weight': 8.0,
                'data_source': 'midas',
                'task': 'binary_classification'
            })

        return data

    def _load_real_data(self):
        """Load real MIDAS data"""
        # TODO: Implement MIDAS data loading
        raise NotImplementedError("Real MIDAS data loading not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['is_binder'], dtype=torch.float32),
            'metabolite_id': item['metabolite_id'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


class SyntheticDecoysDataset(Dataset):
    """
    Synthetic Decoys: Generated negative examples for data augmentation.

    Boltz-2 usage:
    - 1.2M synthetic decoys
    - Binary classification (all negatives)
    - Sampling weight: 0.25
    - Focus: Data augmentation, hard negatives
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True,
        num_decoys: int = 10000  # Configurable for testing
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic
        self.num_decoys = num_decoys

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def _generate_synthetic_data(self):
        """Generate synthetic decoys for testing"""
        np.random.seed(49)
        data = []

        for i in range(self.num_decoys):
            data.append({
                'protein_features': np.random.randn(100, 64).astype(np.float32),
                'ligand_features': np.random.randn(25, 64).astype(np.float32),
                'is_binder': 0.0,  # All decoys
                'decoy_id': f'DECOY_{i:06d}',
                'weight': 3.0,  # Lower weight for synthetic data
                'data_source': 'synthetic_decoys',
                'task': 'binary_classification'
            })

        return data

    def _load_real_data(self):
        """Load real synthetic decoys"""
        # TODO: Implement synthetic decoy generation/loading
        # - Generate using SMILES randomization
        # - Generate using docking-based methods
        # - Generate using GAN-based approaches
        raise NotImplementedError("Real synthetic decoy generation not yet implemented")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        return {
            'protein_features': torch.from_numpy(item['protein_features']),
            'ligand_features': torch.from_numpy(item['ligand_features']),
            'target': torch.tensor(item['is_binder'], dtype=torch.float32),
            'decoy_id': item['decoy_id'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source']
        }


def create_multitask_dataset(
    data_dirs: Dict[str, str],
    split: str = 'train',
    use_synthetic: bool = True
) -> ConcatDataset:
    """
    Create a combined HYBRID multi-task dataset.

    Combines original multi-task datasets with Boltz-2 binding affinity datasets.

    Args:
        data_dirs: Dictionary mapping dataset names to data directories
                   Supported keys:
                   - Original: 'pdbind', 'skempi2', 'brenda', 'proteingym'
                   - Boltz-2: 'chembl', 'bindingdb', 'pubchem_hts',
                              'pubchem_small', 'cemm', 'midas', 'synthetic_decoys'
        split: 'train', 'val', or 'test'
        use_synthetic: Whether to use synthetic data for testing

    Returns:
        ConcatDataset combining all requested datasets
    """
    datasets = []

    # ========== ORIGINAL MULTI-TASK DATASETS ==========
    if 'pdbind' in data_dirs:
        datasets.append(PDBBindDataset(data_dirs['pdbind'], split, use_synthetic=use_synthetic))

    if 'skempi2' in data_dirs:
        datasets.append(SKEMPI2Dataset(data_dirs['skempi2'], split, use_synthetic=use_synthetic))

    if 'brenda' in data_dirs:
        datasets.append(BRENDADataset(data_dirs['brenda'], split, use_synthetic=use_synthetic))

    if 'proteingym' in data_dirs:
        datasets.append(ProteinGymDataset(data_dirs['proteingym'], split, use_synthetic=use_synthetic))

    # ========== BOLTZ-2 DATASETS ==========
    if 'chembl' in data_dirs:
        datasets.append(ChEMBLDataset(data_dirs['chembl'], split, use_synthetic=use_synthetic))

    if 'bindingdb' in data_dirs:
        datasets.append(BindingDBDataset(data_dirs['bindingdb'], split, use_synthetic=use_synthetic))

    if 'pubchem_hts' in data_dirs:
        datasets.append(PubChemHTSDataset(data_dirs['pubchem_hts'], split, use_synthetic=use_synthetic))

    if 'pubchem_small' in data_dirs:
        datasets.append(PubChemSmallAssaysDataset(data_dirs['pubchem_small'], split, use_synthetic=use_synthetic))

    if 'cemm' in data_dirs:
        datasets.append(CeMMFragmentsDataset(data_dirs['cemm'], split, use_synthetic=use_synthetic))

    if 'midas' in data_dirs:
        datasets.append(MIDASDataset(data_dirs['midas'], split, use_synthetic=use_synthetic))

    if 'synthetic_decoys' in data_dirs:
        datasets.append(SyntheticDecoysDataset(data_dirs['synthetic_decoys'], split, use_synthetic=use_synthetic))

    return ConcatDataset(datasets)

