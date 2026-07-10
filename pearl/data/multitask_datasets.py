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
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def _deterministic_features(seed_str: str, n_rows: int, dim: int) -> np.ndarray:
    """
    Generate reproducible feature matrices keyed on a molecule/sequence string.

    The current PEARL trunk (MockPearl) consumes fixed-width numeric feature
    tensors rather than raw sequences/SMILES. Until a learned tokenizer/encoder
    is wired in, we derive a *deterministic* feature block from a hash of the
    identifying string so that identical molecules/targets always map to the
    same features (unlike random synthetic noise, which differs every epoch and
    carries no signal). This keeps the pipeline runnable on the real BindingDB
    labels while remaining stable and reproducible.
    """
    digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2 ** 32)
    rng = np.random.RandomState(seed)
    return rng.randn(n_rows, dim).astype(np.float32)


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

    # Feature dimensions expected by the (Mock)PEARL trunk.
    N_PROTEIN = 100
    N_LIGAND = 30
    FEATURE_DIM = 64

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_synthetic: bool = True,
        tsv_path: Optional[str] = None,
        processed_csv: Optional[str] = None,
        max_samples: Optional[int] = None,
        chunksize: int = 200_000,
        weight: float = 10.0,
        featurizer: str = 'hash',
        esm2_model: str = 'esm2_t33_650M_UR50D',
        molformer_model: str = 'ibm/MoLFormer-XL-both-10pct',
        max_protein_len: int = 200,
        max_ligand_len: int = 64,
        emb_cache_dir: Optional[str] = None,
        feature_device: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic
        self.tsv_path = tsv_path
        # Default processed cache lives next to the data dir.
        self.processed_csv = (
            Path(processed_csv) if processed_csv is not None
            else self.data_dir / 'bindingdb_processed.csv'
        )
        self.max_samples = max_samples
        self.chunksize = chunksize
        self.weight = weight

        # Featurization: 'hash' = deterministic placeholder features (fixed dim
        # 64); 'esm2_molformer' = real ESM2 (protein) + MolFormer (ligand)
        # per-token embeddings, padded/truncated to fixed lengths.
        self.featurizer = featurizer
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        self.protein_featurizer = None
        self.ligand_featurizer = None

        if featurizer == 'esm2_molformer':
            from .featurizers import ESM2Featurizer, MolFormerFeaturizer
            cache_root = (
                Path(emb_cache_dir) if emb_cache_dir is not None
                else self.data_dir / 'emb_cache'
            )
            self.protein_featurizer = ESM2Featurizer(
                model_name=esm2_model, device=feature_device,
                cache_dir=str(cache_root / f'esm2_{esm2_model}'),
            )
            self.ligand_featurizer = MolFormerFeaturizer(
                model_name=molformer_model, device=feature_device,
                cache_dir=str(cache_root / 'molformer'),
            )
            self.protein_dim = self.protein_featurizer.dim
            self.ligand_dim = self.ligand_featurizer.dim
        elif featurizer == 'hash':
            self.protein_dim = self.FEATURE_DIM
            self.ligand_dim = self.FEATURE_DIM
        else:
            raise ValueError(f"Unknown featurizer '{featurizer}'")

        if use_synthetic:
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data()

    def precompute_embeddings(self, batch_size: int = 8, verbose: bool = True):
        """
        Warm the on-disk embedding cache for all unique sequences / SMILES.

        Runs the ESM2 and MolFormer encoders on GPU up front so that later
        DataLoader workers (which must not touch CUDA) only read cached files.
        No-op unless featurizer == 'esm2_molformer'.
        """
        if self.featurizer != 'esm2_molformer':
            return
        seqs = sorted({item['sequence'] for item in self.data})
        smis = sorted({item['smiles'] for item in self.data})
        if verbose:
            print(f"[precompute] {len(seqs)} unique sequences, "
                  f"{len(smis)} unique SMILES")
        self.protein_featurizer.embed_batch(seqs, batch_size=batch_size)
        self.ligand_featurizer.embed_batch(smis, batch_size=max(batch_size, 32))
        # Free encoder GPU memory; DataLoader workers only read the disk cache.
        self.protein_featurizer.release()
        self.ligand_featurizer.release()
        if verbose:
            print("[precompute] embedding cache warmed")

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
        """
        Load real BindingDB data from the processed CSV cache.

        If the processed CSV does not exist yet, it is built by streaming the
        raw BindingDB_All.tsv (see scripts/prepare_bindingdb.py for the same
        logic as a standalone step). Rows are stored as lightweight records;
        fixed-width numeric features are derived deterministically per sample
        in __getitem__ from the target sequence and ligand SMILES.
        """
        import pandas as pd

        if not self.processed_csv.exists():
            if self.tsv_path is None:
                raise FileNotFoundError(
                    f"Processed BindingDB cache not found at {self.processed_csv} "
                    f"and no tsv_path provided. Run scripts/prepare_bindingdb.py "
                    f"or pass tsv_path to build it."
                )
            self._build_processed_csv(self.tsv_path)

        df = pd.read_csv(self.processed_csv)
        if self.max_samples is not None:
            df = df.head(self.max_samples)

        required = {'smiles', 'target_sequence', 'affinity_value'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Processed BindingDB CSV {self.processed_csv} missing columns: {missing}"
            )

        data = []
        for row in df.itertuples(index=False):
            data.append({
                'smiles': row.smiles,
                'sequence': row.target_sequence,
                'affinity_value': float(row.affinity_value),
                'affinity_type': getattr(row, 'affinity_type', 'unknown'),
                'weight': self.weight,
                'data_source': 'bindingdb',
                'task': 'binding_affinity',
            })

        if len(data) == 0:
            raise RuntimeError(
                f"No usable BindingDB rows loaded from {self.processed_csv}."
            )

        return data

    def _build_processed_csv(self, tsv_path: str):
        """Stream the raw BindingDB TSV into the compact processed CSV cache."""
        import pandas as pd

        smiles_col = "Ligand SMILES"
        name_col = "Target Name"
        seq_col = "BindingDB Target Chain Sequence 1"
        affinity_cols = ["Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)"]
        usecols = [smiles_col, name_col, seq_col] + affinity_cols

        kept_frames = []
        total_kept = 0

        reader = pd.read_csv(
            tsv_path, sep="\t", usecols=usecols, chunksize=self.chunksize,
            low_memory=False, on_bad_lines="skip", dtype=str,
        )

        for chunk in reader:
            chunk = chunk[chunk[smiles_col].notna() & chunk[seq_col].notna()].copy()
            if chunk.empty:
                continue

            affinity_nm = pd.Series(np.nan, index=chunk.index, dtype=float)
            affinity_type = pd.Series(np.nan, index=chunk.index, dtype=object)
            for col in affinity_cols:
                vals = pd.to_numeric(
                    chunk[col].astype(str).str.replace(r"[><~=\s]", "", regex=True)
                    .replace({"": np.nan, "nan": np.nan}),
                    errors="coerce",
                )
                take = affinity_nm.isna() & vals.notna() & (vals > 0)
                affinity_nm[take] = vals[take]
                affinity_type[take] = col.split(" ")[0]

            chunk = chunk[affinity_nm.notna()]
            if chunk.empty:
                continue

            affinity_nm = affinity_nm[chunk.index]
            out = pd.DataFrame({
                'smiles': chunk[smiles_col].values,
                'target_name': chunk[name_col].values,
                'target_sequence': chunk[seq_col].values,
                'affinity_type': affinity_type[chunk.index].values,
                'affinity_nm': affinity_nm.values,
                'affinity_value': np.log10(affinity_nm.values / 1000.0),
            })

            if self.max_samples is not None and total_kept + len(out) > self.max_samples:
                out = out.iloc[: self.max_samples - total_kept]

            kept_frames.append(out)
            total_kept += len(out)
            if self.max_samples is not None and total_kept >= self.max_samples:
                break

        if not kept_frames:
            raise RuntimeError(f"No usable rows found in BindingDB TSV {tsv_path}.")

        result = pd.concat(kept_frames, ignore_index=True)
        self.processed_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.processed_csv, index=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            item = self.transform(item)

        result = {
            'target': torch.tensor(item['affinity_value'], dtype=torch.float32),
            'affinity_type': item['affinity_type'],
            'weight': torch.tensor(item['weight'], dtype=torch.float32),
            'task': item['task'],
            'data_source': item['data_source'],
        }

        if self.featurizer == 'esm2_molformer':
            from .featurizers import pad_or_truncate
            # Real per-token embeddings (read from cache; computed on miss).
            prot_emb = self.protein_featurizer.get_cached(item['sequence'])
            if prot_emb is None:
                prot_emb = self.protein_featurizer.embed(item['sequence'])
            lig_emb = self.ligand_featurizer.get_cached(item['smiles'])
            if lig_emb is None:
                lig_emb = self.ligand_featurizer.embed(item['smiles'])

            pf, pm = pad_or_truncate(prot_emb, self.max_protein_len, self.protein_dim)
            lf, lm = pad_or_truncate(lig_emb, self.max_ligand_len, self.ligand_dim)
            result['protein_features'] = torch.from_numpy(pf)
            result['ligand_features'] = torch.from_numpy(lf)
            result['protein_mask'] = torch.from_numpy(pm)
            result['ligand_mask'] = torch.from_numpy(lm)
            return result

        # 'hash' mode. Synthetic records carry precomputed feature arrays; real
        # records derive deterministic placeholder features from their identity.
        if 'protein_features' in item:
            protein_features = item['protein_features']
            ligand_features = item['ligand_features']
        else:
            protein_features = _deterministic_features(
                item['sequence'], self.N_PROTEIN, self.FEATURE_DIM
            )
            ligand_features = _deterministic_features(
                item['smiles'], self.N_LIGAND, self.FEATURE_DIM
            )

        result['protein_features'] = torch.from_numpy(protein_features)
        result['ligand_features'] = torch.from_numpy(ligand_features)
        return result


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
    use_synthetic: bool = True,
    dataset_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
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
        dataset_kwargs: Optional per-dataset extra keyword arguments, keyed by
                        dataset name (e.g. {'bindingdb': {'tsv_path': ...,
                        'processed_csv': ..., 'max_samples': ...}}). Only
                        datasets whose loaders accept these kwargs use them.

    Returns:
        ConcatDataset combining all requested datasets
    """
    datasets = []
    dataset_kwargs = dataset_kwargs or {}

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
        datasets.append(BindingDBDataset(
            data_dirs['bindingdb'], split, use_synthetic=use_synthetic,
            **dataset_kwargs.get('bindingdb', {})
        ))

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

