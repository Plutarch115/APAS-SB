"""
Dataset for ΔΔG prediction training and evaluation.

Handles loading of wild-type/mutant pairs with experimental ΔΔG values.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DDGDataPoint:
    """Single ΔΔG training example"""
    wt_protein_features: torch.Tensor
    wt_ligand_features: torch.Tensor
    mut_protein_features: torch.Tensor
    mut_ligand_features: torch.Tensor
    protein_mask: torch.Tensor
    ligand_mask: torch.Tensor
    ddg_exp: float  # kcal/mol
    ddg_error: float  # Experimental uncertainty
    mutation: str  # e.g., "A:L99A"
    data_source: str  # "experimental", "md_fep", "pseudo_label"
    temperature: float = 298.0  # K
    ph: float = 7.4


class DDGDataset(Dataset):
    """
    Dataset for ΔΔG prediction training.
    
    Loads wild-type and mutant structure pairs with experimental ΔΔG values.
    """
    
    def __init__(
        self,
        data_points: List[DDGDataPoint],
        data_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            data_points: List of ΔΔG training examples
            data_weights: Weights for different data sources
        """
        self.data_points = data_points
        
        if data_weights is None:
            data_weights = {
                "experimental": 10.0,
                "md_fep": 1.0,
                "pseudo_label": 0.1
            }
        self.data_weights = data_weights
        
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a single training example"""
        dp = self.data_points[idx]
        
        # Get data weight
        weight = self.data_weights.get(dp.data_source, 1.0)
        
        return {
            'wt_protein_features': dp.wt_protein_features,
            'wt_ligand_features': dp.wt_ligand_features,
            'mut_protein_features': dp.mut_protein_features,
            'mut_ligand_features': dp.mut_ligand_features,
            'protein_mask': dp.protein_mask,
            'ligand_mask': dp.ligand_mask,
            'ddg_true': torch.tensor(dp.ddg_exp, dtype=torch.float32),
            'ddg_error': torch.tensor(dp.ddg_error, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'mutation': dp.mutation,
            'data_source': dp.data_source
        }


class SyntheticDDGDataset(Dataset):
    """
    Synthetic dataset for testing ΔΔG prediction.
    
    Generates random protein/ligand features with synthetic ΔΔG values
    based on simple rules.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        n_protein: int = 100,
        n_ligand: int = 20,
        protein_feature_dim: int = 64,
        ligand_feature_dim: int = 64,
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of synthetic examples
            n_protein: Number of protein residues
            n_ligand: Number of ligand atoms
            protein_feature_dim: Protein feature dimension
            ligand_feature_dim: Ligand feature dimension
            seed: Random seed
        """
        self.num_samples = num_samples
        self.n_protein = n_protein
        self.n_ligand = n_ligand
        self.protein_feature_dim = protein_feature_dim
        self.ligand_feature_dim = ligand_feature_dim
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic ΔΔG data"""
        data = []
        
        for i in range(self.num_samples):
            # Generate wild-type features
            wt_protein = torch.randn(self.n_protein, self.protein_feature_dim)
            wt_ligand = torch.randn(self.n_ligand, self.ligand_feature_dim)
            
            # Generate mutant features (small perturbation)
            # Randomly select mutation site
            mut_site = np.random.randint(0, self.n_protein)
            
            mut_protein = wt_protein.clone()
            # Perturb mutation site
            perturbation = torch.randn(self.protein_feature_dim) * 0.5
            mut_protein[mut_site] = mut_protein[mut_site] + perturbation
            
            # Optionally perturb nearby residues (local effect)
            for offset in [-2, -1, 1, 2]:
                nearby_site = mut_site + offset
                if 0 <= nearby_site < self.n_protein:
                    local_perturbation = torch.randn(self.protein_feature_dim) * 0.1
                    mut_protein[nearby_site] = mut_protein[nearby_site] + local_perturbation
            
            # Ligand stays the same (protein mutation)
            mut_ligand = wt_ligand.clone()
            
            # Compute synthetic ΔΔG based on perturbation magnitude
            # Larger perturbations = larger ΔΔG
            perturbation_magnitude = torch.norm(perturbation).item()
            
            # Add some nonlinearity and noise
            ddg_true = perturbation_magnitude * 2.0 + np.random.randn() * 0.5
            
            # Randomly make some mutations favorable (negative ΔΔG)
            if np.random.rand() < 0.3:
                ddg_true = -ddg_true
            
            # Experimental error
            ddg_error = 0.3 + np.random.rand() * 0.3  # 0.3-0.6 kcal/mol
            
            # Masks (all valid)
            protein_mask = torch.ones(self.n_protein, dtype=torch.bool)
            ligand_mask = torch.ones(self.n_ligand, dtype=torch.bool)
            
            # Mutation label
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            wt_aa = amino_acids[np.random.randint(0, 20)]
            mut_aa = amino_acids[np.random.randint(0, 20)]
            mutation = f'{wt_aa}{mut_site}{mut_aa}'
            
            # Data source (mostly experimental for synthetic data)
            data_source = 'experimental'
            
            # Weight
            weight = 10.0
            
            data.append({
                'wt_protein_features': wt_protein,
                'wt_ligand_features': wt_ligand,
                'mut_protein_features': mut_protein,
                'mut_ligand_features': mut_ligand,
                'protein_mask': protein_mask,
                'ligand_mask': ligand_mask,
                'ddg_true': torch.tensor(ddg_true, dtype=torch.float32),
                'ddg_error': torch.tensor(ddg_error, dtype=torch.float32),
                'weight': torch.tensor(weight, dtype=torch.float32),
                'mutation': mutation,
                'data_source': data_source
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_ddg_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for ΔΔG dataset.
    
    Args:
        batch: List of data dictionaries
        
    Returns:
        Batched dictionary
    """
    # Stack tensors
    batched = {
        'wt_protein_features': torch.stack([b['wt_protein_features'] for b in batch]),
        'wt_ligand_features': torch.stack([b['wt_ligand_features'] for b in batch]),
        'mut_protein_features': torch.stack([b['mut_protein_features'] for b in batch]),
        'mut_ligand_features': torch.stack([b['mut_ligand_features'] for b in batch]),
        'protein_mask': torch.stack([b['protein_mask'] for b in batch]),
        'ligand_mask': torch.stack([b['ligand_mask'] for b in batch]),
        'ddg_true': torch.stack([b['ddg_true'] for b in batch]),
        'ddg_error': torch.stack([b['ddg_error'] for b in batch]),
        'weight': torch.stack([b['weight'] for b in batch]),
    }
    
    # Keep lists for non-tensor data
    batched['mutation'] = [b['mutation'] for b in batch]
    batched['data_source'] = [b['data_source'] for b in batch]
    
    return batched


def create_train_val_split(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Full dataset
        val_fraction: Fraction for validation
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset
    """
    torch.manual_seed(seed)
    
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    return train_dataset, val_dataset

