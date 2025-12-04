"""
Data preprocessing and featurization for Pearl.

Implements atom featurization, coordinate normalization, cropping strategies,
and data augmentation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProteinFeatures:
    """Protein features for Pearl model."""
    coords: np.ndarray  # (n_atoms, 3)
    atom_types: np.ndarray  # (n_atoms,)
    residue_indices: np.ndarray  # (n_atoms,)
    residue_types: np.ndarray  # (n_residues,)
    sequence: str
    msa_features: Optional[np.ndarray] = None  # (n_residues, msa_dim)
    secondary_structure: Optional[np.ndarray] = None  # (n_residues,)


@dataclass
class LigandFeatures:
    """Ligand features for Pearl model."""
    coords: np.ndarray  # (n_atoms, 3)
    atom_types: np.ndarray  # (n_atoms,)
    bonds: np.ndarray  # (n_bonds, 2)
    bond_types: np.ndarray  # (n_bonds,)
    formal_charges: np.ndarray  # (n_atoms,)
    aromatic: np.ndarray  # (n_atoms,) boolean


class ProteinFeaturizer:
    """Featurize protein structures for Pearl."""
    
    # Amino acid vocabulary
    AA_VOCAB = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20  # X for unknown
    }
    
    # Atom type vocabulary
    ATOM_VOCAB = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5, 'X': 6  # X for other
    }
    
    def __init__(self, feature_dim: int = 64):
        """
        Args:
            feature_dim: Dimension of protein features (default: 64 from paper)
        """
        self.feature_dim = feature_dim
    
    def featurize(
        self,
        coords: np.ndarray,
        atoms: List[str],
        residues: np.ndarray,
        sequence: str,
        msa: Optional[np.ndarray] = None,
    ) -> ProteinFeatures:
        """Featurize protein structure.
        
        Args:
            coords: Atom coordinates (n_atoms, 3)
            atoms: Atom element symbols
            residues: Residue index for each atom
            sequence: Amino acid sequence
            msa: Multiple sequence alignment features
            
        Returns:
            ProteinFeatures object
        """
        # Encode atom types
        atom_types = np.array([
            self.ATOM_VOCAB.get(a, self.ATOM_VOCAB['X']) for a in atoms
        ], dtype=np.int64)
        
        # Encode residue types
        residue_types = np.array([
            self.AA_VOCAB.get(aa, self.AA_VOCAB['X']) for aa in sequence
        ], dtype=np.int64)
        
        return ProteinFeatures(
            coords=coords.astype(np.float32),
            atom_types=atom_types,
            residue_indices=residues.astype(np.int64),
            residue_types=residue_types,
            sequence=sequence,
            msa_features=msa,
        )
    
    def compute_msa_features(self, sequence: str, msa_file: Optional[str] = None) -> np.ndarray:
        """Compute MSA features (placeholder).
        
        In practice, this would run MSA search using tools like HHblits or MMseqs2.
        """
        # Placeholder: return random features
        n_residues = len(sequence)
        msa_dim = 256
        return np.random.randn(n_residues, msa_dim).astype(np.float32)


class LigandFeaturizer:
    """Featurize ligand structures for Pearl."""
    
    # Atom type vocabulary (extended for ligands)
    ATOM_VOCAB = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7,
        'I': 8, 'H': 9, 'X': 10  # X for other
    }
    
    # Bond type vocabulary
    BOND_VOCAB = {
        'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3
    }
    
    def __init__(self, feature_dim: int = 64):
        """
        Args:
            feature_dim: Dimension of ligand features (default: 64 from paper)
        """
        self.feature_dim = feature_dim
    
    def featurize(
        self,
        coords: np.ndarray,
        atoms: List[str],
        bonds: Optional[List[Tuple[int, int, str]]] = None,
    ) -> LigandFeatures:
        """Featurize ligand structure.
        
        Args:
            coords: Atom coordinates (n_atoms, 3)
            atoms: Atom element symbols
            bonds: List of (atom1_idx, atom2_idx, bond_type)
            
        Returns:
            LigandFeatures object
        """
        # Encode atom types
        atom_types = np.array([
            self.ATOM_VOCAB.get(a, self.ATOM_VOCAB['X']) for a in atoms
        ], dtype=np.int64)
        
        # Encode bonds
        if bonds:
            bond_indices = np.array([[b[0], b[1]] for b in bonds], dtype=np.int64)
            bond_types = np.array([
                self.BOND_VOCAB.get(b[2], 0) for b in bonds
            ], dtype=np.int64)
        else:
            # Infer bonds from distances
            bond_indices, bond_types = self._infer_bonds(coords)
        
        # Placeholder for other features
        n_atoms = len(atoms)
        formal_charges = np.zeros(n_atoms, dtype=np.int64)
        aromatic = np.zeros(n_atoms, dtype=bool)
        
        return LigandFeatures(
            coords=coords.astype(np.float32),
            atom_types=atom_types,
            bonds=bond_indices,
            bond_types=bond_types,
            formal_charges=formal_charges,
            aromatic=aromatic,
        )
    
    def _infer_bonds(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Infer bonds from atomic coordinates based on distances."""
        n_atoms = len(coords)
        bonds = []
        bond_types = []
        
        # Compute pairwise distances
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                # Typical covalent bond length: 1.0-2.0 Å
                if 0.8 < dist < 2.0:
                    bonds.append([i, j])
                    bond_types.append(0)  # Assume single bond
        
        if bonds:
            return np.array(bonds, dtype=np.int64), np.array(bond_types, dtype=np.int64)
        else:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)


class CroppingStrategy:
    """Cropping strategy for managing large structures.
    
    Pearl uses curriculum training with progressive crop sizes:
    Stage 1-2: 100 atoms
    Stage 3: 200 atoms
    Stage 4: 500 atoms
    Stage 5: 1000 atoms
    Final: Unlimited
    """
    
    def __init__(self, max_atoms: int = 1000, strategy: str = 'pocket_centered'):
        """
        Args:
            max_atoms: Maximum number of atoms to include
            strategy: Cropping strategy ('pocket_centered', 'random', 'full')
        """
        self.max_atoms = max_atoms
        self.strategy = strategy
    
    def crop_protein(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        protein_atoms: np.ndarray,
        protein_residues: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop protein to max_atoms around binding pocket.
        
        Returns:
            Tuple of (cropped_coords, cropped_atoms, cropped_residues)
        """
        n_atoms = len(protein_coords)
        
        if n_atoms <= self.max_atoms or self.strategy == 'full':
            return protein_coords, protein_atoms, protein_residues
        
        if self.strategy == 'pocket_centered':
            # Compute distances to ligand
            ligand_center = ligand_coords.mean(axis=0)
            distances = np.linalg.norm(protein_coords - ligand_center, axis=1)
            
            # Select closest atoms
            indices = np.argsort(distances)[:self.max_atoms]
            indices = np.sort(indices)  # Maintain order
            
        elif self.strategy == 'random':
            # Random sampling
            indices = np.random.choice(n_atoms, self.max_atoms, replace=False)
            indices = np.sort(indices)
        
        else:
            raise ValueError(f"Unknown cropping strategy: {self.strategy}")
        
        return (
            protein_coords[indices],
            protein_atoms[indices],
            protein_residues[indices],
        )


class ComplexPreprocessor:
    """Preprocess protein-ligand complexes for training."""
    
    def __init__(
        self,
        protein_featurizer: ProteinFeaturizer,
        ligand_featurizer: LigandFeaturizer,
        cropping_strategy: CroppingStrategy,
        normalize_coords: bool = True,
        augment: bool = True,
    ):
        """
        Args:
            protein_featurizer: Protein featurizer
            ligand_featurizer: Ligand featurizer
            cropping_strategy: Cropping strategy
            normalize_coords: Whether to normalize coordinates
            augment: Whether to apply data augmentation
        """
        self.protein_featurizer = protein_featurizer
        self.ligand_featurizer = ligand_featurizer
        self.cropping_strategy = cropping_strategy
        self.normalize_coords = normalize_coords
        self.augment = augment
    
    def preprocess(self, complex_data: Dict) -> Dict:
        """Preprocess a protein-ligand complex.
        
        Args:
            complex_data: Dictionary with protein and ligand data
            
        Returns:
            Preprocessed complex ready for model input
        """
        # Extract data
        protein_coords = complex_data['protein_coords']
        ligand_coords = complex_data['ligand_coords']
        
        # Apply cropping
        protein_coords, protein_atoms, protein_residues = self.cropping_strategy.crop_protein(
            protein_coords,
            ligand_coords,
            np.array(complex_data.get('protein_atoms', [])),
            complex_data.get('protein_residues', np.arange(len(protein_coords))),
        )
        
        # Data augmentation (random rotation/translation)
        if self.augment:
            protein_coords, ligand_coords = self._augment(protein_coords, ligand_coords)
        
        # Normalize coordinates
        if self.normalize_coords:
            protein_coords, ligand_coords = self._normalize(protein_coords, ligand_coords)
        
        # Featurize
        protein_features = self.protein_featurizer.featurize(
            protein_coords,
            complex_data.get('protein_atoms', []),
            protein_residues,
            complex_data.get('protein_sequence', ''),
        )
        
        ligand_features = self.ligand_featurizer.featurize(
            ligand_coords,
            complex_data.get('ligand_atoms', []),
            complex_data.get('ligand_bonds', None),
        )
        
        return {
            'protein': protein_features,
            'ligand': ligand_features,
            'pdb_id': complex_data.get('pdb_id', ''),
        }
    
    def _augment(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rotation and translation."""
        # Random rotation
        rotation = self._random_rotation_matrix()
        protein_coords = protein_coords @ rotation.T
        ligand_coords = ligand_coords @ rotation.T
        
        # Random translation
        translation = np.random.randn(3) * 5.0
        protein_coords = protein_coords + translation
        ligand_coords = ligand_coords + translation
        
        return protein_coords, ligand_coords
    
    def _normalize(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize coordinates to zero mean."""
        # Center at complex centroid
        all_coords = np.vstack([protein_coords, ligand_coords])
        center = all_coords.mean(axis=0)
        
        protein_coords = protein_coords - center
        ligand_coords = ligand_coords - center
        
        return protein_coords, ligand_coords
    
    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate random 3D rotation matrix."""
        # Random quaternion
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        
        # Convert to rotation matrix
        w, x, y, z = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
        return R

