"""
Synthetic data generation for Pearl.

Generates protein-ligand complexes using physics-based docking methods
with diverse virtual ligands to augment training data.

Based on Pearl paper: "Derived from public data, this dataset is generated 
using physics-based methods with diverse virtual ligands."
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install with: pip install rdkit")

try:
    from openbabel import openbabel as ob
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    warnings.warn("OpenBabel not available for format conversion")


class VirtualLigandLibrary:
    """Generate diverse virtual ligands for synthetic data."""
    
    def __init__(
        self,
        smiles_file: Optional[str] = None,
        min_heavy_atoms: int = 10,
        max_heavy_atoms: int = 50,
        diversity_threshold: float = 0.3,
    ):
        """
        Args:
            smiles_file: File containing SMILES strings (one per line)
            min_heavy_atoms: Minimum number of heavy atoms
            max_heavy_atoms: Maximum number of heavy atoms
            diversity_threshold: Tanimoto similarity threshold for diversity
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ligand generation")
        
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.diversity_threshold = diversity_threshold
        
        # Load or generate ligand library
        if smiles_file:
            self.ligands = self._load_smiles_library(smiles_file)
        else:
            self.ligands = self._generate_default_library()
    
    def _load_smiles_library(self, smiles_file: str) -> List[Chem.Mol]:
        """Load ligands from SMILES file."""
        ligands = []
        with open(smiles_file, 'r') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and self._is_valid_ligand(mol):
                        # Add hydrogens and generate 3D coordinates
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, randomSeed=42)
                        AllChem.MMFFOptimizeMolecule(mol)
                        ligands.append(mol)
        return ligands
    
    def _generate_default_library(self) -> List[Chem.Mol]:
        """Generate a default library of drug-like molecules."""
        # Common drug-like scaffolds
        scaffolds = [
            'c1ccccc1',  # Benzene
            'c1ccc2ccccc2c1',  # Naphthalene
            'c1ccc2c(c1)ccc1ccccc12',  # Anthracene
            'c1cnc2ccccc2c1',  # Quinoline
            'c1ccc2[nH]ccc2c1',  # Indole
            'c1ccc2c(c1)ncn2',  # Benzimidazole
            'c1ccc2c(c1)oc1ccccc12',  # Dibenzofuran
        ]
        
        # Functional groups to add
        functional_groups = [
            'C(=O)O',  # Carboxylic acid
            'C(=O)N',  # Amide
            'S(=O)(=O)N',  # Sulfonamide
            'C#N',  # Nitrile
            'N',  # Amine
            'O',  # Hydroxyl
            'F',  # Fluorine
            'Cl',  # Chlorine
        ]
        
        ligands = []
        for scaffold_smiles in scaffolds:
            scaffold = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold:
                # Add functional groups
                for fg in functional_groups[:3]:  # Limit combinations
                    try:
                        # Simple combination (this is a placeholder)
                        combined_smiles = scaffold_smiles + fg
                        mol = Chem.MolFromSmiles(combined_smiles)
                        if mol and self._is_valid_ligand(mol):
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol, randomSeed=42)
                            AllChem.MMFFOptimizeMolecule(mol)
                            ligands.append(mol)
                    except:
                        continue
        
        return ligands
    
    def _is_valid_ligand(self, mol: Chem.Mol) -> bool:
        """Check if molecule is a valid ligand."""
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.min_heavy_atoms or n_heavy > self.max_heavy_atoms:
            return False
        
        # Check Lipinski's rule of five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
            return False
        
        return True
    
    def sample_ligand(self) -> Chem.Mol:
        """Sample a random ligand from library."""
        return np.random.choice(self.ligands)
    
    def get_ligand_coords(self, mol: Chem.Mol) -> np.ndarray:
        """Get 3D coordinates of ligand atoms."""
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        return np.array(coords, dtype=np.float32)


class PhysicsBasedDocker:
    """Physics-based docking for synthetic data generation.
    
    This is a simplified implementation. In practice, you would use
    tools like AutoDock Vina, GOLD, or other docking software.
    """
    
    def __init__(self, method: str = 'random_placement'):
        """
        Args:
            method: Docking method ('random_placement', 'vina', 'smina')
        """
        self.method = method
    
    def dock_ligand(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        pocket_center: Optional[np.ndarray] = None,
        pocket_radius: float = 10.0,
    ) -> Tuple[np.ndarray, float]:
        """Dock ligand into protein pocket.
        
        Args:
            protein_coords: Protein atom coordinates (n_protein, 3)
            ligand_coords: Ligand atom coordinates (n_ligand, 3)
            pocket_center: Center of binding pocket
            pocket_radius: Radius of binding pocket
            
        Returns:
            Tuple of (docked_coords, score)
        """
        if self.method == 'random_placement':
            return self._random_placement(
                protein_coords, ligand_coords, pocket_center, pocket_radius
            )
        elif self.method == 'vina':
            return self._dock_with_vina(protein_coords, ligand_coords, pocket_center)
        else:
            raise ValueError(f"Unknown docking method: {self.method}")
    
    def _random_placement(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        pocket_center: Optional[np.ndarray],
        pocket_radius: float,
    ) -> Tuple[np.ndarray, float]:
        """Place ligand randomly in pocket region."""
        # Center ligand at origin
        ligand_centered = ligand_coords - ligand_coords.mean(axis=0)
        
        # Determine pocket center
        if pocket_center is None:
            # Use protein center
            pocket_center = protein_coords.mean(axis=0)
        
        # Random rotation
        rotation = self._random_rotation_matrix()
        ligand_rotated = ligand_centered @ rotation.T
        
        # Random translation within pocket
        offset = np.random.randn(3) * pocket_radius * 0.3
        ligand_placed = ligand_rotated + pocket_center + offset
        
        # Compute simple clash score
        score = self._compute_clash_score(protein_coords, ligand_placed)
        
        return ligand_placed, score
    
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
    
    def _compute_clash_score(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        clash_threshold: float = 2.0,
    ) -> float:
        """Compute clash score (lower is better)."""
        # Compute pairwise distances
        dists = np.linalg.norm(
            protein_coords[:, None, :] - ligand_coords[None, :, :],
            axis=2
        )
        
        # Count clashes
        n_clashes = np.sum(dists < clash_threshold)
        
        # Compute score (penalize clashes)
        score = n_clashes * 10.0
        
        return score
    
    def _dock_with_vina(
        self,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray,
        pocket_center: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Dock using AutoDock Vina (placeholder)."""
        # This would require calling external Vina executable
        # For now, fall back to random placement
        warnings.warn("Vina docking not implemented, using random placement")
        return self._random_placement(protein_coords, ligand_coords, pocket_center, 10.0)


class SyntheticDataGenerator:
    """Generate synthetic protein-ligand complexes for training.
    
    Based on Pearl paper: generates ~582,065 synthetic structures across
    910 proteins with diverse virtual ligands.
    """
    
    def __init__(
        self,
        protein_structures: List[Dict],
        ligand_library: VirtualLigandLibrary,
        docker: PhysicsBasedDocker,
        n_ligands_per_protein: int = 640,
    ):
        """
        Args:
            protein_structures: List of protein structures (from PDB)
            ligand_library: Library of virtual ligands
            docker: Docking engine
            n_ligands_per_protein: Number of ligands to dock per protein
        """
        self.protein_structures = protein_structures
        self.ligand_library = ligand_library
        self.docker = docker
        self.n_ligands_per_protein = n_ligands_per_protein
    
    def generate_dataset(
        self,
        output_dir: str,
        max_structures: Optional[int] = None,
    ) -> List[Dict]:
        """Generate synthetic dataset.
        
        Args:
            output_dir: Directory to save generated structures
            max_structures: Maximum number of structures to generate
            
        Returns:
            List of generated complex dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated = []
        total_count = 0
        
        for protein_idx, protein in enumerate(self.protein_structures):
            if max_structures and total_count >= max_structures:
                break
            
            protein_coords = protein['protein_coords']
            pocket_center = protein.get('pocket_center', None)
            
            # Generate multiple ligand poses for this protein
            for ligand_idx in range(self.n_ligands_per_protein):
                if max_structures and total_count >= max_structures:
                    break
                
                # Sample ligand
                ligand_mol = self.ligand_library.sample_ligand()
                ligand_coords = self.ligand_library.get_ligand_coords(ligand_mol)
                
                # Dock ligand
                docked_coords, score = self.docker.dock_ligand(
                    protein_coords, ligand_coords, pocket_center
                )
                
                # Create complex
                complex_data = {
                    'protein_id': protein.get('pdb_id', f'protein_{protein_idx}'),
                    'ligand_id': f'ligand_{ligand_idx}',
                    'protein_coords': protein_coords,
                    'protein_atoms': protein.get('protein_atoms', []),
                    'protein_sequence': protein.get('protein_sequence', ''),
                    'ligand_coords': docked_coords,
                    'ligand_atoms': [ligand_mol.GetAtomWithIdx(i).GetSymbol() 
                                    for i in range(ligand_mol.GetNumAtoms())],
                    'docking_score': score,
                    'is_synthetic': True,
                }
                
                generated.append(complex_data)
                total_count += 1
                
                # Save periodically
                if total_count % 1000 == 0:
                    print(f"Generated {total_count} synthetic structures...")
        
        # Save metadata
        metadata = {
            'n_structures': len(generated),
            'n_proteins': len(self.protein_structures),
            'n_ligands_per_protein': self.n_ligands_per_protein,
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {len(generated)} synthetic structures")
        return generated

