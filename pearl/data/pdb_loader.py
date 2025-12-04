"""
PDB data loading and parsing for Pearl.

Loads protein-ligand complexes from PDB files and prepares them for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

try:
    from Bio import PDB
    from Bio.PDB import PDBIO, Select
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available. Install with: pip install biopython")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install with: pip install rdkit")


class PDBParser:
    """Parse PDB files to extract protein-ligand complexes."""
    
    def __init__(self, pdb_dir: str):
        """
        Args:
            pdb_dir: Directory containing PDB files
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for PDB parsing")
        
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDB.PDBParser(QUIET=True)
        
    def parse_structure(self, pdb_id: str) -> Optional[PDB.Structure.Structure]:
        """Parse a PDB structure.
        
        Args:
            pdb_id: PDB ID (e.g., '1ABC')
            
        Returns:
            BioPython Structure object or None if parsing fails
        """
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_file.exists():
            # Try with .ent extension
            pdb_file = self.pdb_dir / f"pdb{pdb_id.lower()}.ent"
        
        if not pdb_file.exists():
            warnings.warn(f"PDB file not found: {pdb_id}")
            return None
        
        try:
            structure = self.parser.get_structure(pdb_id, str(pdb_file))
            return structure
        except Exception as e:
            warnings.warn(f"Failed to parse {pdb_id}: {e}")
            return None
    
    def extract_protein_chains(self, structure: PDB.Structure.Structure) -> List[PDB.Chain.Chain]:
        """Extract protein chains from structure."""
        protein_chains = []
        for model in structure:
            for chain in model:
                # Check if chain contains amino acids
                residues = list(chain.get_residues())
                if residues and self._is_protein_chain(residues):
                    protein_chains.append(chain)
        return protein_chains
    
    def extract_ligands(self, structure: PDB.Structure.Structure) -> List[PDB.Residue.Residue]:
        """Extract ligand residues (HETATM) from structure."""
        ligands = []
        # Common solvent/ion molecules to exclude
        exclude_resnames = {'HOH', 'WAT', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'SO4', 'PO4'}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag, resseq, icode = residue.get_id()
                    # HETATM residues have hetflag starting with 'H_'
                    if hetflag.startswith('H_') and residue.get_resname() not in exclude_resnames:
                        # Check if it has enough atoms to be a ligand (>= 5 heavy atoms)
                        heavy_atoms = [a for a in residue.get_atoms() if a.element != 'H']
                        if len(heavy_atoms) >= 5:
                            ligands.append(residue)
        return ligands
    
    def _is_protein_chain(self, residues: List[PDB.Residue.Residue]) -> bool:
        """Check if residues form a protein chain."""
        standard_aa = {
            'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
            'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
        }
        aa_count = sum(1 for r in residues if r.get_resname() in standard_aa)
        return aa_count / len(residues) > 0.8  # At least 80% standard amino acids
    
    def get_atom_coordinates(self, residue: PDB.Residue.Residue) -> np.ndarray:
        """Extract atom coordinates from residue.
        
        Returns:
            Array of shape (n_atoms, 3)
        """
        coords = []
        for atom in residue.get_atoms():
            coords.append(atom.get_coord())
        return np.array(coords)
    
    def get_chain_sequence(self, chain: PDB.Chain.Chain) -> str:
        """Extract amino acid sequence from chain."""
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        sequence = []
        for residue in chain.get_residues():
            resname = residue.get_resname()
            if resname in three_to_one:
                sequence.append(three_to_one[resname])
            else:
                sequence.append('X')  # Unknown amino acid
        return ''.join(sequence)


class PDBDataset(Dataset):
    """Dataset for loading protein-ligand complexes from PDB files."""
    
    def __init__(
        self,
        pdb_dir: str,
        pdb_ids: Optional[List[str]] = None,
        max_protein_atoms: int = 5000,
        max_ligand_atoms: int = 100,
        release_date_cutoff: Optional[str] = None,
    ):
        """
        Args:
            pdb_dir: Directory containing PDB files
            pdb_ids: List of PDB IDs to load. If None, loads all PDB files in directory
            max_protein_atoms: Maximum number of protein atoms to include
            max_ligand_atoms: Maximum number of ligand atoms
            release_date_cutoff: Only include structures released before this date (YYYY-MM-DD)
        """
        self.pdb_dir = Path(pdb_dir)
        self.max_protein_atoms = max_protein_atoms
        self.max_ligand_atoms = max_ligand_atoms
        self.parser = PDBParser(pdb_dir)
        
        # Get list of PDB IDs
        if pdb_ids is None:
            pdb_ids = self._discover_pdb_files()
        
        # Filter by release date if specified
        if release_date_cutoff:
            pdb_ids = self._filter_by_release_date(pdb_ids, release_date_cutoff)
        
        self.pdb_ids = pdb_ids
        
        # Cache for parsed structures
        self._structure_cache = {}
    
    def _discover_pdb_files(self) -> List[str]:
        """Discover all PDB files in directory."""
        pdb_files = list(self.pdb_dir.glob("*.pdb")) + list(self.pdb_dir.glob("*.ent"))
        pdb_ids = []
        for f in pdb_files:
            # Extract PDB ID from filename
            name = f.stem
            if name.startswith('pdb'):
                pdb_id = name[3:7].upper()
            else:
                pdb_id = name[:4].upper()
            pdb_ids.append(pdb_id)
        return pdb_ids
    
    def _filter_by_release_date(self, pdb_ids: List[str], cutoff: str) -> List[str]:
        """Filter PDB IDs by release date."""
        # This would require accessing PDB metadata
        # For now, return all IDs
        warnings.warn("Release date filtering not implemented yet")
        return pdb_ids
    
    def __len__(self) -> int:
        return len(self.pdb_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a protein-ligand complex.

        Returns:
            Dictionary containing:
                - pdb_id: PDB identifier
                - protein_coords: Protein atom coordinates (n_protein_atoms, 3)
                - protein_atoms: Protein atom types
                - protein_residues: Residue indices
                - protein_sequence: Amino acid sequence
                - protein_bfactors: B-factors for protein atoms
                - ligand_coords: Ligand atom coordinates (n_ligand_atoms, 3)
                - ligand_atoms: Ligand atom types
                - ligand_bfactors: B-factors for ligand atoms
                - ligand_bonds: Bond connectivity
                - resolution: Overall structure resolution (Å)
                - experimental_method: 'xray', 'em', or 'nmr'
        """
        pdb_id = self.pdb_ids[idx]

        # Parse structure
        structure = self.parser.parse_structure(pdb_id)
        if structure is None:
            # Return empty sample
            return self._empty_sample(pdb_id)

        # Extract protein chains
        protein_chains = self.parser.extract_protein_chains(structure)
        if not protein_chains:
            return self._empty_sample(pdb_id)

        # Extract ligands
        ligands = self.parser.extract_ligands(structure)
        if not ligands:
            return self._empty_sample(pdb_id)

        # Use first protein chain and first ligand for simplicity
        # In practice, you'd want to handle multiple chains/ligands
        chain = protein_chains[0]
        ligand = ligands[0]

        # Extract protein data (including B-factors)
        protein_coords = []
        protein_atoms = []
        protein_residues = []
        protein_bfactors = []

        for res_idx, residue in enumerate(chain.get_residues()):
            for atom in residue.get_atoms():
                if atom.element != 'H':  # Skip hydrogens
                    protein_coords.append(atom.get_coord())
                    protein_atoms.append(atom.element)
                    protein_residues.append(res_idx)
                    protein_bfactors.append(atom.bfactor)

        protein_sequence = self.parser.get_chain_sequence(chain)

        # Extract ligand data (including B-factors)
        ligand_coords = []
        ligand_atoms = []
        ligand_bfactors = []
        for atom in ligand.get_atoms():
            if atom.element != 'H':  # Skip hydrogens
                ligand_coords.append(atom.get_coord())
                ligand_atoms.append(atom.element)
                ligand_bfactors.append(atom.bfactor)

        # Extract experimental metadata
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        method, resolution = self._extract_experimental_metadata(pdb_file)

        return {
            'pdb_id': pdb_id,
            'protein_coords': np.array(protein_coords, dtype=np.float32),
            'protein_atoms': protein_atoms,
            'protein_residues': np.array(protein_residues, dtype=np.int64),
            'protein_sequence': protein_sequence,
            'protein_bfactors': np.array(protein_bfactors, dtype=np.float32),
            'ligand_coords': np.array(ligand_coords, dtype=np.float32),
            'ligand_atoms': ligand_atoms,
            'ligand_bfactors': np.array(ligand_bfactors, dtype=np.float32),
            'ligand_bonds': [],  # Would need to infer or load from SDF
            'resolution': resolution,
            'experimental_method': method,
        }
    
    def _extract_experimental_metadata(self, pdb_file: Path) -> Tuple[str, Optional[float]]:
        """Extract experimental method and resolution from PDB header."""
        method = 'xray'  # Default
        resolution = None

        if not pdb_file.exists():
            return method, resolution

        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('EXPDTA'):
                        if 'ELECTRON MICROSCOPY' in line:
                            method = 'em'
                        elif 'NMR' in line:
                            method = 'nmr'
                        elif 'X-RAY' in line or 'DIFFRACTION' in line:
                            method = 'xray'
                    elif line.startswith('REMARK   2 RESOLUTION'):
                        try:
                            resolution = float(line.split()[3])
                        except (IndexError, ValueError):
                            pass
        except Exception as e:
            warnings.warn(f"Failed to extract metadata from {pdb_file}: {e}")

        return method, resolution

    def _empty_sample(self, pdb_id: str) -> Dict:
        """Return an empty sample for failed parsing."""
        return {
            'pdb_id': pdb_id,
            'protein_coords': np.zeros((0, 3), dtype=np.float32),
            'protein_atoms': [],
            'protein_residues': np.zeros(0, dtype=np.int64),
            'protein_sequence': '',
            'protein_bfactors': np.zeros(0, dtype=np.float32),
            'ligand_coords': np.zeros((0, 3), dtype=np.float32),
            'ligand_atoms': [],
            'ligand_bfactors': np.zeros(0, dtype=np.float32),
            'ligand_bonds': [],
            'resolution': None,
            'experimental_method': 'xray',
        }


class PDBDataLoader:
    """DataLoader wrapper for PDB dataset with batching."""
    
    def __init__(
        self,
        dataset: PDBDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def __iter__(self):
        # For now, simple iteration without batching
        # In practice, you'd implement proper batching with padding
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            yield self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

