"""
Protein-Protein Interaction (PPI) Data Loader for Pearl

This module extends Pearl to handle protein-protein complexes for:
- Biologics drug discovery
- Antibody-antigen prediction
- Protein interface modeling
- Multi-protein assemblies

Integrates with existing Pearl architecture for unified training.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Chain import Chain
except ImportError:
    raise ImportError("Biopython not installed. Install with: pip install biopython")

logger = logging.getLogger(__name__)


@dataclass
class PPIComplex:
    """Protein-protein interaction complex"""
    
    pdb_id: str
    chain_a_id: str
    chain_b_id: str
    chain_a_sequence: str
    chain_b_sequence: str
    chain_a_coords: np.ndarray  # (N_a, 3)
    chain_b_coords: np.ndarray  # (N_b, 3)
    interface_residues_a: Set[int]
    interface_residues_b: Set[int]
    resolution: Optional[float] = None
    method: Optional[str] = None
    
    def __post_init__(self):
        """Validate complex"""
        assert self.chain_a_coords.shape[1] == 3
        assert self.chain_b_coords.shape[1] == 3
        assert len(self.chain_a_sequence) > 0
        assert len(self.chain_b_sequence) > 0


class PPIDataLoader:
    """
    Load and process protein-protein interaction complexes from PDB.
    
    Features:
    - Extract protein-protein interfaces
    - Identify interface residues
    - Compute interface metrics
    - Support for antibody-antigen complexes
    - Compatible with Pearl's multi-chain templating
    """
    
    def __init__(self, interface_distance_cutoff: float = 8.0):
        """
        Initialize PPI data loader.
        
        Args:
            interface_distance_cutoff: Distance cutoff (Å) for interface residues
        """
        self.parser = PDBParser(QUIET=True)
        self.interface_cutoff = interface_distance_cutoff
        
    def load_ppi_complex(
        self,
        pdb_file: str,
        chain_a_id: str,
        chain_b_id: str
    ) -> PPIComplex:
        """
        Load protein-protein complex from PDB file.
        
        Args:
            pdb_file: Path to PDB file
            chain_a_id: Chain ID for first protein
            chain_b_id: Chain ID for second protein
            
        Returns:
            PPIComplex object
        """
        logger.info(f"Loading PPI complex from {pdb_file}...")
        
        # Parse structure
        structure = self.parser.get_structure("complex", pdb_file)
        
        # Get chains
        chain_a = None
        chain_b = None
        
        for model in structure:
            for chain in model:
                if chain.id == chain_a_id:
                    chain_a = chain
                elif chain.id == chain_b_id:
                    chain_b = chain
        
        if chain_a is None or chain_b is None:
            raise ValueError(f"Could not find chains {chain_a_id} and {chain_b_id} in {pdb_file}")
        
        # Extract sequences and coordinates
        chain_a_seq, chain_a_coords = self._extract_chain_data(chain_a)
        chain_b_seq, chain_b_coords = self._extract_chain_data(chain_b)
        
        # Identify interface residues
        interface_a, interface_b = self._identify_interface_residues(
            chain_a_coords,
            chain_b_coords
        )
        
        # Extract metadata
        pdb_id = Path(pdb_file).stem
        resolution = self._extract_resolution(structure)
        method = self._extract_method(structure)
        
        complex_obj = PPIComplex(
            pdb_id=pdb_id,
            chain_a_id=chain_a_id,
            chain_b_id=chain_b_id,
            chain_a_sequence=chain_a_seq,
            chain_b_sequence=chain_b_seq,
            chain_a_coords=chain_a_coords,
            chain_b_coords=chain_b_coords,
            interface_residues_a=interface_a,
            interface_residues_b=interface_b,
            resolution=resolution,
            method=method
        )
        
        logger.info(f"Loaded PPI complex: {pdb_id}")
        logger.info(f"  Chain A: {len(chain_a_seq)} residues, {len(interface_a)} interface residues")
        logger.info(f"  Chain B: {len(chain_b_seq)} residues, {len(interface_b)} interface residues")
        logger.info(f"  Resolution: {resolution} Å" if resolution else "  Resolution: N/A")
        
        return complex_obj
    
    def _extract_chain_data(self, chain: Chain) -> Tuple[str, np.ndarray]:
        """
        Extract sequence and CA coordinates from chain.
        
        Args:
            chain: BioPython Chain object
            
        Returns:
            Tuple of (sequence, coordinates)
        """
        from Bio.SeqUtils import seq1
        
        sequence = []
        coords = []
        
        for residue in chain:
            # Skip hetero residues
            if residue.id[0] != ' ':
                continue
            
            # Get residue name
            resname = residue.resname
            
            # Convert to single letter code
            try:
                aa = seq1(resname)
            except KeyError:
                logger.warning(f"Unknown residue: {resname}")
                continue
            
            sequence.append(aa)
            
            # Get CA coordinates
            if 'CA' in residue:
                coords.append(residue['CA'].coord)
            else:
                logger.warning(f"No CA atom in residue {residue.id}")
                coords.append(np.array([0.0, 0.0, 0.0]))
        
        return ''.join(sequence), np.array(coords)
    
    def _identify_interface_residues(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify interface residues based on distance cutoff.
        
        Args:
            coords_a: Coordinates of chain A (N_a, 3)
            coords_b: Coordinates of chain B (N_b, 3)
            
        Returns:
            Tuple of (interface_residues_a, interface_residues_b)
        """
        # Compute pairwise distances
        distances = np.linalg.norm(
            coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :],
            axis=2
        )
        
        # Find residues within cutoff
        interface_a = set(np.where(distances.min(axis=1) < self.interface_cutoff)[0].tolist())
        interface_b = set(np.where(distances.min(axis=0) < self.interface_cutoff)[0].tolist())
        
        return interface_a, interface_b
    
    def _extract_resolution(self, structure: Structure) -> Optional[float]:
        """Extract resolution from structure"""
        try:
            return structure.header.get('resolution')
        except:
            return None
    
    def _extract_method(self, structure: Structure) -> Optional[str]:
        """Extract experimental method from structure"""
        try:
            return structure.header.get('structure_method')
        except:
            return None
    
    def compute_interface_metrics(self, complex_obj: PPIComplex) -> Dict[str, float]:
        """
        Compute interface metrics for PPI complex.
        
        Args:
            complex_obj: PPIComplex object
            
        Returns:
            Dictionary of interface metrics
        """
        # Interface size
        n_interface_a = len(complex_obj.interface_residues_a)
        n_interface_b = len(complex_obj.interface_residues_b)
        
        # Interface area (approximate)
        # Each residue contributes ~20 Ų to interface
        interface_area = (n_interface_a + n_interface_b) * 20.0
        
        # Interface RMSD (for evaluation)
        interface_coords_a = complex_obj.chain_a_coords[list(complex_obj.interface_residues_a)]
        interface_coords_b = complex_obj.chain_b_coords[list(complex_obj.interface_residues_b)]
        
        # Center of mass distance
        com_a = interface_coords_a.mean(axis=0)
        com_b = interface_coords_b.mean(axis=0)
        com_distance = np.linalg.norm(com_a - com_b)
        
        metrics = {
            "n_interface_residues_a": n_interface_a,
            "n_interface_residues_b": n_interface_b,
            "interface_area_approx": interface_area,
            "com_distance": com_distance,
            "interface_fraction_a": n_interface_a / len(complex_obj.chain_a_sequence),
            "interface_fraction_b": n_interface_b / len(complex_obj.chain_b_sequence)
        }
        
        return metrics
    
    def save_interface_pdb(
        self,
        complex_obj: PPIComplex,
        output_file: str,
        include_full_chains: bool = False
    ):
        """
        Save interface residues to PDB file.
        
        Args:
            complex_obj: PPIComplex object
            output_file: Output PDB file path
            include_full_chains: If True, save full chains; if False, only interface
        """
        # This is a simplified version - in production, use BioPython's PDBIO
        logger.info(f"Saving interface to {output_file}")
        
        # For now, just log the action
        # Full implementation would use PDBIO with custom Select class
        logger.warning("save_interface_pdb not fully implemented - placeholder only")


class PPIDataset:
    """
    Dataset for protein-protein interaction complexes.
    
    Compatible with Pearl's training pipeline.
    """
    
    def __init__(
        self,
        pdb_dir: str,
        ppi_list_file: Optional[str] = None,
        interface_cutoff: float = 8.0
    ):
        """
        Initialize PPI dataset.
        
        Args:
            pdb_dir: Directory containing PDB files
            ppi_list_file: File with list of PPI complexes (format: pdb_id chain_a chain_b)
            interface_cutoff: Distance cutoff for interface residues
        """
        self.pdb_dir = Path(pdb_dir)
        self.loader = PPIDataLoader(interface_cutoff)
        self.complexes = []
        
        if ppi_list_file:
            self._load_from_list(ppi_list_file)
    
    def _load_from_list(self, list_file: str):
        """Load PPI complexes from list file"""
        logger.info(f"Loading PPI list from {list_file}...")
        
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    logger.warning(f"Invalid line: {line}")
                    continue
                
                pdb_id, chain_a, chain_b = parts[:3]
                pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
                
                if not pdb_file.exists():
                    logger.warning(f"PDB file not found: {pdb_file}")
                    continue
                
                try:
                    complex_obj = self.loader.load_ppi_complex(
                        str(pdb_file),
                        chain_a,
                        chain_b
                    )
                    self.complexes.append(complex_obj)
                except Exception as e:
                    logger.error(f"Error loading {pdb_id}: {e}")
        
        logger.info(f"Loaded {len(self.complexes)} PPI complexes")
    
    def __len__(self) -> int:
        return len(self.complexes)
    
    def __getitem__(self, idx: int) -> PPIComplex:
        return self.complexes[idx]

