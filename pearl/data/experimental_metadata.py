"""
Experimental Metadata Extraction for Pearl

Extracts experimental uncertainty information from structural data:
- B-factors (temperature factors) from X-ray crystallography
- Local resolution maps from CryoEM
- Overall resolution and quality metrics
- Per-atom confidence scores

This enables resolution-aware training where high-confidence regions
are weighted more heavily than uncertain regions.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings

try:
    from Bio import PDB
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("BioPython not available")

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    warnings.warn("mrcfile not available. Install with: pip install mrcfile")


class ExperimentalUncertainty:
    """
    Container for experimental uncertainty information.
    
    Stores per-atom confidence scores derived from:
    - B-factors (X-ray)
    - Local resolution (CryoEM)
    - Occupancy values
    - Overall resolution
    """
    
    def __init__(
        self,
        atom_confidence: np.ndarray,  # [n_atoms] - higher is better
        method: str,  # 'xray', 'em', 'nmr', 'predicted'
        resolution: Optional[float] = None,  # Overall resolution (Å)
        source: str = 'bfactor',  # 'bfactor', 'local_resolution', 'plddt'
    ):
        """
        Args:
            atom_confidence: Per-atom confidence scores [0, 1]
            method: Experimental method
            resolution: Overall resolution in Angstroms
            source: Source of confidence scores
        """
        self.atom_confidence = atom_confidence
        self.method = method
        self.resolution = resolution
        self.source = source
        
    def get_loss_weights(self, weighting_scheme: str = 'linear') -> np.ndarray:
        """
        Convert confidence scores to loss weights.
        
        Args:
            weighting_scheme: How to convert confidence to weights
                - 'linear': Direct use of confidence
                - 'squared': Square of confidence (emphasize high-confidence)
                - 'inverse_variance': Treat as inverse variance
                - 'sigmoid': Sigmoid transformation
                
        Returns:
            Loss weights [n_atoms]
        """
        if weighting_scheme == 'linear':
            weights = self.atom_confidence
        elif weighting_scheme == 'squared':
            weights = self.atom_confidence ** 2
        elif weighting_scheme == 'inverse_variance':
            # Confidence ~ 1/sigma, so weight ~ 1/sigma^2
            weights = self.atom_confidence ** 2
        elif weighting_scheme == 'sigmoid':
            # Sigmoid to smooth the weighting
            weights = 1.0 / (1.0 + np.exp(-5 * (self.atom_confidence - 0.5)))
        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        
        # Normalize so mean weight is 1.0 (preserves loss scale)
        weights = weights / (weights.mean() + 1e-8)
        
        return weights


class BFactorExtractor:
    """
    Extract B-factors from PDB/mmCIF files and convert to confidence scores.
    
    B-factors (temperature factors) represent atomic displacement parameters.
    Lower B-factors indicate more certain positions.
    """
    
    def __init__(
        self,
        normalization: str = 'per_structure',  # 'per_structure', 'per_chain', 'global'
        outlier_clip: float = 3.0,  # Clip outliers beyond N standard deviations
    ):
        """
        Args:
            normalization: How to normalize B-factors
            outlier_clip: Clip B-factors beyond this many std devs
        """
        self.normalization = normalization
        self.outlier_clip = outlier_clip
        
    def extract_from_structure(
        self,
        structure: 'PDB.Structure.Structure',
        chain_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract B-factors from BioPython structure.
        
        Args:
            structure: BioPython Structure object
            chain_ids: Optional list of chain IDs to extract
            
        Returns:
            Dictionary mapping chain_id -> B-factors array
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython required for B-factor extraction")
        
        chain_bfactors = {}
        
        for model in structure:
            for chain in model:
                if chain_ids and chain.id not in chain_ids:
                    continue
                
                bfactors = []
                for residue in chain:
                    for atom in residue:
                        if atom.element != 'H':  # Skip hydrogens
                            bfactors.append(atom.bfactor)
                
                if bfactors:
                    chain_bfactors[chain.id] = np.array(bfactors)
        
        return chain_bfactors
    
    def bfactor_to_confidence(
        self,
        bfactors: np.ndarray,
        resolution: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert B-factors to confidence scores [0, 1].
        
        Lower B-factors -> higher confidence
        
        Args:
            bfactors: Array of B-factors
            resolution: Overall resolution (used for normalization)
            
        Returns:
            Confidence scores [0, 1]
        """
        bfactors = np.array(bfactors, dtype=np.float32)
        
        # Clip outliers
        if self.outlier_clip > 0:
            mean_b = np.mean(bfactors)
            std_b = np.std(bfactors)
            bfactors = np.clip(
                bfactors,
                mean_b - self.outlier_clip * std_b,
                mean_b + self.outlier_clip * std_b
            )
        
        # Normalize to [0, 1] range
        # Lower B-factor = higher confidence
        min_b = np.min(bfactors)
        max_b = np.max(bfactors)
        
        if max_b - min_b < 1e-6:
            # All B-factors are the same
            confidence = np.ones_like(bfactors)
        else:
            # Invert: low B-factor -> high confidence
            confidence = 1.0 - (bfactors - min_b) / (max_b - min_b)
        
        # Resolution-dependent scaling
        # Worse resolution -> lower overall confidence
        if resolution is not None:
            # Scale by resolution: better resolution (lower value) -> higher confidence
            # Typical range: 1.0-4.0 Å for X-ray
            resolution_factor = np.clip(1.0 / (resolution / 2.0), 0.3, 1.0)
            confidence = confidence * resolution_factor
        
        return confidence


class CryoEMLocalResolution:
    """
    Extract local resolution information from CryoEM maps.
    
    CryoEM structures often have variable local resolution across the structure.
    This information is stored in local resolution maps (MRC format).
    """
    
    def __init__(
        self,
        resolution_range: Tuple[float, float] = (2.0, 10.0),  # Typical CryoEM range (Å)
    ):
        """
        Args:
            resolution_range: (min, max) resolution in Angstroms
        """
        self.resolution_range = resolution_range
        
    def load_local_resolution_map(self, mrc_file: Path) -> np.ndarray:
        """
        Load local resolution map from MRC file.
        
        Args:
            mrc_file: Path to MRC file containing local resolution
            
        Returns:
            3D array of local resolution values
        """
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile required. Install with: pip install mrcfile")
        
        with mrcfile.open(mrc_file, mode='r') as mrc:
            resolution_map = mrc.data.copy()
        
        return resolution_map
    
    def interpolate_atom_resolution(
        self,
        atom_coords: np.ndarray,  # [n_atoms, 3]
        resolution_map: np.ndarray,  # [nx, ny, nz]
        voxel_size: float,  # Angstroms per voxel
        origin: np.ndarray,  # Map origin [3]
    ) -> np.ndarray:
        """
        Interpolate local resolution at atom positions.
        
        Args:
            atom_coords: Atom coordinates in Angstroms
            resolution_map: 3D local resolution map
            voxel_size: Size of each voxel in Angstroms
            origin: Origin of the map in Angstroms
            
        Returns:
            Local resolution at each atom position
        """
        # Convert atom coordinates to voxel indices
        voxel_coords = (atom_coords - origin) / voxel_size
        
        # Trilinear interpolation
        atom_resolutions = []
        
        for coord in voxel_coords:
            # Get integer and fractional parts
            i, j, k = np.floor(coord).astype(int)
            di, dj, dk = coord - np.floor(coord)
            
            # Bounds checking
            if (i < 0 or i >= resolution_map.shape[0] - 1 or
                j < 0 or j >= resolution_map.shape[1] - 1 or
                k < 0 or k >= resolution_map.shape[2] - 1):
                # Outside map bounds - use worst resolution
                atom_resolutions.append(self.resolution_range[1])
                continue
            
            # Trilinear interpolation
            c000 = resolution_map[i, j, k]
            c001 = resolution_map[i, j, k+1]
            c010 = resolution_map[i, j+1, k]
            c011 = resolution_map[i, j+1, k+1]
            c100 = resolution_map[i+1, j, k]
            c101 = resolution_map[i+1, j, k+1]
            c110 = resolution_map[i+1, j+1, k]
            c111 = resolution_map[i+1, j+1, k+1]
            
            c00 = c000 * (1 - di) + c100 * di
            c01 = c001 * (1 - di) + c101 * di
            c10 = c010 * (1 - di) + c110 * di
            c11 = c011 * (1 - di) + c111 * di
            
            c0 = c00 * (1 - dj) + c10 * dj
            c1 = c01 * (1 - dj) + c11 * dj
            
            resolution = c0 * (1 - dk) + c1 * dk
            atom_resolutions.append(resolution)
        
        return np.array(atom_resolutions)
    
    def resolution_to_confidence(self, local_resolutions: np.ndarray) -> np.ndarray:
        """
        Convert local resolution to confidence scores.
        
        Better resolution (lower value) -> higher confidence
        
        Args:
            local_resolutions: Local resolution at each atom (Å)
            
        Returns:
            Confidence scores [0, 1]
        """
        min_res, max_res = self.resolution_range
        
        # Clip to expected range
        local_resolutions = np.clip(local_resolutions, min_res, max_res)
        
        # Convert to confidence: lower resolution value = higher confidence
        confidence = 1.0 - (local_resolutions - min_res) / (max_res - min_res)
        
        return confidence


class ExperimentalMetadataExtractor:
    """
    Main interface for extracting experimental metadata from structures.
    
    Automatically detects experimental method and extracts appropriate
    uncertainty information.
    """
    
    def __init__(self):
        self.bfactor_extractor = BFactorExtractor()
        self.cryoem_extractor = CryoEMLocalResolution()
        
    def extract_from_pdb(
        self,
        pdb_file: Path,
        local_resolution_map: Optional[Path] = None,
    ) -> ExperimentalUncertainty:
        """
        Extract experimental uncertainty from PDB file.
        
        Args:
            pdb_file: Path to PDB file
            local_resolution_map: Optional path to CryoEM local resolution map
            
        Returns:
            ExperimentalUncertainty object
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython required")
        
        # Parse structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', str(pdb_file))
        
        # Get experimental method and resolution from header
        method, resolution = self._get_experimental_info(pdb_file)
        
        # Extract atom coordinates
        atom_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element != 'H':
                            atom_coords.append(atom.coord)
        atom_coords = np.array(atom_coords)
        
        # Extract confidence based on method
        if method == 'em' and local_resolution_map is not None:
            # Use CryoEM local resolution
            confidence = self._extract_cryoem_confidence(
                atom_coords, local_resolution_map
            )
            source = 'local_resolution'
        else:
            # Use B-factors (works for X-ray and CryoEM)
            chain_bfactors = self.bfactor_extractor.extract_from_structure(structure)
            all_bfactors = np.concatenate(list(chain_bfactors.values()))
            confidence = self.bfactor_extractor.bfactor_to_confidence(
                all_bfactors, resolution
            )
            source = 'bfactor'
        
        return ExperimentalUncertainty(
            atom_confidence=confidence,
            method=method,
            resolution=resolution,
            source=source,
        )
    
    def _get_experimental_info(self, pdb_file: Path) -> Tuple[str, Optional[float]]:
        """Extract experimental method and resolution from PDB header."""
        method = 'xray'  # Default
        resolution = None
        
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
        
        return method, resolution
    
    def _extract_cryoem_confidence(
        self,
        atom_coords: np.ndarray,
        local_resolution_map: Path,
    ) -> np.ndarray:
        """Extract confidence from CryoEM local resolution map."""
        # Load resolution map
        resolution_map = self.cryoem_extractor.load_local_resolution_map(
            local_resolution_map
        )
        
        # TODO: Get voxel size and origin from MRC header
        # For now, use typical values
        voxel_size = 1.0  # Angstroms
        origin = np.array([0.0, 0.0, 0.0])
        
        # Interpolate resolution at atom positions
        local_resolutions = self.cryoem_extractor.interpolate_atom_resolution(
            atom_coords, resolution_map, voxel_size, origin
        )
        
        # Convert to confidence
        confidence = self.cryoem_extractor.resolution_to_confidence(local_resolutions)
        
        return confidence

