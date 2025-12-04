"""
Density Map Loader for Pearl

Load and process experimental density maps:
- X-ray electron density maps (CCP4/MRC format)
- CryoEM density maps (MRC/MAP format)
- Structure factors (MTZ/CIF format)

This enables density-aware training where the model learns to place atoms
to match experimental density, not just fitted coordinates.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    MRCFILE_AVAILABLE = False
    warnings.warn("mrcfile not available. Install with: pip install mrcfile")

try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False
    warnings.warn("gemmi not available. Install with: pip install gemmi")


class DensityMap:
    """
    Container for experimental density map.
    
    Stores:
    - 3D density grid
    - Voxel size (Å/voxel)
    - Origin (Å)
    - Unit cell parameters
    - Resolution
    """
    
    def __init__(
        self,
        data: np.ndarray,           # [nx, ny, nz] density values
        voxel_size: np.ndarray,     # [3] Å per voxel
        origin: np.ndarray,         # [3] origin in Å
        resolution: float,          # Overall resolution (Å)
        map_type: str = 'cryoem',   # 'cryoem', 'xray_2fofc', 'xray_fofc'
    ):
        self.data = data
        self.voxel_size = voxel_size
        self.origin = origin
        self.resolution = resolution
        self.map_type = map_type
        self.shape = data.shape
    
    def get_value_at_position(self, position: np.ndarray) -> float:
        """
        Get density value at a specific position using trilinear interpolation.
        
        Args:
            position: [3] position in Angstroms
            
        Returns:
            Interpolated density value
        """
        # Convert position to voxel coordinates
        voxel_coord = (position - self.origin) / self.voxel_size
        
        # Get integer and fractional parts
        i, j, k = np.floor(voxel_coord).astype(int)
        di, dj, dk = voxel_coord - np.floor(voxel_coord)
        
        # Bounds checking
        if (i < 0 or i >= self.shape[0] - 1 or
            j < 0 or j >= self.shape[1] - 1 or
            k < 0 or k >= self.shape[2] - 1):
            return 0.0
        
        # Trilinear interpolation
        c000 = self.data[i, j, k]
        c001 = self.data[i, j, k+1]
        c010 = self.data[i, j+1, k]
        c011 = self.data[i, j+1, k+1]
        c100 = self.data[i+1, j, k]
        c101 = self.data[i+1, j, k+1]
        c110 = self.data[i+1, j+1, k]
        c111 = self.data[i+1, j+1, k+1]
        
        c00 = c000 * (1 - di) + c100 * di
        c01 = c001 * (1 - di) + c101 * di
        c10 = c010 * (1 - di) + c110 * di
        c11 = c011 * (1 - di) + c111 * di
        
        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj
        
        value = c0 * (1 - dk) + c1 * dk
        
        return float(value)
    
    def get_values_at_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Get density values at multiple positions.
        
        Args:
            positions: [n, 3] positions in Angstroms
            
        Returns:
            [n] density values
        """
        return np.array([self.get_value_at_position(pos) for pos in positions])
    
    def to_torch(self, device='cpu') -> torch.Tensor:
        """Convert density map to PyTorch tensor."""
        return torch.from_numpy(self.data).float().to(device)


class CryoEMDensityLoader:
    """
    Load CryoEM density maps from EMDB.
    
    Supports:
    - Primary maps (experimental density)
    - Half-maps (for FSC calculation)
    - Local resolution maps
    """
    
    def __init__(self):
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile required. Install with: pip install mrcfile")
    
    def load_map(self, mrc_file: Path) -> DensityMap:
        """
        Load CryoEM density map from MRC file.
        
        Args:
            mrc_file: Path to MRC/MAP file
            
        Returns:
            DensityMap object
        """
        with mrcfile.open(mrc_file, mode='r') as mrc:
            # Get density data
            data = mrc.data.copy()
            
            # Get voxel size from header
            voxel_size = np.array([
                mrc.voxel_size.x,
                mrc.voxel_size.y,
                mrc.voxel_size.z
            ])
            
            # Get origin
            # MRC origin is in voxels, convert to Angstroms
            origin = np.array([
                mrc.header.origin.x,
                mrc.header.origin.y,
                mrc.header.origin.z
            ])
            
            # Try to get resolution from header (not always present)
            resolution = getattr(mrc.header, 'resolution', None)
            if resolution is None or resolution == 0:
                # Estimate from voxel size (Nyquist: resolution = 2 * voxel_size)
                resolution = 2.0 * float(voxel_size.mean())
        
        return DensityMap(
            data=data,
            voxel_size=voxel_size,
            origin=origin,
            resolution=resolution,
            map_type='cryoem'
        )
    
    def load_half_maps(self, half1_file: Path, half2_file: Path) -> Tuple[DensityMap, DensityMap]:
        """
        Load CryoEM half-maps for FSC calculation.
        
        Args:
            half1_file: Path to first half-map
            half2_file: Path to second half-map
            
        Returns:
            Tuple of (half1_map, half2_map)
        """
        half1 = self.load_map(half1_file)
        half2 = self.load_map(half2_file)
        
        return half1, half2


class XrayDensityLoader:
    """
    Load X-ray electron density maps.
    
    Supports:
    - 2Fo-Fc maps (standard electron density)
    - Fo-Fc maps (difference density)
    - Structure factors (for on-the-fly density generation)
    """
    
    def __init__(self):
        if not GEMMI_AVAILABLE:
            warnings.warn("gemmi not available. Install with: pip install gemmi")
    
    def load_map(self, map_file: Path) -> DensityMap:
        """
        Load X-ray density map from CCP4/MRC file.
        
        Args:
            map_file: Path to CCP4/MRC file
            
        Returns:
            DensityMap object
        """
        if not GEMMI_AVAILABLE:
            raise ImportError("gemmi required for X-ray maps. Install with: pip install gemmi")
        
        # Load with gemmi
        ccp4 = gemmi.read_ccp4_map(str(map_file))
        grid = ccp4.grid
        
        # Convert to numpy array
        data = np.array(grid, copy=True)
        
        # Get voxel size
        unit_cell = ccp4.grid.unit_cell
        voxel_size = np.array([
            unit_cell.a / grid.nu,
            unit_cell.b / grid.nv,
            unit_cell.c / grid.nw
        ])
        
        # Get origin (usually 0,0,0 for X-ray maps)
        origin = np.array([0.0, 0.0, 0.0])
        
        # Resolution (would need to get from MTZ file or metadata)
        resolution = 2.0  # Default, should be provided separately
        
        return DensityMap(
            data=data,
            voxel_size=voxel_size,
            origin=origin,
            resolution=resolution,
            map_type='xray_2fofc'
        )
    
    def load_structure_factors(self, mtz_file: Path) -> Dict:
        """
        Load structure factors from MTZ file.
        
        Args:
            mtz_file: Path to MTZ file
            
        Returns:
            Dictionary with structure factors
        """
        if not GEMMI_AVAILABLE:
            raise ImportError("gemmi required. Install with: pip install gemmi")
        
        mtz = gemmi.read_mtz_file(str(mtz_file))
        
        # Extract F_obs and phases
        # Column names vary, common ones: FP, SIGFP, PHIB, FOM
        structure_factors = {
            'h': [],
            'k': [],
            'l': [],
            'F_obs': [],
            'phase': [],
        }
        
        # This is simplified - actual implementation would need to handle
        # different column naming conventions
        for reflection in mtz:
            structure_factors['h'].append(reflection[0])
            structure_factors['k'].append(reflection[1])
            structure_factors['l'].append(reflection[2])
            # ... extract F and phase
        
        return structure_factors


class DensityMapDataset:
    """
    Dataset that loads both coordinates and density maps.
    
    For each structure:
    - Load PDB coordinates
    - Load experimental density map
    - Load local resolution map (if available)
    """
    
    def __init__(
        self,
        pdb_dir: Path,
        density_dir: Path,
        map_type: str = 'cryoem',  # 'cryoem' or 'xray'
        max_structures: Optional[int] = None,
    ):
        self.pdb_dir = Path(pdb_dir)
        self.density_dir = Path(density_dir)
        self.map_type = map_type
        
        # Initialize loaders
        if map_type == 'cryoem':
            self.density_loader = CryoEMDensityLoader()
        else:
            self.density_loader = XrayDensityLoader()
        
        # Find all structures with both PDB and density
        self.structure_ids = self._find_structures()
        
        if max_structures is not None:
            self.structure_ids = self.structure_ids[:max_structures]
    
    def _find_structures(self) -> List[str]:
        """Find all structures with both PDB and density files."""
        structure_ids = []
        
        for pdb_file in self.pdb_dir.glob("*.pdb"):
            pdb_id = pdb_file.stem
            
            # Check for corresponding density file
            if self.map_type == 'cryoem':
                density_file = self.density_dir / f"{pdb_id}.mrc"
            else:
                density_file = self.density_dir / f"{pdb_id}.ccp4"
            
            if density_file.exists():
                structure_ids.append(pdb_id)
        
        return structure_ids
    
    def __len__(self) -> int:
        return len(self.structure_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load structure with density map.
        
        Returns:
            Dictionary with:
            - pdb_id: Structure identifier
            - coords: Atomic coordinates
            - atom_types: Atom types
            - density_map: Experimental density map
            - local_resolution: Local resolution map (if available)
        """
        pdb_id = self.structure_ids[idx]
        
        # Load PDB (simplified - would use proper PDB loader)
        pdb_file = self.pdb_dir / f"{pdb_id}.pdb"
        # coords, atom_types = load_pdb(pdb_file)
        
        # Load density map
        if self.map_type == 'cryoem':
            density_file = self.density_dir / f"{pdb_id}.mrc"
        else:
            density_file = self.density_dir / f"{pdb_id}.ccp4"
        
        density_map = self.density_loader.load_map(density_file)
        
        # Load local resolution (if available)
        local_res_file = self.density_dir / f"{pdb_id}_local_res.mrc"
        local_resolution = None
        if local_res_file.exists() and self.map_type == 'cryoem':
            local_resolution = self.density_loader.load_map(local_res_file)
        
        return {
            'pdb_id': pdb_id,
            # 'coords': coords,
            # 'atom_types': atom_types,
            'density_map': density_map,
            'local_resolution': local_resolution,
        }

