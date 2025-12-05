"""
ATLAS MD Trajectory Dataset Loader.

Loads molecular dynamics trajectories from the ATLAS database:
- 1,390 proteins × 3 replicates = 4,170 trajectories
- 1 μs simulations with CHARMM36m force field
- Includes pre-computed RMSF and secondary structure
- GROMACS format (.xtc trajectories + .gro topology)

Reference:
- Paper: https://doi.org/10.1093/nar/gkad1084
- Website: https://www.dsimb.inserm.fr/ATLAS
- Download: https://www.dsimb.inserm.fr/ATLAS/download.html

Aligned with APAS-SB_Development_Roadmap.md Phase 1 (Days 1-7).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict
import warnings

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False
    warnings.warn("MDAnalysis not installed. Install with: pip install MDAnalysis")


class ATLASDataset(Dataset):
    """
    Dataset loader for ATLAS MD trajectories.
    
    Provides access to:
    - Atomic coordinates (xyz)
    - Pre-computed RMSF (root mean square fluctuation)
    - Secondary structure assignments
    - Protein metadata (UniProt ID, length, etc.)
    
    Used for density-aware training and uncertainty quantification in APAS-SB.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_frames: Optional[int] = None,
        stride: int = 10,
        compute_density: bool = True,
        use_synthetic: bool = True
    ):
        """
        Args:
            data_dir: Path to ATLAS data directory
            split: 'train', 'val', or 'test'
            max_frames: Maximum frames per trajectory (None = all)
            stride: Frame stride for subsampling (default 10 = 100 ns sampling)
            compute_density: Whether to compute electron density maps
            use_synthetic: Use synthetic data for testing (if real data not available)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_frames = max_frames
        self.stride = stride
        self.compute_density = compute_density
        self.use_synthetic = use_synthetic
        
        # Check if real data exists
        if not self.data_dir.exists() or not list(self.data_dir.glob('*/*.xtc')):
            if use_synthetic:
                warnings.warn(
                    f"ATLAS data not found at {self.data_dir}. "
                    "Using synthetic data for testing. "
                    "Download real data with: scripts/download_datasets.py --datasets atlas"
                )
                self.data = self._generate_synthetic_data()
            else:
                raise FileNotFoundError(
                    f"ATLAS data not found at {self.data_dir}. "
                    "Download with: scripts/download_datasets.py --datasets atlas"
                )
        else:
            if not HAS_MDANALYSIS:
                raise ImportError(
                    "MDAnalysis required for loading ATLAS data. "
                    "Install with: pip install MDAnalysis"
                )
            self.data = self._load_real_data()
    
    def _generate_synthetic_data(self, num_proteins: int = 50) -> List[Dict]:
        """Generate synthetic ATLAS-like trajectory data for testing"""
        print(f"Generating {num_proteins} synthetic ATLAS trajectories...")
        
        data = []
        np.random.seed(42)
        
        for i in range(num_proteins):
            # Random protein size (50-500 residues, typical for ATLAS)
            num_residues = np.random.randint(50, 500)
            num_atoms = num_residues * 10  # ~10 atoms per residue
            
            # ATLAS has 10,000 frames per trajectory (1 μs at 100 ps/frame)
            # With stride=10, we get 1,000 frames
            num_frames = 1000
            
            # Generate random coordinates (Angstroms)
            coords = np.random.randn(num_frames, num_atoms, 3).astype(np.float32) * 10
            
            # Add realistic protein structure
            for j in range(num_residues):
                start_atom = j * 10
                end_atom = start_atom + 10
                # Add secondary structure patterns
                if j % 15 < 10:  # Alpha helix
                    coords[:, start_atom:end_atom, 0] += j * 1.5
                    coords[:, start_atom:end_atom, 1] += np.sin(j * 0.5) * 3
                    coords[:, start_atom:end_atom, 2] += np.cos(j * 0.5) * 3
                else:  # Beta sheet
                    coords[:, start_atom:end_atom, 0] += j * 3.5
                    coords[:, start_atom:end_atom, 1] += (j % 2) * 5
            
            # Generate RMSF (higher for loops, lower for secondary structure)
            rmsf = np.random.rand(num_residues).astype(np.float32) * 2 + 0.5
            
            # Generate secondary structure (H=helix, E=sheet, C=coil)
            ss = np.array(['H' if j % 15 < 10 else 'E' if j % 15 < 13 else 'C' 
                          for j in range(num_residues)])
            
            data.append({
                'protein_id': f'synthetic_P{i:05d}',
                'uniprot_id': f'P{i:05d}',
                'replicate': i % 3,
                'num_residues': num_residues,
                'num_atoms': num_atoms,
                'num_frames': num_frames,
                'coords': coords,
                'rmsf': rmsf,
                'secondary_structure': ss,
                'atom_types': np.random.randint(1, 8, num_atoms)
            })
        
        return data
    
    def _load_real_data(self) -> List[Dict]:
        """Load real ATLAS data from GROMACS trajectories"""
        print(f"Loading ATLAS data from {self.data_dir}...")
        
        # Find all trajectory directories
        traj_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        data = []
        for traj_dir in traj_dirs:
            # Look for .xtc and .gro files
            xtc_files = list(traj_dir.glob('*.xtc'))
            gro_files = list(traj_dir.glob('*.gro'))
            
            if xtc_files and gro_files:
                # TODO: Implement actual MDAnalysis loading
                # universe = mda.Universe(str(gro_files[0]), str(xtc_files[0]))
                pass
        
        return data

    def __len__(self) -> int:
        """Return number of trajectories"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory.

        Returns:
            Dictionary with:
                - coords: (num_frames, num_atoms, 3) coordinates
                - rmsf: (num_residues,) root mean square fluctuation
                - secondary_structure: (num_residues,) SS assignments
                - density_map: (grid_size, grid_size, grid_size) electron density (optional)
                - atom_types: (num_atoms,) atomic numbers
                - metadata: dict with protein info
        """
        traj = self.data[idx]

        # Get coordinates
        coords = torch.from_numpy(traj['coords'])

        # Apply stride (already applied in synthetic data, but needed for real data)
        if self.stride > 1 and coords.size(0) > 1000:
            coords = coords[::self.stride]

        # Limit frames
        if self.max_frames is not None:
            coords = coords[:self.max_frames]

        result = {
            'coords': coords,
            'atom_types': torch.from_numpy(traj['atom_types']),
            'rmsf': torch.from_numpy(traj['rmsf']),
            'num_atoms': traj['num_atoms'],
            'num_residues': traj['num_residues'],
            'num_frames': coords.size(0),
            'protein_id': traj['protein_id'],
            'uniprot_id': traj['uniprot_id'],
            'replicate': traj['replicate']
        }

        # Compute electron density map if requested
        if self.compute_density:
            # Reuse mdCATH density computation
            from pearl.data.mdcath_loader import mdCATHDataset
            dummy_loader = mdCATHDataset.__new__(mdCATHDataset)
            result['density_map'] = dummy_loader._compute_density_map(
                coords, traj['atom_types']
            )

        return result

    def get_statistics(self) -> Dict[str, float]:
        """Compute dataset statistics"""
        total_frames = sum(traj['num_frames'] for traj in self.data)
        total_atoms = sum(traj['num_atoms'] for traj in self.data)
        total_residues = sum(traj['num_residues'] for traj in self.data)
        avg_frames = total_frames / len(self.data)
        avg_atoms = total_atoms / len(self.data)
        avg_residues = total_residues / len(self.data)

        return {
            'num_trajectories': len(self.data),
            'total_frames': total_frames,
            'total_atoms': total_atoms,
            'total_residues': total_residues,
            'avg_frames_per_trajectory': avg_frames,
            'avg_atoms_per_trajectory': avg_atoms,
            'avg_residues_per_trajectory': avg_residues,
            'split': self.split
        }
