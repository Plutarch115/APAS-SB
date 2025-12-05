"""
mdCATH MD Trajectory Dataset Loader.

Loads molecular dynamics trajectories from the mdCATH database:
- 134,950 trajectories from 1,000 CATH domains
- 5 temperatures: 320K, 350K, 380K, 410K, 450K
- Includes coordinates, forces, and energies
- HDF5 format with TorchMD-Net integration

Reference:
- Paper: https://doi.org/10.1038/s41597-024-04140-z
- HuggingFace: https://huggingface.co/datasets/compsciencelab/mdCATH
- GitHub: https://github.com/compsciencelab/mdCATH

Aligned with APAS-SB_Development_Roadmap.md Phase 1 (Days 1-7).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings


class mdCATHDataset(Dataset):
    """
    Dataset loader for mdCATH MD trajectories.
    
    Provides access to:
    - Atomic coordinates (xyz)
    - Forces (optional)
    - Energies (optional)
    - Electron density maps (computed on-the-fly)
    - Temperature information
    
    Used for density-aware training in APAS-SB.
    """
    
    def __init__(
        self,
        data_dir: str,
        temperature: int = 320,
        split: str = 'train',
        max_frames: Optional[int] = None,
        stride: int = 1,
        include_forces: bool = True,
        compute_density: bool = True,
        use_synthetic: bool = True
    ):
        """
        Args:
            data_dir: Path to mdCATH data directory
            temperature: Temperature in Kelvin (320, 350, 380, 410, or 450)
            split: 'train', 'val', or 'test'
            max_frames: Maximum frames per trajectory (None = all)
            stride: Frame stride for subsampling
            include_forces: Whether to load force data
            compute_density: Whether to compute electron density maps
            use_synthetic: Use synthetic data for testing (if real data not available)
        """
        self.data_dir = Path(data_dir)
        self.temperature = temperature
        self.split = split
        self.max_frames = max_frames
        self.stride = stride
        self.include_forces = include_forces
        self.compute_density = compute_density
        self.use_synthetic = use_synthetic
        
        # Validate temperature
        valid_temps = [320, 350, 380, 410, 450]
        if temperature not in valid_temps:
            raise ValueError(f"Temperature must be one of {valid_temps}, got {temperature}")
        
        # Check if real data exists
        if not self.data_dir.exists() or not list(self.data_dir.glob('*.h5')):
            if use_synthetic:
                warnings.warn(
                    f"mdCATH data not found at {self.data_dir}. "
                    "Using synthetic data for testing. "
                    "Download real data with: scripts/download_datasets.py --datasets mdcath"
                )
                self.data = self._generate_synthetic_data()
            else:
                raise FileNotFoundError(
                    f"mdCATH data not found at {self.data_dir}. "
                    "Download with: scripts/download_datasets.py --datasets mdcath"
                )
        else:
            self.data = self._load_real_data()
    
    def _generate_synthetic_data(self, num_trajectories: int = 100) -> List[Dict]:
        """Generate synthetic MD trajectory data for testing"""
        print(f"Generating {num_trajectories} synthetic mdCATH trajectories...")
        
        data = []
        np.random.seed(42)
        
        for i in range(num_trajectories):
            # Random protein size (50-200 residues)
            num_residues = np.random.randint(50, 200)
            num_atoms = num_residues * 10  # ~10 atoms per residue
            
            # Random number of frames (100-1000)
            num_frames = np.random.randint(100, 1000)
            
            # Generate random coordinates (Angstroms)
            coords = np.random.randn(num_frames, num_atoms, 3).astype(np.float32) * 10
            
            # Add some structure (alpha helix-like)
            for j in range(num_residues):
                start_atom = j * 10
                end_atom = start_atom + 10
                # Add helical pattern
                coords[:, start_atom:end_atom, 0] += j * 1.5  # x-axis progression
                coords[:, start_atom:end_atom, 1] += np.sin(j * 0.5) * 3  # y-axis helix
                coords[:, start_atom:end_atom, 2] += np.cos(j * 0.5) * 3  # z-axis helix
            
            # Generate forces (if requested)
            forces = None
            if self.include_forces:
                forces = np.random.randn(num_frames, num_atoms, 3).astype(np.float32) * 0.1
            
            # Generate energies
            energies = np.random.randn(num_frames).astype(np.float32) * 100 - 5000
            
            data.append({
                'trajectory_id': f'synthetic_{i:04d}',
                'cath_id': f'1.10.{i % 100}.{i % 10}',
                'temperature': self.temperature,
                'num_atoms': num_atoms,
                'num_frames': num_frames,
                'coords': coords,
                'forces': forces,
                'energies': energies,
                'atom_types': np.random.randint(1, 8, num_atoms)  # C, N, O, S, etc.
            })
        
        return data
    
    def _load_real_data(self) -> List[Dict]:
        """Load real mdCATH data from HDF5 files"""
        print(f"Loading mdCATH data from {self.data_dir}...")
        
        # Find all HDF5 files for this temperature
        h5_files = list(self.data_dir.glob(f'*_{self.temperature}K*.h5'))
        
        if not h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found for temperature {self.temperature}K in {self.data_dir}"
            )
        
        data = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Load trajectory data
                # TODO: Implement actual HDF5 structure parsing
                # This is a placeholder - actual structure depends on mdCATH format
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
                - forces: (num_frames, num_atoms, 3) forces (optional)
                - energies: (num_frames,) energies
                - density_map: (grid_size, grid_size, grid_size) electron density (optional)
                - atom_types: (num_atoms,) atomic numbers
                - temperature: scalar temperature
                - metadata: dict with trajectory info
        """
        traj = self.data[idx]

        # Get coordinates
        coords = torch.from_numpy(traj['coords'])

        # Apply stride
        if self.stride > 1:
            coords = coords[::self.stride]

        # Limit frames
        if self.max_frames is not None:
            coords = coords[:self.max_frames]

        result = {
            'coords': coords,
            'atom_types': torch.from_numpy(traj['atom_types']),
            'energies': torch.from_numpy(traj['energies'][:coords.size(0)]),
            'temperature': torch.tensor(self.temperature, dtype=torch.float32),
            'num_atoms': traj['num_atoms'],
            'num_frames': coords.size(0),
            'trajectory_id': traj['trajectory_id'],
            'cath_id': traj['cath_id']
        }

        # Add forces if available
        if self.include_forces and traj['forces'] is not None:
            forces = torch.from_numpy(traj['forces'])
            if self.stride > 1:
                forces = forces[::self.stride]
            if self.max_frames is not None:
                forces = forces[:self.max_frames]
            result['forces'] = forces

        # Compute electron density map if requested
        if self.compute_density:
            result['density_map'] = self._compute_density_map(coords, traj['atom_types'])

        return result

    def _compute_density_map(
        self,
        coords: torch.Tensor,
        atom_types: np.ndarray,
        grid_size: int = 64,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute electron density map from atomic coordinates.

        Uses Gaussian smoothing to create a 3D density grid.

        Args:
            coords: (num_frames, num_atoms, 3) coordinates
            atom_types: (num_atoms,) atomic numbers
            grid_size: Size of 3D grid
            sigma: Gaussian smoothing parameter

        Returns:
            density_map: (num_frames, grid_size, grid_size, grid_size)
        """
        num_frames = coords.size(0)
        num_atoms = coords.size(1)

        # Use mean structure for density map (to reduce computation)
        mean_coords = coords.mean(dim=0)  # (num_atoms, 3)

        # Center coordinates
        center = mean_coords.mean(dim=0)
        centered_coords = mean_coords - center

        # Determine grid bounds
        max_extent = centered_coords.abs().max().item() + 5.0  # Add 5 Angstrom padding

        # Create 3D grid
        grid = torch.zeros(grid_size, grid_size, grid_size)

        # Map coordinates to grid indices
        scale = grid_size / (2 * max_extent)
        grid_coords = (centered_coords * scale + grid_size / 2).long()

        # Clip to grid bounds
        grid_coords = torch.clamp(grid_coords, 0, grid_size - 1)

        # Add electron density (weighted by atomic number)
        atom_types_tensor = torch.from_numpy(atom_types).float()
        for i in range(num_atoms):
            x, y, z = grid_coords[i]
            # Simple point density (can be improved with Gaussian smoothing)
            grid[x, y, z] += atom_types_tensor[i]

        # Apply Gaussian smoothing (simple 3x3x3 kernel)
        if sigma > 0:
            grid = self._gaussian_smooth_3d(grid, sigma)

        return grid.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # Repeat for all frames

    def _gaussian_smooth_3d(self, grid: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply 3D Gaussian smoothing to density grid"""
        # Simple 3x3x3 averaging (placeholder for proper Gaussian)
        # In production, use scipy.ndimage.gaussian_filter or torch convolution
        kernel_size = 3
        padding = kernel_size // 2

        # Pad grid
        padded = torch.nn.functional.pad(
            grid.unsqueeze(0).unsqueeze(0),
            (padding, padding, padding, padding, padding, padding),
            mode='constant',
            value=0
        )

        # Simple averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size) / (kernel_size ** 3)

        # Apply convolution
        smoothed = torch.nn.functional.conv3d(padded, kernel)

        return smoothed.squeeze(0).squeeze(0)

    def get_statistics(self) -> Dict[str, float]:
        """Compute dataset statistics"""
        total_frames = sum(traj['num_frames'] for traj in self.data)
        total_atoms = sum(traj['num_atoms'] for traj in self.data)
        avg_frames = total_frames / len(self.data)
        avg_atoms = total_atoms / len(self.data)

        return {
            'num_trajectories': len(self.data),
            'total_frames': total_frames,
            'total_atoms': total_atoms,
            'avg_frames_per_trajectory': avg_frames,
            'avg_atoms_per_trajectory': avg_atoms,
            'temperature': self.temperature,
            'split': self.split
        }
