"""
Molecular Dynamics Integration for Pearl Training

This module provides tools to:
1. Run MD simulations on protein-ligand complexes
2. Extract conformational ensembles and uncertainty
3. Compute dynamic B-factors (RMSF) from trajectories
4. Combine experimental and MD-derived confidence scores
5. Generate training data from MD trajectories

Author: Pearl Team
Date: 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle


class MDTrajectoryProcessor:
    """
    Process MD trajectories to extract training data with uncertainty.
    """
    
    def __init__(
        self,
        trajectory_dir: Path,
        output_dir: Path,
        sampling_method: str = 'clustered',
        n_samples: int = 1000,
    ):
        """
        Args:
            trajectory_dir: Directory containing MD trajectories
            output_dir: Directory to save processed data
            sampling_method: 'uniform', 'clustered', or 'ensemble'
            n_samples: Number of frames to sample per trajectory
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.output_dir = Path(output_dir)
        self.sampling_method = sampling_method
        self.n_samples = n_samples
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_trajectory(self, trajectory_file: Path) -> Dict:
        """
        Load MD trajectory from file.
        
        Supports multiple formats:
        - DCD (CHARMM/NAMD)
        - XTC (GROMACS)
        - NetCDF (AMBER)
        - HDF5 (custom)
        
        Args:
            trajectory_file: Path to trajectory file
            
        Returns:
            Dictionary with trajectory data
        """
        # Try to import MDAnalysis or MDTraj
        try:
            import MDAnalysis as mda
            
            # Load trajectory
            u = mda.Universe(
                str(trajectory_file.with_suffix('.pdb')),  # Topology
                str(trajectory_file),  # Trajectory
            )
            
            # Extract coordinates for all frames
            n_frames = len(u.trajectory)
            n_atoms = len(u.atoms)
            
            coords = np.zeros((n_frames, n_atoms, 3))
            for i, ts in enumerate(u.trajectory):
                coords[i] = u.atoms.positions
            
            return {
                'coords': coords,  # [n_frames, n_atoms, 3]
                'n_frames': n_frames,
                'n_atoms': n_atoms,
                'atom_names': [atom.name for atom in u.atoms],
                'residue_names': [atom.resname for atom in u.atoms],
            }
            
        except ImportError:
            print("⚠ MDAnalysis not available. Install with: pip install MDAnalysis")
            print("  Falling back to simple numpy loader...")
            
            # Fallback: load from numpy file
            data = np.load(trajectory_file)
            return {
                'coords': data['coords'],
                'n_frames': data['coords'].shape[0],
                'n_atoms': data['coords'].shape[1],
            }
    
    def compute_rmsf(
        self,
        coords: np.ndarray,
        align: bool = True,
    ) -> np.ndarray:
        """
        Compute Root Mean Square Fluctuation (RMSF) for each atom.
        
        RMSF measures the average deviation of each atom from its mean position.
        
        Args:
            coords: Trajectory coordinates [n_frames, n_atoms, 3]
            align: Whether to align frames before computing RMSF
            
        Returns:
            RMSF values [n_atoms]
        """
        n_frames, n_atoms, _ = coords.shape
        
        if align:
            # Align all frames to first frame (remove rotation/translation)
            coords_aligned = self._align_trajectory(coords)
        else:
            coords_aligned = coords
        
        # Compute mean position
        mean_coords = coords_aligned.mean(axis=0)  # [n_atoms, 3]
        
        # Compute squared deviations
        deviations = coords_aligned - mean_coords[np.newaxis, :, :]  # [n_frames, n_atoms, 3]
        squared_deviations = (deviations ** 2).sum(axis=2)  # [n_frames, n_atoms]
        
        # Compute RMSF
        rmsf = np.sqrt(squared_deviations.mean(axis=0))  # [n_atoms]
        
        return rmsf
    
    def _align_trajectory(self, coords: np.ndarray) -> np.ndarray:
        """
        Align all frames to the first frame using Kabsch algorithm.
        
        Args:
            coords: Trajectory coordinates [n_frames, n_atoms, 3]
            
        Returns:
            Aligned coordinates [n_frames, n_atoms, 3]
        """
        n_frames, n_atoms, _ = coords.shape
        
        # Reference frame (first frame)
        ref = coords[0]
        ref_center = ref.mean(axis=0)
        ref_centered = ref - ref_center
        
        aligned = np.zeros_like(coords)
        aligned[0] = coords[0]
        
        for i in range(1, n_frames):
            # Center current frame
            frame = coords[i]
            frame_center = frame.mean(axis=0)
            frame_centered = frame - frame_center
            
            # Compute rotation matrix using Kabsch algorithm
            H = frame_centered.T @ ref_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation and translation
            aligned[i] = (frame_centered @ R) + ref_center
        
        return aligned
    
    def rmsf_to_confidence(
        self,
        rmsf: np.ndarray,
        method: str = 'exponential',
    ) -> np.ndarray:
        """
        Convert RMSF to confidence scores [0, 1].
        
        Lower RMSF → higher confidence (more stable)
        Higher RMSF → lower confidence (more flexible)
        
        Args:
            rmsf: RMSF values [n_atoms]
            method: Conversion method ('exponential', 'linear', 'sigmoid')
            
        Returns:
            Confidence scores [n_atoms]
        """
        if method == 'exponential':
            # Exponential decay: conf = exp(-RMSF / scale)
            scale = 2.0  # Å
            confidence = np.exp(-rmsf / scale)
            
        elif method == 'linear':
            # Linear mapping: conf = 1 - (RMSF - min) / (max - min)
            rmsf_min = np.percentile(rmsf, 5)
            rmsf_max = np.percentile(rmsf, 95)
            confidence = 1.0 - (rmsf - rmsf_min) / (rmsf_max - rmsf_min + 1e-8)
            confidence = np.clip(confidence, 0, 1)
            
        elif method == 'sigmoid':
            # Sigmoid: conf = 1 / (1 + exp((RMSF - midpoint) / scale))
            midpoint = np.median(rmsf)
            scale = 1.0
            confidence = 1.0 / (1.0 + np.exp((rmsf - midpoint) / scale))
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return confidence
    
    def cluster_trajectory(
        self,
        coords: np.ndarray,
        n_clusters: int = 10,
        method: str = 'kmeans',
    ) -> List[Dict]:
        """
        Cluster trajectory into conformational states.
        
        Args:
            coords: Trajectory coordinates [n_frames, n_atoms, 3]
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical')
            
        Returns:
            List of cluster dictionaries
        """
        from sklearn.cluster import KMeans
        
        n_frames, n_atoms, _ = coords.shape
        
        # Flatten coordinates for clustering
        coords_flat = coords.reshape(n_frames, -1)  # [n_frames, n_atoms*3]
        
        # Perform clustering
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(coords_flat)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Group frames by cluster
        clusters = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_frames = coords[mask]
            
            clusters.append({
                'cluster_id': cluster_id,
                'frames': cluster_frames,
                'n_frames': cluster_frames.shape[0],
                'population': cluster_frames.shape[0] / n_frames,
                'mean_coords': cluster_frames.mean(axis=0),
                'rmsf': self.compute_rmsf(cluster_frames),
            })
        
        return clusters
    
    def sample_trajectory_uniform(
        self,
        coords: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Sample frames uniformly from trajectory.
        
        Args:
            coords: Trajectory coordinates [n_frames, n_atoms, 3]
            n_samples: Number of frames to sample
            
        Returns:
            Sampled coordinates [n_samples, n_atoms, 3]
        """
        n_frames = coords.shape[0]
        indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)
        return coords[indices]
    
    def sample_trajectory_clustered(
        self,
        coords: np.ndarray,
        n_samples: int,
        n_clusters: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample frames using clustering (representative structures).
        
        Args:
            coords: Trajectory coordinates [n_frames, n_atoms, 3]
            n_samples: Number of frames to sample
            n_clusters: Number of clusters to use
            
        Returns:
            Tuple of (sampled_coords, confidence_scores)
        """
        # Cluster trajectory
        clusters = self.cluster_trajectory(coords, n_clusters=n_clusters)
        
        # Sample from each cluster proportionally to population
        sampled_coords = []
        sampled_confidence = []
        
        for cluster in clusters:
            # Number of samples from this cluster
            n_cluster_samples = max(1, int(n_samples * cluster['population']))
            
            # Sample frames from cluster
            cluster_frames = cluster['frames']
            if len(cluster_frames) >= n_cluster_samples:
                indices = np.random.choice(
                    len(cluster_frames),
                    size=n_cluster_samples,
                    replace=False,
                )
                samples = cluster_frames[indices]
            else:
                samples = cluster_frames
            
            # Compute confidence for each sample
            rmsf = cluster['rmsf']
            confidence = self.rmsf_to_confidence(rmsf)
            
            sampled_coords.append(samples)
            sampled_confidence.extend([confidence] * len(samples))
        
        sampled_coords = np.concatenate(sampled_coords, axis=0)
        sampled_confidence = np.array(sampled_confidence)
        
        # Trim to exact n_samples
        if len(sampled_coords) > n_samples:
            sampled_coords = sampled_coords[:n_samples]
            sampled_confidence = sampled_confidence[:n_samples]
        
        return sampled_coords, sampled_confidence
    
    def process_trajectory(
        self,
        pdb_id: str,
        trajectory_file: Path,
        experimental_structure: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Process a single MD trajectory to generate training data.
        
        Args:
            pdb_id: PDB ID
            trajectory_file: Path to trajectory file
            experimental_structure: Optional experimental structure for comparison
            
        Returns:
            List of training examples
        """
        print(f"Processing trajectory for {pdb_id}...")
        
        # Load trajectory
        trajectory = self.load_trajectory(trajectory_file)
        coords = trajectory['coords']
        
        print(f"  Loaded {trajectory['n_frames']} frames, {trajectory['n_atoms']} atoms")
        
        # Sample frames
        if self.sampling_method == 'uniform':
            sampled_coords = self.sample_trajectory_uniform(coords, self.n_samples)
            
            # Compute global RMSF
            rmsf = self.compute_rmsf(coords)
            confidence = self.rmsf_to_confidence(rmsf)
            
            # Create training examples
            training_examples = []
            for i, frame_coords in enumerate(sampled_coords):
                training_examples.append({
                    'pdb_id': pdb_id,
                    'source': 'md_uniform',
                    'frame_id': i,
                    'coords': frame_coords,
                    'confidence': confidence,
                    'weight': 0.5,  # Lower weight than experimental
                })
        
        elif self.sampling_method == 'clustered':
            sampled_coords, sampled_confidence = self.sample_trajectory_clustered(
                coords,
                self.n_samples,
            )
            
            # Create training examples
            training_examples = []
            for i, (frame_coords, frame_confidence) in enumerate(
                zip(sampled_coords, sampled_confidence)
            ):
                training_examples.append({
                    'pdb_id': pdb_id,
                    'source': 'md_clustered',
                    'frame_id': i,
                    'coords': frame_coords,
                    'confidence': frame_confidence,
                    'weight': 0.7,  # Higher weight for representative structures
                })
        
        elif self.sampling_method == 'ensemble':
            # Cluster trajectory
            clusters = self.cluster_trajectory(coords, n_clusters=10)
            
            # Create training examples from cluster centroids
            training_examples = []
            for cluster in clusters:
                confidence = self.rmsf_to_confidence(cluster['rmsf'])
                
                training_examples.append({
                    'pdb_id': pdb_id,
                    'source': 'md_ensemble',
                    'cluster_id': cluster['cluster_id'],
                    'coords': cluster['mean_coords'],
                    'confidence': confidence,
                    'weight': cluster['population'],  # Weight by population
                    'n_frames': cluster['n_frames'],
                })
        
        print(f"  Generated {len(training_examples)} training examples")
        
        return training_examples


class HybridConfidenceEstimator:
    """
    Combine experimental and MD-derived confidence scores.
    """
    
    def __init__(
        self,
        exp_weight: float = 0.7,
        md_weight: float = 0.3,
        method: str = 'weighted_average',
    ):
        """
        Args:
            exp_weight: Weight for experimental confidence
            md_weight: Weight for MD confidence
            method: Combination method ('weighted_average', 'min', 'max', 'product')
        """
        self.exp_weight = exp_weight
        self.md_weight = md_weight
        self.method = method
        
        # Normalize weights
        total = exp_weight + md_weight
        self.exp_weight /= total
        self.md_weight /= total
    
    def combine_confidence(
        self,
        exp_confidence: np.ndarray,
        md_confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Combine experimental and MD confidence scores.
        
        Args:
            exp_confidence: Experimental confidence [n_atoms]
            md_confidence: MD confidence [n_atoms]
            
        Returns:
            Combined confidence [n_atoms]
        """
        if self.method == 'weighted_average':
            combined = (
                self.exp_weight * exp_confidence +
                self.md_weight * md_confidence
            )
        
        elif self.method == 'min':
            # Conservative: take minimum (most uncertain)
            combined = np.minimum(exp_confidence, md_confidence)
        
        elif self.method == 'max':
            # Optimistic: take maximum (most certain)
            combined = np.maximum(exp_confidence, md_confidence)
        
        elif self.method == 'product':
            # Multiplicative: both must be confident
            combined = exp_confidence * md_confidence
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return combined


# Example usage
if __name__ == "__main__":
    # Process MD trajectories
    processor = MDTrajectoryProcessor(
        trajectory_dir=Path("data/md_trajectories"),
        output_dir=Path("data/md_processed"),
        sampling_method='clustered',
        n_samples=1000,
    )
    
    # Process a single trajectory
    training_examples = processor.process_trajectory(
        pdb_id="1ATP",
        trajectory_file=Path("data/md_trajectories/1ATP_trajectory.dcd"),
    )
    
    print(f"Generated {len(training_examples)} training examples")
    print(f"Example: {training_examples[0]}")

