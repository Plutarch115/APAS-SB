"""
MD Trajectory Processing for Pearl Training

This module processes MD trajectories to extract:
- Ensemble-averaged structures
- Per-atom uncertainty (RMSF)
- Confidence scores for training
- Conformational clustering

Integrates with md_simulation.py and md_integration.py
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import mdtraj as md
except ImportError:
    raise ImportError("MDTraj not installed. Install with: conda install -c conda-forge mdtraj")

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryProcessingConfig:
    """Configuration for trajectory processing"""
    
    # Alignment
    align_selection: str = "protein and name CA"  # Selection for alignment
    
    # RMSF calculation
    rmsf_selection: str = "all"  # Selection for RMSF
    
    # Clustering
    n_clusters: int = 5  # Number of conformational clusters
    clustering_selection: str = "protein and name CA"  # Selection for clustering
    clustering_method: str = "average"  # Linkage method
    
    # Ensemble averaging
    skip_frames: int = 10  # Skip first N frames (equilibration)
    stride: int = 1  # Use every Nth frame
    
    # Confidence mapping
    rmsf_to_confidence_scale: float = 10.0  # Å (RMSF at which confidence = 0.5)
    
    # Output
    save_representative_structures: bool = True
    save_ensemble_average: bool = True


class MDTrajectoryProcessor:
    """
    Process MD trajectories to extract uncertainty information for Pearl training.
    
    Features:
    - Kabsch alignment
    - RMSF calculation
    - Conformational clustering
    - Ensemble averaging
    - Confidence score generation
    """
    
    def __init__(self, config: Optional[TrajectoryProcessingConfig] = None):
        """
        Initialize trajectory processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or TrajectoryProcessingConfig()
        self.trajectory = None
        self.topology = None
        
    def load_trajectory(
        self,
        trajectory_file: str,
        topology_file: str,
        stride: Optional[int] = None
    ) -> md.Trajectory:
        """
        Load MD trajectory.
        
        Args:
            trajectory_file: Path to trajectory (DCD, XTC, etc.)
            topology_file: Path to topology (PDB, PSF, etc.)
            stride: Load every Nth frame
            
        Returns:
            MDTraj trajectory object
        """
        logger.info(f"Loading trajectory from {trajectory_file}...")
        
        stride = stride or self.config.stride
        
        self.trajectory = md.load(
            trajectory_file,
            top=topology_file,
            stride=stride
        )
        
        self.topology = self.trajectory.topology
        
        logger.info(f"Loaded {self.trajectory.n_frames} frames, "
                   f"{self.trajectory.n_atoms} atoms")
        
        return self.trajectory
    
    def align_trajectory(
        self,
        reference_frame: int = 0,
        selection: Optional[str] = None
    ) -> md.Trajectory:
        """
        Align trajectory using Kabsch algorithm.
        
        Args:
            reference_frame: Frame to use as reference
            selection: Atom selection for alignment
            
        Returns:
            Aligned trajectory
        """
        logger.info("Aligning trajectory...")
        
        selection = selection or self.config.align_selection
        
        # Get atom indices for alignment
        atom_indices = self.topology.select(selection)
        
        if len(atom_indices) == 0:
            logger.warning(f"No atoms selected with '{selection}', using all atoms")
            atom_indices = None
        
        # Align
        self.trajectory.superpose(
            self.trajectory,
            frame=reference_frame,
            atom_indices=atom_indices
        )
        
        logger.info(f"Aligned to frame {reference_frame} using {len(atom_indices) if atom_indices is not None else self.trajectory.n_atoms} atoms")
        
        return self.trajectory
    
    def compute_rmsf(
        self,
        selection: Optional[str] = None,
        skip_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute root mean square fluctuation (RMSF) for each atom.
        
        Args:
            selection: Atom selection for RMSF
            skip_frames: Skip first N frames
            
        Returns:
            RMSF values (Å) for each atom
        """
        logger.info("Computing RMSF...")
        
        selection = selection or self.config.rmsf_selection
        skip_frames = skip_frames or self.config.skip_frames
        
        # Skip equilibration frames
        traj = self.trajectory[skip_frames:]
        
        # Get atom indices
        if selection == "all":
            atom_indices = np.arange(traj.n_atoms)
        else:
            atom_indices = self.topology.select(selection)
        
        # Compute mean positions
        mean_positions = traj.xyz[:, atom_indices, :].mean(axis=0)
        
        # Compute RMSF
        deviations = traj.xyz[:, atom_indices, :] - mean_positions
        rmsf = np.sqrt((deviations ** 2).mean(axis=0).sum(axis=1))
        
        # Convert to Å
        rmsf *= 10.0  # nm to Å
        
        logger.info(f"RMSF computed for {len(atom_indices)} atoms")
        logger.info(f"RMSF range: {rmsf.min():.2f} - {rmsf.max():.2f} Å")
        logger.info(f"RMSF mean: {rmsf.mean():.2f} Å")
        
        # Create full RMSF array (all atoms)
        full_rmsf = np.zeros(traj.n_atoms)
        full_rmsf[atom_indices] = rmsf
        
        return full_rmsf
    
    def rmsf_to_confidence(
        self,
        rmsf: np.ndarray,
        scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert RMSF to confidence scores.
        
        Uses sigmoid mapping: confidence = 1 / (1 + (RMSF / scale))
        
        Args:
            rmsf: RMSF values (Å)
            scale: RMSF value at which confidence = 0.5
            
        Returns:
            Confidence scores [0, 1]
        """
        scale = scale or self.config.rmsf_to_confidence_scale
        
        # Sigmoid mapping
        confidence = 1.0 / (1.0 + (rmsf / scale))
        
        logger.info(f"Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
        logger.info(f"Confidence mean: {confidence.mean():.3f}")
        
        return confidence
    
    def cluster_trajectory(
        self,
        n_clusters: Optional[int] = None,
        selection: Optional[str] = None,
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Cluster trajectory into conformational states.
        
        Args:
            n_clusters: Number of clusters
            selection: Atom selection for clustering
            method: Linkage method (single, complete, average, ward)
            
        Returns:
            Tuple of (cluster_labels, representative_frame_indices)
        """
        logger.info("Clustering trajectory...")
        
        n_clusters = n_clusters or self.config.n_clusters
        selection = selection or self.config.clustering_selection
        method = method or self.config.clustering_method
        
        # Get atom indices
        atom_indices = self.topology.select(selection)
        
        # Compute pairwise RMSD
        logger.info("Computing pairwise RMSD...")
        rmsd_matrix = np.zeros((self.trajectory.n_frames, self.trajectory.n_frames))
        
        for i in range(self.trajectory.n_frames):
            rmsd_matrix[i, :] = md.rmsd(
                self.trajectory,
                self.trajectory,
                frame=i,
                atom_indices=atom_indices
            )
        
        # Convert to distance matrix
        distance_matrix = squareform(rmsd_matrix)
        
        # Hierarchical clustering
        logger.info(f"Performing hierarchical clustering ({method})...")
        linkage_matrix = linkage(distance_matrix, method=method)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Find representative structures (medoids)
        representative_frames = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_frames = np.where(cluster_labels == cluster_id)[0]
            
            # Find frame with minimum average RMSD to other frames in cluster
            cluster_rmsd = rmsd_matrix[np.ix_(cluster_frames, cluster_frames)]
            medoid_idx = cluster_frames[cluster_rmsd.mean(axis=1).argmin()]
            representative_frames.append(medoid_idx)
        
        logger.info(f"Clustered into {n_clusters} states")
        for i, (cluster_id, frame_idx) in enumerate(zip(range(1, n_clusters + 1), representative_frames)):
            n_frames = (cluster_labels == cluster_id).sum()
            logger.info(f"  Cluster {cluster_id}: {n_frames} frames, representative frame {frame_idx}")
        
        return cluster_labels, representative_frames
    
    def compute_ensemble_average(
        self,
        skip_frames: Optional[int] = None
    ) -> md.Trajectory:
        """
        Compute ensemble-averaged structure.
        
        Args:
            skip_frames: Skip first N frames
            
        Returns:
            Single-frame trajectory with averaged coordinates
        """
        logger.info("Computing ensemble average...")
        
        skip_frames = skip_frames or self.config.skip_frames
        
        # Skip equilibration frames
        traj = self.trajectory[skip_frames:]
        
        # Compute mean positions
        mean_positions = traj.xyz.mean(axis=0, keepdims=True)
        
        # Create new trajectory with mean positions
        ensemble_avg = md.Trajectory(
            mean_positions,
            topology=self.topology
        )
        
        logger.info("Ensemble average computed")

        return ensemble_avg

    def process_trajectory_for_pearl(
        self,
        trajectory_file: str,
        topology_file: str,
        output_dir: str,
        stride: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Complete trajectory processing workflow for Pearl training.

        Args:
            trajectory_file: Path to trajectory file
            topology_file: Path to topology file
            output_dir: Output directory
            stride: Load every Nth frame

        Returns:
            Dictionary with processed data
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load trajectory
        self.load_trajectory(trajectory_file, topology_file, stride)

        # Align trajectory
        self.align_trajectory()

        # Compute RMSF
        rmsf = self.compute_rmsf()

        # Convert to confidence
        confidence = self.rmsf_to_confidence(rmsf)

        # Cluster trajectory
        cluster_labels, representative_frames = self.cluster_trajectory()

        # Compute ensemble average
        ensemble_avg = self.compute_ensemble_average()

        # Save outputs
        results = {
            "rmsf": rmsf,
            "confidence": confidence,
            "cluster_labels": cluster_labels,
            "representative_frames": representative_frames,
            "n_frames": self.trajectory.n_frames,
            "n_atoms": self.trajectory.n_atoms
        }

        # Save ensemble average structure
        if self.config.save_ensemble_average:
            ensemble_avg_path = os.path.join(output_dir, "ensemble_average.pdb")
            ensemble_avg.save_pdb(ensemble_avg_path)
            results["ensemble_average_pdb"] = ensemble_avg_path
            logger.info(f"Saved ensemble average to {ensemble_avg_path}")

        # Save representative structures
        if self.config.save_representative_structures:
            for i, frame_idx in enumerate(representative_frames):
                rep_path = os.path.join(output_dir, f"representative_cluster_{i+1}.pdb")
                self.trajectory[frame_idx].save_pdb(rep_path)
                logger.info(f"Saved representative structure {i+1} to {rep_path}")

        # Save RMSF and confidence
        np.save(os.path.join(output_dir, "rmsf.npy"), rmsf)
        np.save(os.path.join(output_dir, "confidence.npy"), confidence)
        np.save(os.path.join(output_dir, "cluster_labels.npy"), cluster_labels)

        logger.info(f"Trajectory processing complete. Results saved to {output_dir}")

        return results


def run_md_and_process_for_pearl(
    protein_pdb: str,
    ligand_sdf: Optional[str] = None,
    protein_b_pdb: Optional[str] = None,
    output_dir: str = "./md_pearl_output",
    simulation_config: Optional['MDSimulationConfig'] = None,
    processing_config: Optional[TrajectoryProcessingConfig] = None
) -> Dict[str, any]:
    """
    Complete workflow: Run MD simulation and process for Pearl training.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file (for protein-ligand)
        protein_b_pdb: Path to second protein PDB file (for protein-protein)
        output_dir: Output directory
        simulation_config: MD simulation configuration
        processing_config: Trajectory processing configuration

    Returns:
        Dictionary with all results
    """
    from pearl.data.md_simulation import MDSimulationEngine, MDSimulationConfig

    os.makedirs(output_dir, exist_ok=True)

    # Run MD simulation
    logger.info("=" * 80)
    logger.info("STEP 1: Running MD Simulation")
    logger.info("=" * 80)

    sim_config = simulation_config or MDSimulationConfig()
    md_engine = MDSimulationEngine(sim_config)

    md_output_dir = os.path.join(output_dir, "md_simulation")
    md_results = md_engine.run_complete_workflow(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        protein_b_pdb=protein_b_pdb,
        output_dir=md_output_dir
    )

    # Process trajectory
    logger.info("=" * 80)
    logger.info("STEP 2: Processing Trajectory")
    logger.info("=" * 80)

    proc_config = processing_config or TrajectoryProcessingConfig()
    processor = MDTrajectoryProcessor(proc_config)

    processing_output_dir = os.path.join(output_dir, "processed")
    processing_results = processor.process_trajectory_for_pearl(
        trajectory_file=md_results["trajectory"],
        topology_file=md_results["system_pdb"],
        output_dir=processing_output_dir
    )

    # Combine results
    results = {
        "md_simulation": md_results,
        "trajectory_processing": processing_results,
        "output_dir": output_dir
    }

    logger.info("=" * 80)
    logger.info("COMPLETE WORKFLOW FINISHED")
    logger.info("=" * 80)
    logger.info(f"All results saved to {output_dir}")

    return results
