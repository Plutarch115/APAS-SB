"""
Protein-Protein Interaction (PPI) Evaluation Metrics

This module implements standard PPI evaluation metrics:
- Interface RMSD (i-RMSD)
- Ligand RMSD (l-RMSD)
- Fraction of native contacts (Fnat)
- DockQ score
- CAPRI criteria

Compatible with Pearl's evaluation framework.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PPIMetrics:
    """
    Evaluation metrics for protein-protein interaction prediction.
    
    Implements standard metrics from CAPRI and DockQ.
    """
    
    def __init__(
        self,
        contact_threshold: float = 5.0,
        interface_cutoff: float = 10.0
    ):
        """
        Initialize PPI metrics.
        
        Args:
            contact_threshold: Distance threshold (Å) for native contacts
            interface_cutoff: Distance cutoff (Å) for interface residues
        """
        self.contact_threshold = contact_threshold
        self.interface_cutoff = interface_cutoff
    
    def compute_interface_rmsd(
        self,
        pred_coords_a: np.ndarray,
        pred_coords_b: np.ndarray,
        true_coords_a: np.ndarray,
        true_coords_b: np.ndarray,
        align: bool = True
    ) -> float:
        """
        Compute interface RMSD (i-RMSD).
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (N_b, 3)
            true_coords_a: True coordinates for chain A (N_a, 3)
            true_coords_b: True coordinates for chain B (N_b, 3)
            align: Whether to align structures before computing RMSD
            
        Returns:
            Interface RMSD in Ångströms
        """
        # Identify interface residues
        interface_mask_a, interface_mask_b = self._identify_interface_residues(
            true_coords_a, true_coords_b
        )
        
        # Extract interface coordinates
        pred_interface_a = pred_coords_a[interface_mask_a]
        pred_interface_b = pred_coords_b[interface_mask_b]
        true_interface_a = true_coords_a[interface_mask_a]
        true_interface_b = true_coords_b[interface_mask_b]
        
        # Combine interface coordinates
        pred_interface = np.vstack([pred_interface_a, pred_interface_b])
        true_interface = np.vstack([true_interface_a, true_interface_b])
        
        # Align if requested
        if align:
            pred_interface = self._kabsch_align(pred_interface, true_interface)
        
        # Compute RMSD
        rmsd = np.sqrt(((pred_interface - true_interface) ** 2).sum(axis=1).mean())
        
        return rmsd
    
    def compute_ligand_rmsd(
        self,
        pred_coords_b: np.ndarray,
        true_coords_b: np.ndarray,
        pred_coords_a: np.ndarray,
        true_coords_a: np.ndarray,
        align_on_receptor: bool = True
    ) -> float:
        """
        Compute ligand RMSD (l-RMSD).
        
        Chain B is considered the "ligand" and chain A the "receptor".
        
        Args:
            pred_coords_b: Predicted coordinates for chain B (N_b, 3)
            true_coords_b: True coordinates for chain B (N_b, 3)
            pred_coords_a: Predicted coordinates for chain A (N_a, 3)
            true_coords_a: True coordinates for chain A (N_a, 3)
            align_on_receptor: Whether to align on receptor (chain A) first
            
        Returns:
            Ligand RMSD in Ångströms
        """
        if align_on_receptor:
            # Align on receptor (chain A)
            rotation, translation = self._compute_alignment(pred_coords_a, true_coords_a)
            
            # Apply transformation to ligand (chain B)
            pred_coords_b_aligned = (pred_coords_b - pred_coords_b.mean(axis=0)) @ rotation.T + translation
        else:
            pred_coords_b_aligned = pred_coords_b
        
        # Compute RMSD
        rmsd = np.sqrt(((pred_coords_b_aligned - true_coords_b) ** 2).sum(axis=1).mean())
        
        return rmsd
    
    def compute_fnat(
        self,
        pred_coords_a: np.ndarray,
        pred_coords_b: np.ndarray,
        true_coords_a: np.ndarray,
        true_coords_b: np.ndarray
    ) -> float:
        """
        Compute fraction of native contacts (Fnat).
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (N_b, 3)
            true_coords_a: True coordinates for chain A (N_a, 3)
            true_coords_b: True coordinates for chain B (N_b, 3)
            
        Returns:
            Fnat (fraction between 0 and 1)
        """
        # Compute true contacts
        true_distances = np.linalg.norm(
            true_coords_a[:, np.newaxis, :] - true_coords_b[np.newaxis, :, :],
            axis=2
        )
        true_contacts = (true_distances < self.contact_threshold)
        
        # Compute predicted contacts
        pred_distances = np.linalg.norm(
            pred_coords_a[:, np.newaxis, :] - pred_coords_b[np.newaxis, :, :],
            axis=2
        )
        pred_contacts = (pred_distances < self.contact_threshold)
        
        # Compute Fnat
        true_positives = (true_contacts & pred_contacts).sum()
        total_native = true_contacts.sum()
        
        if total_native == 0:
            return 0.0
        
        fnat = true_positives / total_native
        
        return fnat
    
    def compute_dockq(
        self,
        pred_coords_a: np.ndarray,
        pred_coords_b: np.ndarray,
        true_coords_a: np.ndarray,
        true_coords_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute DockQ score.
        
        DockQ = (Fnat + 1/(1 + (i-RMSD/1.5)^2) + 1/(1 + (l-RMSD/8.5)^2)) / 3
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (N_b, 3)
            true_coords_a: True coordinates for chain A (N_a, 3)
            true_coords_b: True coordinates for chain B (N_b, 3)
            
        Returns:
            Dictionary with DockQ score and components
        """
        # Compute components
        irmsd = self.compute_interface_rmsd(
            pred_coords_a, pred_coords_b,
            true_coords_a, true_coords_b
        )
        
        lrmsd = self.compute_ligand_rmsd(
            pred_coords_b, true_coords_b,
            pred_coords_a, true_coords_a
        )
        
        fnat = self.compute_fnat(
            pred_coords_a, pred_coords_b,
            true_coords_a, true_coords_b
        )
        
        # Compute DockQ
        dockq = (
            fnat +
            1.0 / (1.0 + (irmsd / 1.5) ** 2) +
            1.0 / (1.0 + (lrmsd / 8.5) ** 2)
        ) / 3.0
        
        return {
            "dockq": dockq,
            "irmsd": irmsd,
            "lrmsd": lrmsd,
            "fnat": fnat
        }
    
    def classify_capri(
        self,
        irmsd: float,
        lrmsd: float,
        fnat: float
    ) -> str:
        """
        Classify prediction according to CAPRI criteria.
        
        Args:
            irmsd: Interface RMSD
            lrmsd: Ligand RMSD
            fnat: Fraction of native contacts
            
        Returns:
            CAPRI classification: "high", "medium", "acceptable", or "incorrect"
        """
        # CAPRI criteria
        if fnat >= 0.5 and (lrmsd <= 1.0 or irmsd <= 1.0):
            return "high"
        elif fnat >= 0.3 and (lrmsd <= 5.0 or irmsd <= 2.0):
            return "medium"
        elif fnat >= 0.1 and (lrmsd <= 10.0 or irmsd <= 4.0):
            return "acceptable"
        else:
            return "incorrect"
    
    def evaluate_prediction(
        self,
        pred_coords_a: np.ndarray,
        pred_coords_b: np.ndarray,
        true_coords_a: np.ndarray,
        true_coords_b: np.ndarray
    ) -> Dict[str, any]:
        """
        Complete evaluation of PPI prediction.
        
        Args:
            pred_coords_a: Predicted coordinates for chain A (N_a, 3)
            pred_coords_b: Predicted coordinates for chain B (N_b, 3)
            true_coords_a: True coordinates for chain A (N_a, 3)
            true_coords_b: True coordinates for chain B (N_b, 3)
            
        Returns:
            Dictionary with all metrics
        """
        # Compute DockQ and components
        dockq_results = self.compute_dockq(
            pred_coords_a, pred_coords_b,
            true_coords_a, true_coords_b
        )
        
        # CAPRI classification
        capri_class = self.classify_capri(
            dockq_results["irmsd"],
            dockq_results["lrmsd"],
            dockq_results["fnat"]
        )
        
        results = {
            **dockq_results,
            "capri_class": capri_class
        }
        
        return results
    
    def _identify_interface_residues(
        self,
        coords_a: np.ndarray,
        coords_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify interface residues based on distance cutoff"""
        distances = np.linalg.norm(
            coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :],
            axis=2
        )
        
        interface_mask_a = distances.min(axis=1) < self.interface_cutoff
        interface_mask_b = distances.min(axis=0) < self.interface_cutoff
        
        return interface_mask_a, interface_mask_b
    
    def _kabsch_align(
        self,
        coords_mobile: np.ndarray,
        coords_target: np.ndarray
    ) -> np.ndarray:
        """Align mobile coordinates to target using Kabsch algorithm"""
        rotation, translation = self._compute_alignment(coords_mobile, coords_target)
        
        # Apply transformation
        coords_aligned = (coords_mobile - coords_mobile.mean(axis=0)) @ rotation.T + translation
        
        return coords_aligned
    
    def _compute_alignment(
        self,
        coords_mobile: np.ndarray,
        coords_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rotation and translation for alignment"""
        # Center coordinates
        mobile_center = coords_mobile.mean(axis=0)
        target_center = coords_target.mean(axis=0)
        
        mobile_centered = coords_mobile - mobile_center
        target_centered = coords_target - target_center
        
        # Compute covariance matrix
        H = mobile_centered.T @ target_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        rotation = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T
        
        return rotation, target_center


def evaluate_ppi_batch(
    pred_coords_a_batch: np.ndarray,
    pred_coords_b_batch: np.ndarray,
    true_coords_a_batch: np.ndarray,
    true_coords_b_batch: np.ndarray,
    contact_threshold: float = 5.0,
    interface_cutoff: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Evaluate batch of PPI predictions.
    
    Args:
        pred_coords_a_batch: Predicted coordinates for chain A (B, N_a, 3)
        pred_coords_b_batch: Predicted coordinates for chain B (B, N_b, 3)
        true_coords_a_batch: True coordinates for chain A (B, N_a, 3)
        true_coords_b_batch: True coordinates for chain B (B, N_b, 3)
        contact_threshold: Distance threshold for contacts
        interface_cutoff: Distance cutoff for interface
        
    Returns:
        Dictionary with arrays of metrics for each sample
    """
    metrics = PPIMetrics(contact_threshold, interface_cutoff)
    
    batch_size = pred_coords_a_batch.shape[0]
    
    dockq_scores = []
    irmsd_scores = []
    lrmsd_scores = []
    fnat_scores = []
    capri_classes = []
    
    for i in range(batch_size):
        results = metrics.evaluate_prediction(
            pred_coords_a_batch[i],
            pred_coords_b_batch[i],
            true_coords_a_batch[i],
            true_coords_b_batch[i]
        )
        
        dockq_scores.append(results["dockq"])
        irmsd_scores.append(results["irmsd"])
        lrmsd_scores.append(results["lrmsd"])
        fnat_scores.append(results["fnat"])
        capri_classes.append(results["capri_class"])
    
    return {
        "dockq": np.array(dockq_scores),
        "irmsd": np.array(irmsd_scores),
        "lrmsd": np.array(lrmsd_scores),
        "fnat": np.array(fnat_scores),
        "capri_class": capri_classes
    }

