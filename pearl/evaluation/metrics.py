"""
Evaluation Metrics for Pearl

Implements key metrics from the paper:
- Ligand RMSD (BiSyRMSD - Binding-Site Superposed, Symmetry-Corrected)
- lDDT-PLI (Local Distance Difference Test for Protein-Ligand Interface)
- Success rates at different thresholds
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_rmsd(
    predicted_coords: torch.Tensor,
    true_coords: torch.Tensor,
    align: bool = True
) -> float:
    """
    Compute RMSD between predicted and true coordinates.
    
    Args:
        predicted_coords: Predicted coordinates [n_atoms, 3]
        true_coords: Ground truth coordinates [n_atoms, 3]
        align: Whether to align structures before computing RMSD
        
    Returns:
        RMSD value in Angstroms
    """
    pred = predicted_coords.detach().cpu().numpy()
    true = true_coords.detach().cpu().numpy()
    
    if align:
        # Align predicted to true using Kabsch algorithm
        pred = kabsch_align(pred, true)
    
    # Compute RMSD
    diff = pred - true
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return float(rmsd)


def kabsch_align(
    coords_to_align: np.ndarray,
    reference_coords: np.ndarray
) -> np.ndarray:
    """
    Align coords_to_align to reference_coords using Kabsch algorithm.
    
    Args:
        coords_to_align: Coordinates to align [n_atoms, 3]
        reference_coords: Reference coordinates [n_atoms, 3]
        
    Returns:
        Aligned coordinates [n_atoms, 3]
    """
    # Center both coordinate sets
    coords_centered = coords_to_align - coords_to_align.mean(axis=0)
    ref_centered = reference_coords - reference_coords.mean(axis=0)
    
    # Compute covariance matrix
    H = coords_centered.T @ ref_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    d = np.linalg.det(Vt.T @ U.T)
    correction = np.diag([1, 1, d])
    R = Vt.T @ correction @ U.T
    
    # Apply rotation and translation
    aligned = coords_centered @ R + reference_coords.mean(axis=0)
    
    return aligned


def compute_ligand_rmsd(
    predicted_coords: torch.Tensor,
    true_coords: torch.Tensor,
    protein_predicted: torch.Tensor,
    protein_true: torch.Tensor,
    symmetry_correction: bool = True
) -> float:
    """
    Compute ligand RMSD with binding site superposition (BiSyRMSD).
    
    First aligns the protein binding sites, then computes ligand RMSD.
    Optionally corrects for molecular symmetry.
    
    Args:
        predicted_coords: Predicted ligand coordinates [n_ligand_atoms, 3]
        true_coords: True ligand coordinates [n_ligand_atoms, 3]
        protein_predicted: Predicted protein coordinates [n_protein_atoms, 3]
        protein_true: True protein coordinates [n_protein_atoms, 3]
        symmetry_correction: Whether to correct for molecular symmetry
        
    Returns:
        Ligand RMSD in Angstroms
    """
    # Convert to numpy
    pred_lig = predicted_coords.detach().cpu().numpy()
    true_lig = true_coords.detach().cpu().numpy()
    pred_prot = protein_predicted.detach().cpu().numpy()
    true_prot = protein_true.detach().cpu().numpy()
    
    # Align protein binding sites
    pred_prot_aligned = kabsch_align(pred_prot, true_prot)
    
    # Apply same transformation to ligand
    # Compute transformation from protein alignment
    pred_prot_centered = pred_prot - pred_prot.mean(axis=0)
    true_prot_centered = true_prot - true_prot.mean(axis=0)
    H = pred_prot_centered.T @ true_prot_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    correction = np.diag([1, 1, d])
    R = Vt.T @ correction @ U.T
    
    # Apply to ligand
    pred_lig_centered = pred_lig - pred_prot.mean(axis=0)
    pred_lig_aligned = pred_lig_centered @ R + true_prot.mean(axis=0)
    
    # Symmetry correction if requested
    if symmetry_correction:
        rmsd = compute_symmetry_corrected_rmsd(pred_lig_aligned, true_lig)
    else:
        diff = pred_lig_aligned - true_lig
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return float(rmsd)


def compute_symmetry_corrected_rmsd(
    predicted_coords: np.ndarray,
    true_coords: np.ndarray
) -> float:
    """
    Compute RMSD with symmetry correction using Hungarian algorithm.
    
    Finds optimal atom correspondence to handle molecular symmetry.
    
    Args:
        predicted_coords: Predicted coordinates [n_atoms, 3]
        true_coords: True coordinates [n_atoms, 3]
        
    Returns:
        Symmetry-corrected RMSD
    """
    # Compute pairwise distances
    distances = cdist(predicted_coords, true_coords)
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Compute RMSD with optimal assignment
    matched_true = true_coords[col_ind]
    diff = predicted_coords[row_ind] - matched_true
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return float(rmsd)


def compute_lddt_pli(
    predicted_coords: torch.Tensor,
    true_coords: torch.Tensor,
    protein_predicted: torch.Tensor,
    protein_true: torch.Tensor,
    cutoff: float = 15.0,
    thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
) -> float:
    """
    Compute lDDT-PLI (Local Distance Difference Test for Protein-Ligand Interface).
    
    Measures local distance agreement in the protein-ligand interface.
    
    Args:
        predicted_coords: Predicted ligand coordinates [n_ligand_atoms, 3]
        true_coords: True ligand coordinates [n_ligand_atoms, 3]
        protein_predicted: Predicted protein coordinates [n_protein_atoms, 3]
        protein_true: True protein coordinates [n_protein_atoms, 3]
        cutoff: Distance cutoff for considering interactions
        thresholds: Distance difference thresholds for scoring
        
    Returns:
        lDDT-PLI score (0-100)
    """
    pred_lig = predicted_coords.detach().cpu().numpy()
    true_lig = true_coords.detach().cpu().numpy()
    pred_prot = protein_predicted.detach().cpu().numpy()
    true_prot = protein_true.detach().cpu().numpy()
    
    # Compute protein-ligand distances in true structure
    true_distances = cdist(true_lig, true_prot)
    
    # Find interactions within cutoff
    interactions = true_distances < cutoff
    
    if not interactions.any():
        return 0.0
    
    # Compute distances in predicted structure
    pred_distances = cdist(pred_lig, pred_prot)
    
    # Compute distance differences for interactions
    distance_diffs = np.abs(pred_distances[interactions] - true_distances[interactions])
    
    # Score based on thresholds
    scores = []
    for threshold in thresholds:
        score = (distance_diffs < threshold).mean()
        scores.append(score)
    
    # Average over thresholds
    lddt_pli = np.mean(scores) * 100
    
    return float(lddt_pli)


def compute_success_rate(
    rmsds: List[float],
    threshold: float = 2.0
) -> float:
    """
    Compute success rate at given RMSD threshold.
    
    Args:
        rmsds: List of RMSD values
        threshold: RMSD threshold for success (default 2.0 Å)
        
    Returns:
        Success rate as percentage
    """
    if not rmsds:
        return 0.0
    
    successes = sum(1 for rmsd in rmsds if rmsd < threshold)
    return (successes / len(rmsds)) * 100


def compute_best_at_k(
    samples_rmsds: List[List[float]],
    k: int = 5,
    threshold: float = 2.0
) -> float:
    """
    Compute best@k success rate.
    
    For each structure, checks if any of the top k samples has RMSD < threshold.
    
    Args:
        samples_rmsds: List of RMSD lists, one per structure [n_structures, n_samples]
        k: Number of samples to consider
        threshold: RMSD threshold for success
        
    Returns:
        Best@k success rate as percentage
    """
    if not samples_rmsds:
        return 0.0
    
    successes = 0
    for rmsds in samples_rmsds:
        # Sort and take best k
        best_k = sorted(rmsds)[:k]
        # Check if any are below threshold
        if any(rmsd < threshold for rmsd in best_k):
            successes += 1
    
    return (successes / len(samples_rmsds)) * 100


class MetricsCalculator:
    """
    Convenience class for computing multiple metrics.
    """
    
    def __init__(
        self,
        rmsd_thresholds: List[float] = [1.0, 2.0],
        compute_lddt: bool = True
    ):
        self.rmsd_thresholds = rmsd_thresholds
        self.compute_lddt = compute_lddt
        
    def compute_all_metrics(
        self,
        predicted_ligand: torch.Tensor,
        true_ligand: torch.Tensor,
        predicted_protein: torch.Tensor,
        true_protein: torch.Tensor
    ) -> dict:
        """
        Compute all metrics for a single prediction.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Ligand RMSD
        rmsd = compute_ligand_rmsd(
            predicted_ligand, true_ligand,
            predicted_protein, true_protein,
            symmetry_correction=True
        )
        metrics['ligand_rmsd'] = rmsd
        
        # Success at different thresholds
        for threshold in self.rmsd_thresholds:
            metrics[f'success_rmsd<{threshold}'] = 1.0 if rmsd < threshold else 0.0
        
        # lDDT-PLI
        if self.compute_lddt:
            lddt = compute_lddt_pli(
                predicted_ligand, true_ligand,
                predicted_protein, true_protein
            )
            metrics['lddt_pli'] = lddt
        
        return metrics
    
    def aggregate_metrics(
        self,
        all_metrics: List[dict]
    ) -> dict:
        """
        Aggregate metrics across multiple predictions.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not all_metrics:
            return {}
        
        aggregated = {}
        
        # Average RMSD
        rmsds = [m['ligand_rmsd'] for m in all_metrics]
        aggregated['mean_rmsd'] = np.mean(rmsds)
        aggregated['median_rmsd'] = np.median(rmsds)
        aggregated['std_rmsd'] = np.std(rmsds)
        
        # Success rates
        for threshold in self.rmsd_thresholds:
            key = f'success_rmsd<{threshold}'
            successes = [m[key] for m in all_metrics]
            aggregated[f'success_rate_rmsd<{threshold}'] = np.mean(successes) * 100
        
        # Average lDDT-PLI
        if self.compute_lddt and 'lddt_pli' in all_metrics[0]:
            lddts = [m['lddt_pli'] for m in all_metrics]
            aggregated['mean_lddt_pli'] = np.mean(lddts)
        
        return aggregated

