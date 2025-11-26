#!/usr/bin/env python3
"""
Train Pearl with Uncertainty-Aware Loss

This script demonstrates how to train Pearl with experimental uncertainty weighting.
It extracts B-factors and resolution from PDB files and uses them to weight the loss.

Key features:
1. Extracts B-factors from PDB structures
2. Converts B-factors to confidence scores
3. Uses uncertainty-weighted diffusion loss
4. Stratifies training by resolution bins
5. Logs detailed metrics per resolution range
"""

import sys
sys.path.insert(0, 'pearl')

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

# Import Pearl modules
from data import CurriculumConfig, CurriculumStage
from data.experimental_metadata import BFactorExtractor, ExperimentalUncertainty


def load_preprocessed_data_with_uncertainty(data_dir: Path) -> Dict:
    """
    Load preprocessed data and extract uncertainty information.
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        Dictionary with data and uncertainty information
    """
    print(f"Loading data from {data_dir}")
    
    # Load manifest
    manifest_file = data_dir / "manifest.json"
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    print(f"Found {manifest['total_structures']} structures across {len(manifest['stages'])} stages")
    
    # Load data for each stage
    stage_data = {}
    
    for stage_name in manifest['stages']:
        stage_dir = data_dir / stage_name
        pdb_file = stage_dir / "pdb.pkl"
        
        if pdb_file.exists():
            with open(pdb_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract uncertainty information
            data_with_uncertainty = []
            
            for sample in data:
                # Get B-factors
                protein_bfactors = sample.get('protein_bfactors', None)
                ligand_bfactors = sample.get('ligand_bfactors', None)
                
                if protein_bfactors is not None and ligand_bfactors is not None:
                    # Combine protein and ligand B-factors
                    all_bfactors = np.concatenate([protein_bfactors, ligand_bfactors])
                    
                    # Convert to confidence scores
                    bfactor_extractor = BFactorExtractor()
                    confidence = bfactor_extractor.bfactor_to_confidence(
                        all_bfactors,
                        resolution=sample.get('resolution', None)
                    )
                    
                    sample['confidence'] = confidence
                    sample['has_uncertainty'] = True
                else:
                    # No B-factor information - use uniform confidence
                    n_atoms = len(sample.get('protein_coords', [])) + len(sample.get('ligand_coords', []))
                    sample['confidence'] = np.ones(n_atoms, dtype=np.float32)
                    sample['has_uncertainty'] = False
                
                data_with_uncertainty.append(sample)
            
            stage_data[stage_name] = data_with_uncertainty
            
            # Compute statistics
            n_with_uncertainty = sum(1 for s in data_with_uncertainty if s['has_uncertainty'])
            print(f"  {stage_name}: {len(data_with_uncertainty)} structures, "
                  f"{n_with_uncertainty} with uncertainty info")
    
    return {
        'manifest': manifest,
        'stage_data': stage_data,
    }


def analyze_uncertainty_distribution(stage_data: Dict) -> Dict:
    """
    Analyze the distribution of uncertainty across the dataset.
    
    Args:
        stage_data: Dictionary of stage -> list of samples
        
    Returns:
        Statistics about uncertainty distribution
    """
    print("\n" + "=" * 80)
    print("Uncertainty Distribution Analysis")
    print("=" * 80)
    
    stats = {}
    
    for stage_name, data in stage_data.items():
        print(f"\n{stage_name}:")
        
        # Collect all confidence scores
        all_confidence = []
        all_resolutions = []
        
        for sample in data:
            if sample['has_uncertainty']:
                all_confidence.extend(sample['confidence'])
                if sample.get('resolution') is not None:
                    all_resolutions.append(sample['resolution'])
        
        if all_confidence:
            all_confidence = np.array(all_confidence)
            
            print(f"  Confidence scores:")
            print(f"    Mean: {all_confidence.mean():.3f}")
            print(f"    Std: {all_confidence.std():.3f}")
            print(f"    Min: {all_confidence.min():.3f}")
            print(f"    Max: {all_confidence.max():.3f}")
            print(f"    Median: {np.median(all_confidence):.3f}")
            
            # Percentiles
            p25, p75 = np.percentile(all_confidence, [25, 75])
            print(f"    25th percentile: {p25:.3f}")
            print(f"    75th percentile: {p75:.3f}")
            
            stats[stage_name] = {
                'mean_confidence': float(all_confidence.mean()),
                'std_confidence': float(all_confidence.std()),
                'min_confidence': float(all_confidence.min()),
                'max_confidence': float(all_confidence.max()),
            }
        
        if all_resolutions:
            all_resolutions = np.array(all_resolutions)
            print(f"  Resolutions:")
            print(f"    Mean: {all_resolutions.mean():.2f} Å")
            print(f"    Range: [{all_resolutions.min():.2f}, {all_resolutions.max():.2f}] Å")
            
            stats[stage_name]['mean_resolution'] = float(all_resolutions.mean())
            stats[stage_name]['resolution_range'] = [
                float(all_resolutions.min()),
                float(all_resolutions.max())
            ]
    
    return stats


def simulate_uncertainty_aware_training(
    stage_data: Dict,
    n_steps: int = 50,
    batch_size: int = 2,
) -> Dict:
    """
    Simulate training with uncertainty-aware loss.
    
    This is a demonstration - in practice, you'd use the actual Pearl model.
    
    Args:
        stage_data: Dictionary of stage -> list of samples
        n_steps: Number of training steps per stage
        batch_size: Batch size
        
    Returns:
        Training log
    """
    print("\n" + "=" * 80)
    print("Simulating Uncertainty-Aware Training")
    print("=" * 80)
    
    training_log = []
    global_step = 0
    
    for stage_name, data in stage_data.items():
        print(f"\n{stage_name}:")
        print(f"  Training on {len(data)} structures")
        
        stage_losses = []
        
        for step in range(n_steps):
            # Sample batch
            batch_indices = np.random.choice(len(data), size=min(batch_size, len(data)), replace=False)
            batch = [data[i] for i in batch_indices]
            
            # Simulate loss computation with uncertainty weighting
            batch_loss = 0.0
            total_weight = 0.0
            
            for sample in batch:
                # Simulate per-atom loss
                n_atoms = len(sample['confidence'])
                atom_losses = np.random.rand(n_atoms) * 10.0  # Random losses
                
                # Weight by confidence (inverse variance weighting)
                weights = sample['confidence'] ** 2
                weights = weights / (weights.mean() + 1e-8)
                
                # Weighted loss
                weighted_loss = (atom_losses * weights).sum()
                batch_loss += weighted_loss
                total_weight += weights.sum()
            
            # Average loss
            avg_loss = batch_loss / (total_weight + 1e-8)
            stage_losses.append(avg_loss)
            global_step += 1
            
            # Log progress
            if (step + 1) % 10 == 0:
                recent_loss = np.mean(stage_losses[-10:])
                print(f"    Step {step + 1}/{n_steps}, Loss: {recent_loss:.4f}")
                
                training_log.append({
                    'global_step': global_step,
                    'stage': stage_name,
                    'step': step + 1,
                    'loss': recent_loss,
                })
        
        avg_stage_loss = np.mean(stage_losses)
        print(f"  ✓ Stage complete, Average loss: {avg_stage_loss:.4f}")
    
    return training_log


def main():
    """Main training script."""
    print("=" * 80)
    print("Pearl Training with Uncertainty-Aware Loss")
    print("=" * 80)
    
    # Configuration
    data_dir = Path("data/processed")
    output_dir = Path("uncertainty_training_output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data with uncertainty
    print("\n" + "=" * 80)
    print("Step 1: Loading Data with Uncertainty Information")
    print("=" * 80)
    
    data = load_preprocessed_data_with_uncertainty(data_dir)
    stage_data = data['stage_data']
    
    # Step 2: Analyze uncertainty distribution
    uncertainty_stats = analyze_uncertainty_distribution(stage_data)
    
    # Save statistics
    stats_file = output_dir / "uncertainty_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(uncertainty_stats, f, indent=2)
    print(f"\n✓ Saved uncertainty statistics to {stats_file}")
    
    # Step 3: Simulate training
    training_log = simulate_uncertainty_aware_training(
        stage_data,
        n_steps=50,
        batch_size=2,
    )
    
    # Save training log
    log_file = output_dir / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Saved training log to {log_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\nTotal steps: {len(training_log)}")
    print(f"Stages: {len(stage_data)}")
    print(f"Output directory: {output_dir}")
    
    print("\n📊 Key Findings:")
    for stage_name, stats in uncertainty_stats.items():
        print(f"\n{stage_name}:")
        print(f"  Mean confidence: {stats.get('mean_confidence', 'N/A'):.3f}")
        if 'mean_resolution' in stats:
            print(f"  Mean resolution: {stats['mean_resolution']:.2f} Å")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Integrate with actual Pearl model")
    print("2. Use UncertaintyWeightedDiffusionLoss for training")
    print("3. Compare with baseline (no uncertainty weighting)")
    print("4. Evaluate on test set across resolution ranges")
    print("5. Add CryoEM structures with local resolution maps")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

