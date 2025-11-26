#!/usr/bin/env python3
"""
Complete MD Simulation Workflow for Pearl Training

This script runs:
1. MD simulations with OpenMM + OpenFF
2. Trajectory processing and uncertainty extraction
3. Data preparation for Pearl training

Supports both protein-ligand and protein-protein complexes.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add pearl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pearl.data.md_simulation import MDSimulationEngine, MDSimulationConfig
from pearl.data.md_trajectory_processor import (
    MDTrajectoryProcessor,
    TrajectoryProcessingConfig,
    run_md_and_process_for_pearl
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MD simulation workflow for Pearl")
    
    # Input files
    parser.add_argument("--protein", required=True, help="Protein PDB file")
    parser.add_argument("--ligand", help="Ligand SDF file (for protein-ligand)")
    parser.add_argument("--protein-b", help="Second protein PDB file (for protein-protein)")
    
    # Output
    parser.add_argument("--output-dir", default="./md_output", help="Output directory")
    
    # Simulation parameters
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("--production-time", type=float, default=100.0, help="Production time (ns)")
    parser.add_argument("--equilibration-time", type=float, default=1.0, help="Equilibration time (ns)")
    parser.add_argument("--timestep", type=float, default=2.0, help="Timestep (fs)")
    
    # Platform
    parser.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"], 
                       help="OpenMM platform")
    parser.add_argument("--precision", default="mixed", choices=["single", "mixed", "double"],
                       help="Precision")
    
    # Processing parameters
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of conformational clusters")
    parser.add_argument("--stride", type=int, default=1, help="Trajectory stride")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.ligand and not args.protein_b:
        parser.error("Must provide either --ligand or --protein-b")
    
    if args.ligand and args.protein_b:
        parser.error("Cannot provide both --ligand and --protein-b")
    
    # Setup configuration
    sim_config = MDSimulationConfig(
        temperature=args.temperature,
        production_time=args.production_time,
        equilibration_time=args.equilibration_time,
        timestep=args.timestep,
        platform=args.platform,
        precision=args.precision
    )
    
    proc_config = TrajectoryProcessingConfig(
        n_clusters=args.n_clusters,
        stride=args.stride
    )
    
    # Run workflow
    logger.info("="*80)
    logger.info("MD SIMULATION WORKFLOW FOR PEARL TRAINING")
    logger.info("="*80)
    logger.info(f"Protein: {args.protein}")
    if args.ligand:
        logger.info(f"Ligand: {args.ligand}")
        logger.info("Task: Protein-Ligand")
    else:
        logger.info(f"Protein B: {args.protein_b}")
        logger.info("Task: Protein-Protein")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Production time: {args.production_time} ns")
    logger.info("="*80)
    
    results = run_md_and_process_for_pearl(
        protein_pdb=args.protein,
        ligand_sdf=args.ligand,
        protein_b_pdb=args.protein_b,
        output_dir=args.output_dir,
        simulation_config=sim_config,
        processing_config=proc_config
    )
    
    # Print summary
    logger.info("="*80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*80)
    logger.info(f"MD simulation results: {results['md_simulation']}")
    logger.info(f"Trajectory processing results: {results['trajectory_processing']}")
    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info("="*80)
    
    # Print key metrics
    proc_results = results['trajectory_processing']
    logger.info("\nKey Metrics:")
    logger.info(f"  Frames processed: {proc_results['n_frames']}")
    logger.info(f"  Atoms: {proc_results['n_atoms']}")
    logger.info(f"  RMSF range: {proc_results['rmsf'].min():.2f} - {proc_results['rmsf'].max():.2f} Å")
    logger.info(f"  Confidence range: {proc_results['confidence'].min():.3f} - {proc_results['confidence'].max():.3f}")
    logger.info(f"  Conformational clusters: {len(proc_results['representative_frames'])}")
    
    logger.info("\nFiles generated:")
    logger.info(f"  - Ensemble average: {proc_results.get('ensemble_average_pdb', 'N/A')}")
    logger.info(f"  - RMSF data: {args.output_dir}/processed/rmsf.npy")
    logger.info(f"  - Confidence data: {args.output_dir}/processed/confidence.npy")
    logger.info(f"  - Trajectory: {results['md_simulation']['trajectory']}")
    
    logger.info("\n✅ Ready for Pearl training!")


if __name__ == "__main__":
    main()

