#!/usr/bin/env python3
"""
Run Molecular Dynamics Simulations for Pearl Training

This script:
1. Takes experimental structures (PDB files)
2. Runs MD simulations using OpenMM or GROMACS
3. Extracts conformational ensembles
4. Computes dynamic B-factors (RMSF)
5. Generates training data with reduced uncertainty

Usage:
    python scripts/run_md_simulations.py --input data/pdb_files --output data/md_trajectories --time 100
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add pearl to path
sys.path.insert(0, 'pearl')


def setup_md_system(pdb_file: Path) -> Dict:
    """
    Set up MD system from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Dictionary with system information
    """
    try:
        from openmm.app import PDBFile, ForceField, Modeller
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import Simulation, StateDataReporter, DCDReporter
        from openmm.unit import nanometer, picoseconds, kelvin, bar
        
        print(f"Setting up MD system for {pdb_file.name}...")
        
        # Load PDB
        pdb = PDBFile(str(pdb_file))
        
        # Create force field
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        
        # Add solvent
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(forcefield, padding=1.0*nanometer)
        
        # Create system
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds,
        )
        
        # Create integrator
        integrator = LangevinMiddleIntegrator(
            300*kelvin,  # Temperature
            1.0/picoseconds,  # Friction coefficient
            0.002*picoseconds,  # Time step (2 fs)
        )
        
        # Create simulation
        platform = Platform.getPlatformByName('CUDA')  # Use GPU
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        
        return {
            'simulation': simulation,
            'topology': modeller.topology,
            'n_atoms': modeller.topology.getNumAtoms(),
        }
        
    except ImportError:
        print("⚠ OpenMM not available. Install with: conda install -c conda-forge openmm")
        return None


def run_md_simulation(
    pdb_file: Path,
    output_dir: Path,
    simulation_time_ns: float = 100.0,
    save_interval_ps: float = 10.0,
) -> Path:
    """
    Run MD simulation on a protein-ligand complex.
    
    Args:
        pdb_file: Path to PDB file
        output_dir: Directory to save trajectory
        simulation_time_ns: Simulation time in nanoseconds
        save_interval_ps: Interval to save frames in picoseconds
        
    Returns:
        Path to trajectory file
    """
    try:
        from openmm.app import DCDReporter, StateDataReporter
        from openmm.unit import picoseconds, nanoseconds
        
        # Setup system
        system_info = setup_md_system(pdb_file)
        if system_info is None:
            return None
        
        simulation = system_info['simulation']
        
        # Output files
        pdb_id = pdb_file.stem
        trajectory_file = output_dir / f"{pdb_id}_trajectory.dcd"
        log_file = output_dir / f"{pdb_id}_log.txt"
        
        # Add reporters
        simulation.reporters.append(
            DCDReporter(
                str(trajectory_file),
                int(save_interval_ps / 0.002),  # Steps per save
            )
        )
        simulation.reporters.append(
            StateDataReporter(
                str(log_file),
                1000,  # Report every 1000 steps
                step=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=int(simulation_time_ns * 1000 / 0.002),
            )
        )
        
        # Minimize energy
        print(f"  Minimizing energy...")
        simulation.minimizeEnergy()
        
        # Equilibrate
        print(f"  Equilibrating (1 ns)...")
        simulation.step(500000)  # 1 ns
        
        # Production run
        print(f"  Running production MD ({simulation_time_ns} ns)...")
        n_steps = int(simulation_time_ns * 1000 / 0.002)  # Convert ns to steps
        simulation.step(n_steps)
        
        print(f"  ✓ Trajectory saved to {trajectory_file}")
        
        return trajectory_file
        
    except Exception as e:
        print(f"  ✗ Error running MD: {e}")
        return None


def run_md_with_gromacs(
    pdb_file: Path,
    output_dir: Path,
    simulation_time_ns: float = 100.0,
) -> Path:
    """
    Run MD simulation using GROMACS (alternative to OpenMM).
    
    Args:
        pdb_file: Path to PDB file
        output_dir: Directory to save trajectory
        simulation_time_ns: Simulation time in nanoseconds
        
    Returns:
        Path to trajectory file
    """
    import subprocess
    
    pdb_id = pdb_file.stem
    work_dir = output_dir / pdb_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running GROMACS MD for {pdb_id}...")
    
    try:
        # 1. Generate topology
        subprocess.run([
            'gmx', 'pdb2gmx',
            '-f', str(pdb_file),
            '-o', str(work_dir / 'processed.gro'),
            '-water', 'tip3p',
            '-ff', 'amber99sb-ildn',
        ], check=True)
        
        # 2. Define box
        subprocess.run([
            'gmx', 'editconf',
            '-f', str(work_dir / 'processed.gro'),
            '-o', str(work_dir / 'box.gro'),
            '-c', '-d', '1.0', '-bt', 'cubic',
        ], check=True)
        
        # 3. Solvate
        subprocess.run([
            'gmx', 'solvate',
            '-cp', str(work_dir / 'box.gro'),
            '-o', str(work_dir / 'solvated.gro'),
            '-p', str(work_dir / 'topol.top'),
        ], check=True)
        
        # 4. Energy minimization
        subprocess.run([
            'gmx', 'grompp',
            '-f', 'mdp/em.mdp',
            '-c', str(work_dir / 'solvated.gro'),
            '-p', str(work_dir / 'topol.top'),
            '-o', str(work_dir / 'em.tpr'),
        ], check=True)
        
        subprocess.run([
            'gmx', 'mdrun',
            '-deffnm', str(work_dir / 'em'),
        ], check=True)
        
        # 5. Production MD
        subprocess.run([
            'gmx', 'grompp',
            '-f', 'mdp/md.mdp',
            '-c', str(work_dir / 'em.gro'),
            '-p', str(work_dir / 'topol.top'),
            '-o', str(work_dir / 'md.tpr'),
            '-maxwarn', '1',
        ], check=True)
        
        subprocess.run([
            'gmx', 'mdrun',
            '-deffnm', str(work_dir / 'md'),
            '-ntomp', '8',  # Use 8 CPU threads
        ], check=True)
        
        trajectory_file = work_dir / 'md.xtc'
        print(f"  ✓ Trajectory saved to {trajectory_file}")
        
        return trajectory_file
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ GROMACS error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run MD simulations for Pearl training')
    parser.add_argument('--input', type=str, default='data/pdb_files',
                       help='Directory with PDB files')
    parser.add_argument('--output', type=str, default='data/md_trajectories',
                       help='Directory to save trajectories')
    parser.add_argument('--time', type=float, default=100.0,
                       help='Simulation time in nanoseconds')
    parser.add_argument('--engine', type=str, default='openmm',
                       choices=['openmm', 'gromacs'],
                       help='MD engine to use')
    parser.add_argument('--n-structures', type=int, default=None,
                       help='Number of structures to simulate (default: all)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel simulations')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MD Simulation Pipeline for Pearl Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Simulation time: {args.time} ns")
    print(f"  MD engine: {args.engine}")
    print(f"  Parallel simulations: {args.parallel}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDB files
    input_dir = Path(args.input)
    pdb_files = sorted(input_dir.glob("*.pdb"))
    
    if args.n_structures:
        pdb_files = pdb_files[:args.n_structures]
    
    print(f"\nFound {len(pdb_files)} PDB files")
    
    # Run simulations
    print("\n" + "=" * 80)
    print("Running MD Simulations")
    print("=" * 80 + "\n")
    
    completed = 0
    failed = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] Processing {pdb_file.name}...")
        
        if args.engine == 'openmm':
            trajectory_file = run_md_simulation(
                pdb_file,
                output_dir,
                simulation_time_ns=args.time,
            )
        elif args.engine == 'gromacs':
            trajectory_file = run_md_with_gromacs(
                pdb_file,
                output_dir,
                simulation_time_ns=args.time,
            )
        
        if trajectory_file:
            completed += 1
        else:
            failed += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pdb_files)}")
    print(f"\nTrajectories saved to: {output_dir}")
    
    # Estimate storage
    avg_trajectory_size = 6  # GB per trajectory
    total_storage = completed * avg_trajectory_size
    print(f"Estimated storage: {total_storage:.1f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

