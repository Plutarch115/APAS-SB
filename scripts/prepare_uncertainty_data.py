#!/usr/bin/env python3
"""
Prepare Uncertainty Data for Pearl Training

This script processes both X-ray and CryoEM structures to extract uncertainty information:
- X-ray: Extract B-factors from PDB files
- CryoEM: Extract B-factors AND local resolution from maps

Creates a unified dataset with per-atom confidence scores ready for training.
"""

import sys
sys.path.insert(0, 'pearl')

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

from data.experimental_metadata import (
    BFactorExtractor,
    CryoEMLocalResolution,
    ExperimentalMetadataExtractor,
    ExperimentalUncertainty,
)


def extract_atoms_from_pdb(pdb_file: Path) -> Dict:
    """
    Extract atom information from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Dictionary with atom coordinates, elements, and B-factors
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # Extract protein atoms
    protein_coords = []
    protein_atoms = []
    protein_bfactors = []
    protein_residues = []
    
    for model in structure:
        for chain in model:
            for res_idx, residue in enumerate(chain):
                # Skip hetero atoms for protein
                if residue.id[0] != ' ':
                    continue
                    
                for atom in residue:
                    if atom.element != 'H':  # Skip hydrogens
                        protein_coords.append(atom.get_coord())
                        protein_atoms.append(atom.element)
                        protein_bfactors.append(atom.bfactor)
                        protein_residues.append(res_idx)
    
    # Extract ligand atoms (HETATM)
    ligand_coords = []
    ligand_atoms = []
    ligand_bfactors = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only hetero atoms
                if residue.id[0] == ' ':
                    continue
                
                # Skip water
                if residue.resname == 'HOH':
                    continue
                    
                for atom in residue:
                    if atom.element != 'H':
                        ligand_coords.append(atom.get_coord())
                        ligand_atoms.append(atom.element)
                        ligand_bfactors.append(atom.bfactor)
    
    return {
        'protein_coords': np.array(protein_coords, dtype=np.float32),
        'protein_atoms': protein_atoms,
        'protein_bfactors': np.array(protein_bfactors, dtype=np.float32),
        'protein_residues': np.array(protein_residues, dtype=np.int64),
        'ligand_coords': np.array(ligand_coords, dtype=np.float32),
        'ligand_atoms': ligand_atoms,
        'ligand_bfactors': np.array(ligand_bfactors, dtype=np.float32),
    }


def extract_resolution_from_pdb(pdb_file: Path) -> tuple:
    """
    Extract experimental method and resolution from PDB header.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        (method, resolution) tuple
    """
    method = 'XRAY'
    resolution = None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('EXPDTA'):
                if 'ELECTRON MICROSCOPY' in line:
                    method = 'EM'
                elif 'NMR' in line:
                    method = 'NMR'
            elif line.startswith('REMARK   2 RESOLUTION'):
                try:
                    parts = line.split()
                    resolution = float(parts[3])
                except (IndexError, ValueError):
                    pass
    
    return method, resolution


def process_xray_structure(pdb_file: Path, pdb_id: str) -> Optional[Dict]:
    """
    Process X-ray structure to extract uncertainty from B-factors.
    
    Args:
        pdb_file: Path to PDB file
        pdb_id: PDB identifier
        
    Returns:
        Dictionary with structure data and uncertainty
    """
    print(f"  Processing X-ray structure: {pdb_id}")
    
    try:
        # Extract atoms
        atoms = extract_atoms_from_pdb(pdb_file)
        
        if len(atoms['protein_coords']) == 0:
            print(f"    ✗ No protein atoms found")
            return None
        
        # Extract resolution
        method, resolution = extract_resolution_from_pdb(pdb_file)
        
        # Convert B-factors to confidence
        bfactor_extractor = BFactorExtractor()
        
        # Combine protein and ligand B-factors
        all_bfactors = np.concatenate([
            atoms['protein_bfactors'],
            atoms['ligand_bfactors']
        ])
        
        # Convert to confidence
        confidence = bfactor_extractor.bfactor_to_confidence(
            all_bfactors,
            resolution=resolution
        )
        
        # Split back
        n_protein = len(atoms['protein_bfactors'])
        protein_confidence = confidence[:n_protein]
        ligand_confidence = confidence[n_protein:]
        
        print(f"    ✓ Protein atoms: {len(atoms['protein_coords'])}")
        print(f"    ✓ Ligand atoms: {len(atoms['ligand_coords'])}")
        print(f"    ✓ Resolution: {resolution} Å")
        print(f"    ✓ Confidence: [{confidence.min():.3f}, {confidence.max():.3f}]")
        
        return {
            'pdb_id': pdb_id,
            'method': method,
            'resolution': resolution,
            'protein_coords': atoms['protein_coords'],
            'protein_atoms': atoms['protein_atoms'],
            'protein_bfactors': atoms['protein_bfactors'],
            'protein_confidence': protein_confidence,
            'ligand_coords': atoms['ligand_coords'],
            'ligand_atoms': atoms['ligand_atoms'],
            'ligand_bfactors': atoms['ligand_bfactors'],
            'ligand_confidence': ligand_confidence,
            'uncertainty_source': 'bfactor',
        }
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def process_cryoem_structure(
    pdb_file: Path,
    pdb_id: str,
    local_res_map: Optional[Path] = None
) -> Optional[Dict]:
    """
    Process CryoEM structure to extract uncertainty from B-factors and/or local resolution.
    
    Args:
        pdb_file: Path to PDB file
        pdb_id: PDB identifier
        local_res_map: Optional path to local resolution map
        
    Returns:
        Dictionary with structure data and uncertainty
    """
    print(f"  Processing CryoEM structure: {pdb_id}")
    
    try:
        # Extract atoms
        atoms = extract_atoms_from_pdb(pdb_file)
        
        if len(atoms['protein_coords']) == 0:
            print(f"    ✗ No protein atoms found")
            return None
        
        # Extract resolution
        method, resolution = extract_resolution_from_pdb(pdb_file)
        
        # Try to use local resolution map if available
        if local_res_map and local_res_map.exists():
            print(f"    Using local resolution map")
            try:
                cryoem = CryoEMLocalResolution()
                resolution_map = cryoem.load_local_resolution_map(str(local_res_map))
                
                # Interpolate for protein atoms
                protein_local_res = cryoem.interpolate_atom_resolution(
                    atoms['protein_coords'],
                    resolution_map,
                    voxel_size=1.0,  # Typical value
                    origin=np.array([0, 0, 0])
                )
                
                # Interpolate for ligand atoms
                if len(atoms['ligand_coords']) > 0:
                    ligand_local_res = cryoem.interpolate_atom_resolution(
                        atoms['ligand_coords'],
                        resolution_map,
                        voxel_size=1.0,
                        origin=np.array([0, 0, 0])
                    )
                else:
                    ligand_local_res = np.array([])
                
                # Convert to confidence
                all_local_res = np.concatenate([protein_local_res, ligand_local_res])
                confidence = cryoem.resolution_to_confidence(all_local_res)
                
                protein_confidence = confidence[:len(protein_local_res)]
                ligand_confidence = confidence[len(protein_local_res):]
                
                uncertainty_source = 'local_resolution'
                print(f"    ✓ Using local resolution map")
                
            except Exception as e:
                print(f"    ⚠ Could not use local resolution map: {e}")
                print(f"    Falling back to B-factors")
                local_res_map = None
        
        # Fall back to B-factors if no local resolution
        if not local_res_map or not local_res_map.exists():
            bfactor_extractor = BFactorExtractor()
            
            all_bfactors = np.concatenate([
                atoms['protein_bfactors'],
                atoms['ligand_bfactors']
            ])
            
            confidence = bfactor_extractor.bfactor_to_confidence(
                all_bfactors,
                resolution=resolution
            )
            
            n_protein = len(atoms['protein_bfactors'])
            protein_confidence = confidence[:n_protein]
            ligand_confidence = confidence[n_protein:]
            
            uncertainty_source = 'bfactor'
        
        print(f"    ✓ Protein atoms: {len(atoms['protein_coords'])}")
        print(f"    ✓ Ligand atoms: {len(atoms['ligand_coords'])}")
        print(f"    ✓ Resolution: {resolution} Å")
        print(f"    ✓ Confidence: [{np.concatenate([protein_confidence, ligand_confidence]).min():.3f}, "
              f"{np.concatenate([protein_confidence, ligand_confidence]).max():.3f}]")
        print(f"    ✓ Source: {uncertainty_source}")
        
        return {
            'pdb_id': pdb_id,
            'method': method,
            'resolution': resolution,
            'protein_coords': atoms['protein_coords'],
            'protein_atoms': atoms['protein_atoms'],
            'protein_bfactors': atoms['protein_bfactors'],
            'protein_confidence': protein_confidence,
            'ligand_coords': atoms['ligand_coords'],
            'ligand_atoms': atoms['ligand_atoms'],
            'ligand_bfactors': atoms['ligand_bfactors'],
            'ligand_confidence': ligand_confidence,
            'uncertainty_source': uncertainty_source,
        }
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def main():
    """Main processing script."""
    print("=" * 80)
    print("Prepare Uncertainty Data for Pearl Training")
    print("=" * 80)
    
    # Process X-ray structures
    print("\n" + "=" * 80)
    print("Processing X-ray Structures")
    print("=" * 80 + "\n")
    
    xray_dir = Path("data/pdb_files")
    xray_structures = []
    
    if xray_dir.exists():
        pdb_files = list(xray_dir.glob("*.pdb"))
        print(f"Found {len(pdb_files)} X-ray structures\n")
        
        for pdb_file in pdb_files:
            pdb_id = pdb_file.stem
            result = process_xray_structure(pdb_file, pdb_id)
            if result:
                xray_structures.append(result)
            print()
    else:
        print("No X-ray structures found\n")
    
    # Process CryoEM structures
    print("=" * 80)
    print("Processing CryoEM Structures")
    print("=" * 80 + "\n")
    
    cryoem_dir = Path("data/cryoem_pdb_files")
    map_dir = Path("data/cryoem_maps")
    cryoem_structures = []
    
    if cryoem_dir.exists():
        pdb_files = list(cryoem_dir.glob("*.pdb"))
        print(f"Found {len(pdb_files)} CryoEM structures\n")
        
        for pdb_file in pdb_files:
            pdb_id = pdb_file.stem
            
            # Look for local resolution map
            local_res_map = None
            if map_dir.exists():
                # Try to find corresponding map
                for map_file in map_dir.glob(f"*{pdb_id}*local*.mrc"):
                    local_res_map = map_file
                    break
            
            result = process_cryoem_structure(pdb_file, pdb_id, local_res_map)
            if result:
                cryoem_structures.append(result)
            print()
    else:
        print("No CryoEM structures found\n")
    
    # Combine datasets
    all_structures = xray_structures + cryoem_structures
    
    # Save processed data
    output_dir = Path("data/uncertainty_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = output_dir / "structures_with_uncertainty.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(all_structures, f)
    
    # Save metadata
    metadata = {
        'total_structures': len(all_structures),
        'xray_structures': len(xray_structures),
        'cryoem_structures': len(cryoem_structures),
        'structures': [
            {
                'pdb_id': s['pdb_id'],
                'method': s['method'],
                'resolution': s['resolution'],
                'n_protein_atoms': len(s['protein_coords']),
                'n_ligand_atoms': len(s['ligand_coords']),
                'uncertainty_source': s['uncertainty_source'],
                'confidence_range': [
                    float(np.concatenate([s['protein_confidence'], s['ligand_confidence']]).min()),
                    float(np.concatenate([s['protein_confidence'], s['ligand_confidence']]).max()),
                ],
            }
            for s in all_structures
        ]
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total structures: {len(all_structures)}")
    print(f"  X-ray structures: {len(xray_structures)}")
    print(f"  CryoEM structures: {len(cryoem_structures)}")
    print(f"\nOutput:")
    print(f"  Data: {data_file}")
    print(f"  Metadata: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Run: python scripts/train_with_uncertainty_full.py")
    print("2. Monitor training with W&B dashboard")
    print("3. Compare with baseline training")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

