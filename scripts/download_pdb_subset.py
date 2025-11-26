#!/usr/bin/env python3
"""
Download a curated subset of PDB structures with protein-ligand complexes.

This script downloads well-characterized protein-ligand complexes suitable
for testing Pearl's data pipeline and training.
"""

import os
import sys
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import List, Dict

# Curated list of PDB structures with high-quality protein-ligand complexes
# These are well-studied structures with diverse proteins and ligands
CURATED_PDB_IDS = [
    # Kinases with inhibitors
    "1ATP",  # cAMP-dependent protein kinase with ATP
    "3PY0",  # CDK2 with inhibitor
    "4HNF",  # ABL kinase with imatinib
    
    # Proteases with inhibitors
    "1HVR",  # HIV-1 protease with inhibitor
    "3CL5",  # SARS-CoV main protease
    "1W1P",  # Thrombin with inhibitor
    
    # Nuclear receptors
    "3ERT",  # Estrogen receptor with ligand
    "1M2Z",  # Androgen receptor with ligand
    
    # Enzymes with substrates/inhibitors
    "1HWW",  # Carbonic anhydrase with inhibitor
    "1E66",  # Acetylcholinesterase with inhibitor
    "2ZNL",  # Neuraminidase with inhibitor
    
    # GPCRs and membrane proteins
    "4EIY",  # Beta-2 adrenergic receptor
    "5C1M",  # Adenosine A2A receptor
    
    # Other drug targets
    "1XKK",  # Factor Xa with inhibitor
    "2BRC",  # BACE1 with inhibitor
]

# PDB download URLs
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"
PDB_REST_API = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def download_pdb_structure(pdb_id: str, output_dir: Path) -> bool:
    """
    Download a PDB structure file.
    
    Args:
        pdb_id: PDB identifier (e.g., "1ATP")
        output_dir: Directory to save the PDB file
        
    Returns:
        True if successful, False otherwise
    """
    pdb_id = pdb_id.upper()
    output_file = output_dir / f"{pdb_id}.pdb"
    
    # Skip if already exists
    if output_file.exists():
        print(f"  ✓ {pdb_id} already exists")
        return True
    
    try:
        # Download gzipped PDB file
        url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
        print(f"  Downloading {pdb_id} from {url}...")
        
        gz_file = output_dir / f"{pdb_id}.pdb.gz"
        urllib.request.urlretrieve(url, gz_file)
        
        # Decompress
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove gzipped file
        gz_file.unlink()
        
        print(f"  ✓ {pdb_id} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download {pdb_id}: {e}")
        # Clean up partial files
        if gz_file.exists():
            gz_file.unlink()
        if output_file.exists():
            output_file.unlink()
        return False


def verify_pdb_file(pdb_file: Path) -> Dict:
    """
    Verify PDB file and extract basic information.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Dictionary with structure information
    """
    info = {
        'pdb_id': pdb_file.stem,
        'has_protein': False,
        'has_ligand': False,
        'n_atoms': 0,
        'n_residues': 0,
        'ligands': [],
    }
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    info['has_protein'] = True
                    info['n_atoms'] += 1
                elif line.startswith('HETATM'):
                    # Check if it's a ligand (not water/ion)
                    res_name = line[17:20].strip()
                    if res_name not in ['HOH', 'WAT', 'NA', 'CL', 'CA', 'MG', 'ZN', 'FE']:
                        info['has_ligand'] = True
                        if res_name not in info['ligands']:
                            info['ligands'].append(res_name)
                    info['n_atoms'] += 1
        
        return info
        
    except Exception as e:
        print(f"  ✗ Error verifying {pdb_file}: {e}")
        return info


def main():
    """Download PDB subset and verify files."""
    print("=" * 80)
    print("Pearl PDB Data Downloader")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("data/pdb_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    
    # Download structures
    print(f"\n📥 Downloading {len(CURATED_PDB_IDS)} PDB structures...")
    print("-" * 80)
    
    successful = []
    failed = []
    
    for pdb_id in CURATED_PDB_IDS:
        if download_pdb_structure(pdb_id, output_dir):
            successful.append(pdb_id)
        else:
            failed.append(pdb_id)
    
    # Verify downloaded files
    print("\n" + "=" * 80)
    print("📊 Verifying downloaded structures...")
    print("-" * 80)
    
    valid_structures = []
    
    for pdb_id in successful:
        pdb_file = output_dir / f"{pdb_id}.pdb"
        info = verify_pdb_file(pdb_file)
        
        if info['has_protein'] and info['has_ligand']:
            valid_structures.append(info)
            print(f"✓ {pdb_id}: {info['n_atoms']} atoms, ligands: {', '.join(info['ligands'])}")
        else:
            print(f"⚠ {pdb_id}: Missing protein or ligand")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Successfully downloaded: {len(successful)}/{len(CURATED_PDB_IDS)}")
    print(f"✓ Valid protein-ligand complexes: {len(valid_structures)}")
    
    if failed:
        print(f"✗ Failed downloads: {len(failed)}")
        print(f"  {', '.join(failed)}")
    
    print(f"\n📁 PDB files saved to: {output_dir.absolute()}")
    
    # Save metadata
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("PDB Structures for Pearl Training\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total structures: {len(valid_structures)}\n\n")
        for info in valid_structures:
            f.write(f"{info['pdb_id']}: {info['n_atoms']} atoms, "
                   f"ligands: {', '.join(info['ligands'])}\n")
    
    print(f"📄 Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("✅ Download complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python scripts/generate_synthetic_data.py")
    print("  2. Run: python scripts/prepare_training_data.py")
    print("  3. Run: python scripts/train_pearl.py")
    print("=" * 80 + "\n")
    
    return 0 if len(valid_structures) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

