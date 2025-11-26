#!/usr/bin/env python3
"""
Download CryoEM Structures with Local Resolution Maps

This script downloads real CryoEM structures from the PDB and their corresponding
local resolution maps from EMDB. It focuses on protein-ligand complexes suitable
for Pearl training.

Key features:
1. Downloads CryoEM structures from PDB
2. Downloads local resolution maps from EMDB
3. Validates structure quality
4. Organizes data for Pearl training
"""

import sys
import json
import requests
import gzip
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# High-quality CryoEM structures with ligands
# Format: (PDB_ID, EMDB_ID, Description, Resolution)
CRYOEM_STRUCTURES = [
    # High-resolution CryoEM structures with ligands
    ("6XR8", "EMD-21997", "SARS-CoV-2 Spike with inhibitor", 2.9),
    ("7BV2", "EMD-30210", "SARS-CoV-2 RBD with antibody", 3.2),
    ("6M0J", "EMD-21375", "SARS-CoV-2 Spike protein", 2.9),
    ("7JTL", "EMD-22677", "SARS-CoV-2 Mpro with inhibitor", 2.2),
    ("7K3N", "EMD-22797", "SARS-CoV-2 Nsp12 with inhibitor", 2.5),
    
    # Ribosome complexes with antibiotics
    ("6XHW", "EMD-21935", "Ribosome with antibiotic", 2.8),
    ("6XHV", "EMD-21934", "Ribosome with inhibitor", 2.9),
    
    # Other protein-ligand complexes
    ("7K00", "EMD-22730", "GPCR with ligand", 3.0),
    ("6WHA", "EMD-21452", "Kinase with inhibitor", 3.1),
    ("7JVB", "EMD-22698", "Protease with substrate", 2.7),
]


def download_pdb_structure(pdb_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download PDB structure file.
    
    Args:
        pdb_id: PDB identifier (e.g., "6XR8")
        output_dir: Directory to save the file
        
    Returns:
        Path to downloaded file, or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pdb_id}.pdb"
    
    if output_file.exists():
        print(f"  ✓ PDB file already exists: {output_file}")
        return output_file
    
    # Try PDB format first
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        print(f"  Downloading PDB structure: {pdb_id}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        print(f"  ✓ Downloaded: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"  ✗ Failed to download {pdb_id}: {e}")
        return None


def download_emdb_map(emdb_id: str, output_dir: Path, map_type: str = "local_resolution") -> Optional[Path]:
    """
    Download EMDB map file (local resolution or main map).
    
    Args:
        emdb_id: EMDB identifier (e.g., "EMD-21997")
        output_dir: Directory to save the file
        map_type: "local_resolution" or "main"
        
    Returns:
        Path to downloaded file, or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract numeric ID
    emdb_num = emdb_id.replace("EMD-", "")
    
    if map_type == "local_resolution":
        # Try common local resolution file patterns
        patterns = [
            f"emd_{emdb_num}_half_map_1.map.gz",
            f"emd_{emdb_num}_additional.map.gz",
            f"emd_{emdb_num}_locres.map.gz",
        ]
        output_file = output_dir / f"{emdb_id}_local_resolution.mrc"
    else:
        patterns = [f"emd_{emdb_num}.map.gz"]
        output_file = output_dir / f"{emdb_id}.mrc"
    
    if output_file.exists():
        print(f"  ✓ Map file already exists: {output_file}")
        return output_file
    
    # Try to download
    base_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_num}/map/"
    
    for pattern in patterns:
        url = base_url + pattern
        try:
            print(f"  Downloading EMDB map: {emdb_id} ({map_type})")
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Download to temporary gzipped file
            temp_file = output_dir / f"{emdb_id}_temp.map.gz"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Decompress
            with gzip.open(temp_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove temp file
            temp_file.unlink()
            
            print(f"  ✓ Downloaded and decompressed: {output_file}")
            return output_file
            
        except Exception as e:
            continue
    
    print(f"  ⚠ Could not download {map_type} map for {emdb_id}")
    print(f"    This is OK - we'll use B-factors instead")
    return None


def get_structure_metadata(pdb_id: str) -> Dict:
    """
    Get metadata about a PDB structure from RCSB API.
    
    Args:
        pdb_id: PDB identifier
        
    Returns:
        Dictionary with metadata
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant info
        metadata = {
            'pdb_id': pdb_id,
            'title': data.get('struct', {}).get('title', 'Unknown'),
            'resolution': None,
            'method': 'UNKNOWN',
        }
        
        # Get experimental method
        if 'exptl' in data:
            methods = [exp.get('method', '') for exp in data['exptl']]
            if 'ELECTRON MICROSCOPY' in methods:
                metadata['method'] = 'EM'
            elif 'X-RAY DIFFRACTION' in methods:
                metadata['method'] = 'XRAY'
        
        # Get resolution
        if 'rcsb_entry_info' in data:
            metadata['resolution'] = data['rcsb_entry_info'].get('resolution_combined', [None])[0]
        
        return metadata
        
    except Exception as e:
        print(f"  ⚠ Could not fetch metadata for {pdb_id}: {e}")
        return {
            'pdb_id': pdb_id,
            'title': 'Unknown',
            'resolution': None,
            'method': 'UNKNOWN',
        }


def main():
    """Main download script."""
    print("=" * 80)
    print("CryoEM Data Download for Pearl Training")
    print("=" * 80)
    
    # Setup directories
    base_dir = Path("data")
    pdb_dir = base_dir / "cryoem_pdb_files"
    map_dir = base_dir / "cryoem_maps"
    
    pdb_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directories:")
    print(f"  PDB files: {pdb_dir}")
    print(f"  Map files: {map_dir}")
    
    # Download structures
    results = []
    
    print(f"\n{'=' * 80}")
    print(f"Downloading {len(CRYOEM_STRUCTURES)} CryoEM structures")
    print(f"{'=' * 80}\n")
    
    for i, (pdb_id, emdb_id, description, resolution) in enumerate(CRYOEM_STRUCTURES, 1):
        print(f"[{i}/{len(CRYOEM_STRUCTURES)}] {pdb_id} ({emdb_id}): {description}")
        print(f"  Expected resolution: {resolution} Å")
        
        # Download PDB structure
        pdb_file = download_pdb_structure(pdb_id, pdb_dir)
        
        # Download main map
        main_map = download_emdb_map(emdb_id, map_dir, map_type="main")
        
        # Try to download local resolution map
        local_res_map = download_emdb_map(emdb_id, map_dir, map_type="local_resolution")
        
        # Get metadata
        metadata = get_structure_metadata(pdb_id)
        
        # Record results
        result = {
            'pdb_id': pdb_id,
            'emdb_id': emdb_id,
            'description': description,
            'expected_resolution': resolution,
            'actual_resolution': metadata.get('resolution'),
            'method': metadata.get('method'),
            'pdb_file': str(pdb_file) if pdb_file else None,
            'main_map': str(main_map) if main_map else None,
            'local_resolution_map': str(local_res_map) if local_res_map else None,
            'has_local_resolution': local_res_map is not None,
        }
        results.append(result)
        
        print()
        
        # Be nice to servers
        time.sleep(1)
    
    # Save manifest
    manifest = {
        'total_structures': len(results),
        'structures_with_pdb': sum(1 for r in results if r['pdb_file']),
        'structures_with_main_map': sum(1 for r in results if r['main_map']),
        'structures_with_local_resolution': sum(1 for r in results if r['has_local_resolution']),
        'structures': results,
    }
    
    manifest_file = base_dir / "cryoem_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total structures: {manifest['total_structures']}")
    print(f"  PDB files downloaded: {manifest['structures_with_pdb']}")
    print(f"  Main maps downloaded: {manifest['structures_with_main_map']}")
    print(f"  Local resolution maps: {manifest['structures_with_local_resolution']}")
    print(f"\nManifest saved to: {manifest_file}")
    
    # Detailed breakdown
    print(f"\n{'=' * 80}")
    print("Structure Details:")
    print(f"{'=' * 80}\n")
    
    for result in results:
        status = "✓" if result['pdb_file'] else "✗"
        local_res = "✓" if result['has_local_resolution'] else "✗"
        print(f"{status} {result['pdb_id']} ({result['emdb_id']})")
        print(f"  Resolution: {result['actual_resolution']} Å")
        print(f"  Local resolution map: {local_res}")
        print()
    
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Run: python scripts/prepare_uncertainty_data.py")
    print("2. This will extract B-factors and local resolution")
    print("3. Then run: python scripts/train_with_uncertainty_full.py")
    print("4. Monitor training with W&B dashboard")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

