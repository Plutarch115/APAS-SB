"""
Download all datasets for APAS-SB training.

Downloads:
1. Boltz-2 datasets: ChEMBL, BindingDB, PubChem, CeMM, MIDAS
2. Original datasets: PDBbind, SKEMPI 2.0, BRENDA, ProteinGym
3. MD trajectory databases: mdCATH, ATLAS

Aligned with APAS-SB_Development_Roadmap.md Phase 1 (Days 1-12).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import requests
from tqdm import tqdm


class DatasetDownloader:
    """Manages dataset downloads for APAS-SB"""
    
    def __init__(self, output_dir: str = './data', parallel_workers: int = 8):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_workers = parallel_workers
    
    def download_chembl(self):
        """Download ChEMBL database"""
        print("\n" + "=" * 60)
        print("Downloading ChEMBL v34")
        print("=" * 60)
        
        chembl_dir = self.output_dir / 'chembl'
        chembl_dir.mkdir(exist_ok=True)
        
        # ChEMBL download URL (v34)
        url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz"
        
        print(f"Source: {url}")
        print(f"Target: {chembl_dir}")
        print("\nNote: ChEMBL is ~4 GB compressed, ~20 GB uncompressed")
        print("This will take 10-30 minutes depending on connection speed.")
        print("\nTo download manually:")
        print(f"  wget {url} -P {chembl_dir}")
        print(f"  tar -xzf {chembl_dir}/chembl_34_sqlite.tar.gz -C {chembl_dir}")
        
        return chembl_dir
    
    def download_bindingdb(self):
        """Download BindingDB"""
        print("\n" + "=" * 60)
        print("Downloading BindingDB")
        print("=" * 60)
        
        bindingdb_dir = self.output_dir / 'bindingdb'
        bindingdb_dir.mkdir(exist_ok=True)
        
        url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_202401.tsv.zip"
        
        print(f"Source: {url}")
        print(f"Target: {bindingdb_dir}")
        print("\nNote: BindingDB is ~2 GB compressed, ~10 GB uncompressed")
        print("\nTo download manually:")
        print(f"  wget {url} -P {bindingdb_dir}")
        print(f"  unzip {bindingdb_dir}/BindingDB_All_202401.tsv.zip -d {bindingdb_dir}")
        
        return bindingdb_dir
    
    def download_pubchem(self):
        """Download PubChem datasets"""
        print("\n" + "=" * 60)
        print("Downloading PubChem Datasets")
        print("=" * 60)
        
        pubchem_dir = self.output_dir / 'pubchem'
        pubchem_dir.mkdir(exist_ok=True)
        
        print(f"Target: {pubchem_dir}")
        print("\nPubChem datasets require custom queries via their API:")
        print("1. PubChem HTS: High-throughput screening assays")
        print("2. PubChem Small Assays: Smaller, higher-quality assays")
        print("\nAPI Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest")
        print("\nExample query for bioassays:")
        print("  https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/<AID>/JSON")
        print("\nRecommended: Use PubChem PUG REST API with Python requests")
        
        return pubchem_dir
    
    def download_pdbind(self):
        """Download PDBbind database"""
        print("\n" + "=" * 60)
        print("Downloading PDBbind")
        print("=" * 60)
        
        pdbind_dir = self.output_dir / 'pdbind'
        pdbind_dir.mkdir(exist_ok=True)
        
        print(f"Target: {pdbind_dir}")
        print("\nPDBbind requires registration at: http://www.pdbbind.org.cn/")
        print("After registration, download:")
        print("  - PDBbind v2020 (refined set): ~5 GB")
        print("  - General set (optional): ~20 GB")
        
        return pdbind_dir
    
    def download_skempi2(self):
        """Download SKEMPI 2.0"""
        print("\n" + "=" * 60)
        print("Downloading SKEMPI 2.0")
        print("=" * 60)
        
        skempi_dir = self.output_dir / 'skempi2'
        skempi_dir.mkdir(exist_ok=True)
        
        url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
        
        print(f"Source: {url}")
        print(f"Target: {skempi_dir}")
        print("\nTo download manually:")
        print(f"  wget {url} -P {skempi_dir}")
        
        return skempi_dir
    
    def download_brenda(self):
        """Download BRENDA"""
        print("\n" + "=" * 60)
        print("Downloading BRENDA")
        print("=" * 60)
        
        brenda_dir = self.output_dir / 'brenda'
        brenda_dir.mkdir(exist_ok=True)
        
        print(f"Target: {brenda_dir}")
        print("\nBRENDA requires registration at: https://www.brenda-enzymes.org/")
        print("After registration, download the text file database")
        print("Size: ~500 MB compressed")
        
        return brenda_dir
    
    def download_proteingym(self):
        """Download ProteinGym"""
        print("\n" + "=" * 60)
        print("Downloading ProteinGym")
        print("=" * 60)
        
        proteingym_dir = self.output_dir / 'proteingym'
        proteingym_dir.mkdir(exist_ok=True)
        
        print(f"Target: {proteingym_dir}")
        print("\nProteinGym GitHub: https://github.com/OATML-Markslab/ProteinGym")
        print("Download DMS substitutions benchmark:")
        print("  git clone https://github.com/OATML-Markslab/ProteinGym.git")
        print("  cd ProteinGym && ./download_DMS_data.sh")
        print("\nSize: ~2 GB")

        return proteingym_dir

    def download_mdcath(self):
        """Download mdCATH MD trajectory database"""
        print("\n" + "=" * 60)
        print("Downloading mdCATH (Priority #1 - Critical Path)")
        print("=" * 60)

        mdcath_dir = self.output_dir / 'mdcath'
        mdcath_dir.mkdir(exist_ok=True)

        print(f"Target: {mdcath_dir}")
        print("\nmdCATH: 134,950 MD trajectories from HuggingFace")
        print("Size: ~3 TB")
        print("Time: 1-3 days with high-bandwidth connection")
        print("\nHuggingFace: https://huggingface.co/datasets/compsciencelab/mdCATH")
        print("GitHub: https://github.com/compsciencelab/mdCATH")
        print("Paper: https://doi.org/10.1038/s41597-024-04140-z")

        print("\n" + "-" * 60)
        print("RECOMMENDED DOWNLOAD METHOD:")
        print("-" * 60)
        print("\n# Method 1: HuggingFace CLI (Fastest)")
        print("pip install huggingface_hub[cli]")
        print("huggingface-cli login  # Optional, for faster speeds")
        print(f"huggingface-cli download compsciencelab/mdCATH \\")
        print(f"    --repo-type dataset \\")
        print(f"    --local-dir {mdcath_dir} \\")
        print(f"    --resume-download")

        print("\n# Method 2: Python API")
        print("from huggingface_hub import snapshot_download")
        print("snapshot_download(")
        print("    repo_id='compsciencelab/mdCATH',")
        print("    repo_type='dataset',")
        print(f"    local_dir='{mdcath_dir}',")
        print("    resume_download=True,")
        print("    max_workers=8")
        print(")")

        print("\n# Method 3: TorchMD-Net integration (for training)")
        print("from torchmdnet.datasets import mdCATH")
        print("dataset = mdCATH(")
        print(f"    root='{mdcath_dir}',")
        print("    temperature=320,  # Options: 320, 350, 380, 410, 450 K")
        print("    split='train',")
        print("    download=True")
        print(")")

        return mdcath_dir

    def download_atlas(self):
        """Download ATLAS MD trajectory database"""
        print("\n" + "=" * 60)
        print("Downloading ATLAS (Priority #2)")
        print("=" * 60)

        atlas_dir = self.output_dir / 'atlas'
        atlas_dir.mkdir(exist_ok=True)

        print(f"Target: {atlas_dir}")
        print("\nATLAS: 1,390 proteins × 3 replicates = 4,170 trajectories")
        print("Size: ~500 GB (reduced format) or ~15 TB (full format)")
        print("Time: 1-2 days")
        print("\nWebsite: https://www.dsimb.inserm.fr/ATLAS")
        print("Download page: https://www.dsimb.inserm.fr/ATLAS/download.html")
        print("Paper: https://doi.org/10.1093/nar/gkad1084")

        print("\n" + "-" * 60)
        print("RECOMMENDED DOWNLOAD METHOD:")
        print("-" * 60)
        print("\n# Step 1: Download the ATLAS download script")
        print("wget https://www.dsimb.inserm.fr/ATLAS/download_ATLAS.py")

        print("\n# Step 2: Install aria2c for parallel downloads")
        print("# macOS: brew install aria2")
        print("# Linux: sudo apt-get install aria2")

        print("\n# Step 3: Run bulk download (reduced format recommended)")
        print("python download_ATLAS.py \\")
        print(f"    --output-dir {atlas_dir} \\")
        print("    --format reduced \\")
        print("    --parallel 16")

        print("\n# Alternative: API access for programmatic download")
        print("import requests")
        print("proteins = requests.get(")
        print("    'https://www.dsimb.inserm.fr/ATLAS/api/proteins'")
        print(").json()")

        return atlas_dir

    def download_all(self, datasets: Optional[List[str]] = None):
        """Download all or specified datasets"""
        print("\n" + "🚀" * 40)
        print("APAS-SB DATASET DOWNLOADER")
        print("Aligned with Development Roadmap Phase 1 (Days 1-12)")
        print("🚀" * 40)

        all_datasets = {
            'mdcath': self.download_mdcath,
            'atlas': self.download_atlas,
            'chembl': self.download_chembl,
            'bindingdb': self.download_bindingdb,
            'pubchem': self.download_pubchem,
            'pdbind': self.download_pdbind,
            'skempi2': self.download_skempi2,
            'brenda': self.download_brenda,
            'proteingym': self.download_proteingym,
        }

        if datasets is None:
            datasets = list(all_datasets.keys())

        print(f"\nOutput directory: {self.output_dir}")
        print(f"Datasets to download: {', '.join(datasets)}")
        print(f"Parallel workers: {self.parallel_workers}")

        for dataset_name in datasets:
            if dataset_name in all_datasets:
                all_datasets[dataset_name]()
            else:
                print(f"\n⚠️  Unknown dataset: {dataset_name}")

        print("\n" + "=" * 60)
        print("✅ DOWNLOAD INSTRUCTIONS COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Follow the instructions above to download each dataset")
        print("2. Run preprocessing scripts to prepare data for training")
        print("3. Verify data integrity with validation scripts")
        print("\nEstimated total download time: 3-7 days")
        print("Estimated total storage: ~4-5 TB")


def main():
    parser = argparse.ArgumentParser(description='Download APAS-SB datasets')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to download (default: all)')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel download workers')

    args = parser.parse_args()

    downloader = DatasetDownloader(
        output_dir=args.output_dir,
        parallel_workers=args.parallel
    )

    downloader.download_all(datasets=args.datasets)


if __name__ == '__main__':
    main()

