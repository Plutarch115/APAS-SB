"""
Precompute ESM2 (protein) + MolFormer (ligand) embeddings for the processed
BindingDB dataset and cache them to disk.

Running this once on a GPU warms the on-disk cache so that training
(train_with_wandb.py) with DataLoader workers only ever reads cached ``.npy``
files (no CUDA in workers).

Usage (from the project venv):
    ./.venv/bin/python scripts/precompute_bindingdb_embeddings.py \
        --config scripts/wandb_config.yaml
    # or override pieces:
    ./.venv/bin/python scripts/precompute_bindingdb_embeddings.py \
        --processed-csv data/bindingdb/bindingdb_processed.csv \
        --esm2-model esm2_t33_650M_UR50D \
        --max-samples 200000 --batch-size 8
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from pearl.data.multitask_datasets import BindingDBDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BindingDB embeddings")
    parser.add_argument('--config', type=str, default=None,
                        help='Optional wandb_config.yaml to read defaults from')
    parser.add_argument('--data-dir', type=str, default='data/bindingdb')
    parser.add_argument('--processed-csv', type=str, default=None)
    parser.add_argument('--tsv-path', type=str, default=None)
    parser.add_argument('--esm2-model', type=str, default=None)
    parser.add_argument('--molformer-model', type=str, default=None)
    parser.add_argument('--emb-cache-dir', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    bdb = cfg.get('bindingdb', {})
    feat = cfg.get('features', {})

    dataset = BindingDBDataset(
        data_dir=args.data_dir,
        use_synthetic=False,
        tsv_path=args.tsv_path or bdb.get('tsv_path'),
        processed_csv=args.processed_csv or bdb.get('processed_csv'),
        max_samples=args.max_samples if args.max_samples is not None else bdb.get('max_samples'),
        featurizer='esm2_molformer',
        esm2_model=args.esm2_model or feat.get('esm2_model', 'esm2_t33_650M_UR50D'),
        molformer_model=args.molformer_model or feat.get('molformer_model', 'ibm/MoLFormer-XL-both-10pct'),
        emb_cache_dir=args.emb_cache_dir or feat.get('emb_cache_dir'),
        feature_device=args.device,
    )

    print(f"Loaded {len(dataset)} BindingDB samples")
    print(f"protein_dim={dataset.protein_dim}, ligand_dim={dataset.ligand_dim}")
    dataset.precompute_embeddings(batch_size=args.batch_size)
    print("Done.")


if __name__ == '__main__':
    main()
