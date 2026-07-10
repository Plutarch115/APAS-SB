#!/usr/bin/env bash
#
# End-to-end smoke test for the real BindingDB + ESM2/MolFormer integration.
#
# What it does:
#   1. Creates an isolated uv venv (.venv) if one does not exist and installs
#      the pinned dependency stack (torch, transformers, fair-esm, rdkit, ...).
#   2. Streams a tiny slice of the raw BindingDB_All.tsv into a processed CSV.
#   3. Precomputes real ESM2 (protein) + MolFormer (ligand) embeddings and runs
#      a short training loop through scripts/train_with_wandb.py (W&B disabled).
#
# A "SMOKE TEST PASSED" line at the end means the full pipeline works: real data
# in, real pretrained embeddings, forward/backward pass, no synthetic data.
#
# Usage:
#   bash scripts/test_bindingdb_integration.sh
#
# Configurable via environment variables (all optional):
#   BINDINGDB_TSV   path to raw BindingDB_All.tsv
#                   (default: the shared cluster path)
#   ESM2_MODEL      fair-esm model name (default: esm2_t6_8M_UR50D, small & fast;
#                   production uses esm2_t33_650M_UR50D)
#   MAX_ROWS        rows to keep from the TSV        (default: 200)
#   MAX_SAMPLES     samples to actually train on     (default: 60)
#   CUDA_VISIBLE_DEVICES  GPU to use                 (default: 0)
#   TORCH_INDEX_URL torch wheel index (default: cu124 build)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="${VENV:-$PROJECT_ROOT/.venv}"
PY="$VENV/bin/python"
BINDINGDB_TSV="${BINDINGDB_TSV:-/nfs/lambda_stor_01/data/avasan/APAS-SB/APAS-SB/data/bindingdb/BindingDB_All.tsv}"
ESM2_MODEL="${ESM2_MODEL:-esm2_t6_8M_UR50D}"
MAX_ROWS="${MAX_ROWS:-200}"
MAX_SAMPLES="${MAX_SAMPLES:-60}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "=================================================================="
echo " BindingDB + ESM2/MolFormer integration smoke test"
echo " project     : $PROJECT_ROOT"
echo " venv        : $VENV"
echo " tsv         : $BINDINGDB_TSV"
echo " esm2 model  : $ESM2_MODEL"
echo " rows/samples: $MAX_ROWS / $MAX_SAMPLES"
echo "=================================================================="

# ------------------------------------------------------------------ 1. venv
echo ""
echo ">> [1/4] Ensuring isolated uv venv + dependencies"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/getting-started/"
    exit 1
fi
if [ ! -x "$PY" ]; then
    uv venv --python 3.10 "$VENV"
fi
# Torch first (from the CUDA wheel index), then the rest from PyPI.
VIRTUAL_ENV="$VENV" uv pip install "torch==2.6.0" --index-url "$TORCH_INDEX_URL"
VIRTUAL_ENV="$VENV" uv pip install \
    "transformers==4.53.0" tokenizers huggingface_hub accelerate \
    fair-esm rdkit "numpy<2" pandas pyyaml scipy sentencepiece einops \
    biopython scikit-learn h5py wandb
# NOTE: torchvision is intentionally NOT installed. transformers then treats it
# as unavailable and skips a vision import path that is broken by a torch/
# torchvision ABI mismatch in some base environments.

# ------------------------------------------------------------ 2. processed CSV
echo ""
echo ">> [2/4] Building a tiny processed BindingDB CSV"
if [ ! -f "$BINDINGDB_TSV" ]; then
    echo "ERROR: BindingDB TSV not found at: $BINDINGDB_TSV"
    echo "       Set BINDINGDB_TSV=/path/to/BindingDB_All.tsv and re-run."
    exit 1
fi
CSV="data/bindingdb/test_processed.csv"
"$PY" scripts/prepare_bindingdb.py \
    --tsv "$BINDINGDB_TSV" --out "$CSV" --max-rows "$MAX_ROWS" --chunksize 50000

# ------------------------------------------------------------ 3. test config
echo ""
echo ">> [3/4] Writing test config"
CFG="$(mktemp /tmp/apas_test_config.XXXXXX.yaml)"
trap 'rm -f "$CFG"' EXIT
cat > "$CFG" <<EOF
wandb:
  project: "apas-sb-smoketest"
data_root: "./data"
use_synthetic: false
bindingdb:
  processed_csv: "./${CSV}"
  max_samples: ${MAX_SAMPLES}
features:
  mode: esm2_molformer
  esm2_model: ${ESM2_MODEL}
  molformer_model: ibm/MoLFormer-XL-both-10pct
  max_protein_len: 128
  max_ligand_len: 48
  emb_cache_dir: "./data/bindingdb/emb_cache_test"
  precompute: true
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout: 0.1
  pair_dim: 64
training:
  phase_2a:
    datasets: [bindingdb]
    batch_size_per_gpu: 4
    num_epochs: 1
    learning_rate: 1.0e-4
    weight_decay: 0.01
log_interval: 5
checkpoint_interval: 100
checkpoint_dir: "./checkpoints_test"
EOF

# ------------------------------------------------------------ 4. run training
echo ""
echo ">> [4/4] Running end-to-end training with real ESM2 + MolFormer features"
WANDB_MODE=disabled "$PY" scripts/train_with_wandb.py --config "$CFG" --phase 2a

echo ""
echo "=================================================================="
echo " ✅ SMOKE TEST PASSED"
echo " Real BindingDB data was tokenized with ESM2 (protein) + MolFormer"
echo " (ligand) and trained end-to-end with no synthetic data."
echo "=================================================================="
