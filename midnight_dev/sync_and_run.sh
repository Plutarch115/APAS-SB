#!/usr/bin/env bash
# ============================================================================
# sync_and_run.sh  --  Midnight-Dev remote iteration loop for APAS-SB
# ----------------------------------------------------------------------------
# Runs FROM the Raspberry Pi. Drives Lambda over the ControlMaster socket that
# Joshua opened at bedtime (see midnight_dev/ssh_config_snippet).
#
# What it does, in order:
#   1. Verify the SSH master socket is alive (fail loud if not -- no password).
#   2. Push the local midnight-dev branch to GitHub.
#   3. On Lambda: fetch + hard-reset the repo to origin/midnight-dev.
#   4. On Lambda: load modules, activate env, pick the LEAST-loaded GPU,
#      and run the target command under a wall-clock timeout + memory guard.
#   5. Stream all output to a timestamped local log.
#
# GUARDRAILS (deliberate, see MIDNIGHT_DEV.md):
#   - Only touches the midnight-dev branch. Never main.
#   - Runs on exactly ONE GPU, auto-selected as the emptiest.
#   - Hard wall-clock TIMEOUT so a runaway job self-terminates.
#   - Read-only preflight of GPU state; refuses to launch if none is free.
#   - Never types a password; if the socket is dead it exits with instructions.
#
# Usage:
#   ./sync_and_run.sh                 # uses DEFAULT_CMD below
#   ./sync_and_run.sh "python scripts/foo.py --bar"   # custom command
# ============================================================================

set -euo pipefail

# ---- Configuration (edit these to taste) -----------------------------------
SSH_HOST="lambda"                                    # matches ~/.ssh/config Host
REMOTE_REPO="/nfs/lambda_stor_01/data/avasan/APAS-SB/APAS-SB_Joshua/APAS-SB"
BRANCH="midnight-dev"
MODULES="cuda anaconda3 openmpi/4.0.2"               # module load ...
CONDA_ENV="APAS1"                                    # TODO: verify on-box (APAS1 vs apas-sb vs .venv)
GPU_MEM_FREE_MIN_MB=20000                            # require >=20GB free to claim a GPU
JOB_TIMEOUT="6h"                                     # hard wall-clock cap
LOG_DIR="${HOME}/midnight_dev_logs"
DEFAULT_CMD="python -c 'import torch; print(\"torch\", torch.__version__, \"cuda?\", torch.cuda.is_available())'"

RUN_CMD="${1:-$DEFAULT_CMD}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 1. Preflight: is the master socket alive? -----------------------------
log "=== Midnight-Dev run ${STAMP} ==="
if ! ssh -O check "$SSH_HOST" 2>/dev/null; then
  log "FATAL: no live SSH master connection to '$SSH_HOST'."
  log "Joshua must open one first (one Duo tap):"
  log "    ssh -fN lambda"
  log "Then re-run this script. Exiting without touching anything."
  exit 3
fi
log "SSH master socket is alive. Proceeding."

# ---- 2. Push local branch to GitHub ----------------------------------------
log "Pushing local '$BRANCH' to origin..."
git push origin "$BRANCH" 2>&1 | tee -a "$LOG"

# ---- 3+4. Remote: sync, env, GPU pick, run (all in one SSH session) --------
# Heredoc runs on Lambda. We select the emptiest GPU by parsing nvidia-smi.
log "Syncing + running on Lambda (timeout ${JOB_TIMEOUT})..."
ssh "$SSH_HOST" bash -s <<REMOTE 2>&1 | tee -a "$LOG"
set -euo pipefail
echo "[remote] host: \$(hostname)"

cd "${REMOTE_REPO}"
echo "[remote] repo: \$(pwd)"

# --- sync to the exact pushed commit (no local drift on Lambda) ---
git fetch origin "${BRANCH}"
git checkout "${BRANCH}" 2>/dev/null || git checkout -b "${BRANCH}" "origin/${BRANCH}"
git reset --hard "origin/${BRANCH}"
echo "[remote] now at: \$(git rev-parse --short HEAD) - \$(git log -1 --pretty=%s)"

# --- environment ---
module load ${MODULES}
# shellcheck disable=SC1091
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
echo "[remote] python: \$(which python)"

# --- pick the least-loaded GPU with enough free memory ---
echo "[remote] GPU state:"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
BEST_GPU=\$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -n -r | head -1 | awk -F',' '{gsub(/ /,"",\$1); gsub(/ /,"",\$2); print \$1" "\$2}')
GPU_ID=\$(echo "\$BEST_GPU" | awk '{print \$1}')
GPU_FREE=\$(echo "\$BEST_GPU" | awk '{print \$2}')
echo "[remote] emptiest GPU: index=\$GPU_ID free=\${GPU_FREE}MB"

if [ "\$GPU_FREE" -lt "${GPU_MEM_FREE_MIN_MB}" ]; then
  echo "[remote] REFUSING to run: no GPU with >=${GPU_MEM_FREE_MIN_MB}MB free. Being a good cluster citizen."
  exit 4
fi
export CUDA_VISIBLE_DEVICES=\$GPU_ID
echo "[remote] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

# --- run the target under a hard wall-clock cap ---
echo "[remote] ===== BEGIN JOB ====="
timeout ${JOB_TIMEOUT} bash -c '${RUN_CMD}'
echo "[remote] ===== END JOB (exit \$?) ====="
REMOTE

RC=${PIPESTATUS[0]}
log "Remote run finished with exit code ${RC}. Log: ${LOG}"
exit "$RC"
