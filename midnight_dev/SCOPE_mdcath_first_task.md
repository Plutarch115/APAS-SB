# Scope: First Midnight-Dev Task — mdCATH Streaming → GPU Training Step

**Status:** proposed scope, awaiting Joshua's pick of tier.
**Branch:** `midnight-dev`  ·  **Runner:** `midnight_dev/sync_and_run.sh`

---

## The problem in one sentence
The `mdcath_cuda_loader/` streaming dataset works on CPU but has **never run on a
GPU**, and nothing in the repo consumes its output yet — so we don't actually know
that the mdCATH data path can drive a training step at all.

## Why this is the right FIRST task (not "integrate into PEARL")
The streaming loader emits **per-atom MD graphs** (`z`, `pos`, `neg_dy` forces,
`batch`) — molecular dynamics of a single protein over time. PEARL's runnable
training path (`train_with_wandb.py` -> MockPearl) consumes **protein+ligand
feature tensors** to predict **binding affinity**. These are different data shapes
AND different scientific tasks: mdCATH has no ligand and no affinity label. They
cannot be wired together directly. Fusing mdCATH into PEARL is a real research
question (which task? force prediction? a dynamics-aware auxiliary loss?) — not a
first overnight job. First we prove the loop runs; then we decide what it trains.

## The impedance mismatch, concretely
| | mdCATH streaming loader | PEARL runnable path (MockPearl) |
|---|---|---|
| Output | `z (ΣN,)`, `pos (ΣN,3)`, `neg_dy (ΣN,3)`, `batch (ΣN,)` | `protein_features [B,n,d]`, `ligand_features [B,m,d]` |
| Representation | per-atom point cloud + forces | padded per-residue/token feature tensors |
| Task | protein dynamics / forces | protein-ligand binding affinity |
| Natural model | equivariant force field (torchmd-net; `pearl/models/equivariant.py`) | `MockPearl` / `MultiTaskPEARL` heads |

---

## Tiered scope — pick the ceiling for the first run

### Tier 0 — GPU smoke test (lowest risk, ~30 min of run time)
**Goal:** run the loader's own `demo()` on Lambda's GPU, unchanged.
**Success:** `python mdcath_streaming.py --root ./idx_cache --pdb-list 12asA00
--temperatures 348 --batch-size 8 --num-atoms 999999 --max-batches 3` prints 3
batches with `on cuda:0` and exits 0.
**Proves:** streaming works from Lambda's network, CUDA staging works, env is sane.
**Out of scope:** any model, any training.
**Failure modes I expect:** HF rate-limiting (needs `HF_TOKEN`), torch_geometric
not in the chosen conda env, `numAtoms` filter dropping the demo domain.

### Tier 1 — one real training STEP (the recommended first target)
**Goal:** feed streamed batches into a small **equivariant** model and run one
forward + backward + optimizer step on GPU, predicting **forces** (the label
mdCATH actually provides via `neg_dy`).
**Success:** a new script `mdcath_cuda_loader/train_smoke.py` runs N steps, loss is
finite and **decreases over a handful of steps** on a single overfit domain, exits 0,
logs each step's loss.
**Proves:** the full data->GPU->model->loss->step loop is real and learnable.
**Model choice:** smallest viable — either torchmd-net's TorchMD-Net/ET if present
in the env, or a minimal EGNN-style net over `(z, pos)` predicting per-atom forces.
Decide on-box based on what's installed (verify, don't assume).
**Out of scope:** PEARL, binding affinity, multi-domain, real convergence.

### Tier 2 — documented, resumable mini-training (stretch)
**Goal:** Tier 1 + checkpointing, a tiny config, W&B (offline) logging, and a short
`README` so it's a reviewable artifact for Archit.
**Success:** trains for a fixed budget, writes a checkpoint, resumes from it, all
logged; produces a loss curve.
**Out of scope:** anything claiming scientific results.

---

## Guardrails inherited from the runner (all tiers)
- Runs only on `midnight-dev`, one auto-selected GPU, refuses if <20 GB free,
  hard wall-clock timeout, all output logged. See `MIDNIGHT_DEV.md`.
- **Overfit-one-domain first.** Tier 1/2 deliberately train on a single small
  domain so "loss goes down" is a meaningful, fast signal and we burn minimal GPU.

## Open questions to resolve on first connect (not assumptions)
1. Which conda env has `torch_geometric` + CUDA torch? (`APAS1` / `apas-sb` / a
   `.venv` — notes disagree). The runner's `CONDA_ENV` gets fixed once confirmed.
2. Is torchmd-net installed, or do we use the vendored copy + a hand-rolled tiny model?
3. Does Lambda outbound network reach huggingface.co, and is an `HF_TOKEN` available?

## What I will NOT do without you
- Touch `main`, or claim mdCATH is "integrated into PEARL."
- Present a decreasing loss on one domain as a scientific result — it's a plumbing test.
- Launch a long/multi-domain run before Tier 0+1 pass.
