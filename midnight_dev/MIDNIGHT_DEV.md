# Midnight-Dev: Autonomous Remote Iteration Workflow for APAS-SB

**What this is.** A fenced, auditable workflow that lets a Hermes agent (running on
Joshua's Raspberry Pi) develop on APAS-SB overnight: edit code locally, push to
GitHub, pull onto the Lambda GPU cluster, run, read the logs, and iterate — all
on an isolated `midnight-dev` branch that never touches `main`.

**Status:** experimental, sanctioned by Archit as part of the internship. The goal
is to *try* and learn what an autonomous agent can safely do on shared HPC — not
to ship perfect code on the first run.

---

## The loop

```
  [Pi]  edit code  ->  git commit  ->  git push origin midnight-dev
                                              |
                                              v  (GitHub)
  [Lambda]  git reset --hard origin/midnight-dev  ->  run on 1 GPU  ->  logs
                                              |
                                              v  (logs stream back to Pi)
  [Pi]  agent reads log  ->  diagnoses  ->  edits again  ->  repeat
```

The agent owns the `[Pi]` steps. `sync_and_run.sh` performs the push + remote
sync + run + log capture in one auditable script, so "the agent" is executing a
loop you can read line-by-line, not improvising SSH commands.

---

## Guardrails (why this is safe to run unattended)

| Guardrail | Mechanism |
|---|---|
| **Never touches `main`** | All work on `midnight-dev`; Lambda `git reset --hard` targets that branch only. |
| **One GPU only** | Script parses `nvidia-smi`, picks the *emptiest* GPU, sets `CUDA_VISIBLE_DEVICES` to that single index. |
| **Good cluster citizen** | Refuses to launch if no GPU has >= 20 GB free (`GPU_MEM_FREE_MIN_MB`). |
| **No runaway jobs** | Every remote job runs under a hard `timeout` (`JOB_TIMEOUT`, default 6h). |
| **No secret handling** | The agent never types the Lambda password or taps Duo. It rides an SSH socket Joshua opened. |
| **Fail loud, not silent** | If the SSH master socket is dead, the script exits with instructions and changes nothing. |
| **Everything logged** | All output to `~/midnight_dev_logs/run_<timestamp>.log`. |

---

## The Duo problem and how ControlMaster solves it

Lambda login = **password + Duo push**. An autonomous agent can do neither.

**ControlMaster** (SSH connection multiplexing) is the answer:
1. At bedtime, **Joshua** opens ONE master connection (types password, taps Duo once).
2. That connection becomes a persistent background socket (`ControlPersist 8h`).
3. Every later `ssh lambda ...` — from the agent — **reuses that socket with no
   re-auth** until it expires or the network drops.

The agent never sees a credential. It only uses a door Joshua already opened.

### One-time setup (do once)
Append the block in `midnight_dev/ssh_config_snippet` to `~/.ssh/config` on the Pi.

### Nightly kickoff (Joshua, ~10 seconds before bed)
```bash
ssh -fN lambda        # type SSH password, tap Duo "1". Returns to prompt = socket open.
ssh -O check lambda   # should print: Master running (pid=...)
```
That's the entire "ping-yes-then-run" — the "yes" is opening this socket.

### Then the agent runs
```bash
cd ~/projects/APAS-SB
./midnight_dev/sync_and_run.sh "python scripts/train_with_wandb.py --config scripts/wandb_config_dev.yaml --phase 2a"
```

---

## Lambda facts (from Joshua's notes — VERIFY on first connect)

| Thing | Value |
|---|---|
| Login | `ssh jstamborski@lambda0.cels.anl.gov` (password + Duo) |
| Repo on Lambda | `/nfs/lambda_stor_01/data/avasan/APAS-SB/APAS-SB_Joshua/APAS-SB` |
| Modules | `module load cuda anaconda3 openmpi/4.0.2` |
| Conda env | `APAS1` **(ambiguous in notes — also seen `apas-sb` and a `.venv`; confirm on-box)** |
| GPU etiquette | `nvidia-smi` -> run only on least-loaded GPU via `CUDA_VISIBLE_DEVICES=N` |
| Known run cmd | `python scripts/train_with_wandb.py --config scripts/wandb_config_dev.yaml --phase 2a` |
| Post-run gotcha | `rm data/bindingdb/bindingdb_processed.csv` after each run (stale cache) |

> **Open question resolved on first connect:** which environment actually runs
> training (`APAS1` vs `apas-sb` vs the `.venv`). Edit `CONDA_ENV` in
> `sync_and_run.sh` once confirmed.

---

## Known limits (honest)

- **Not truly "fire and forget" yet.** The ControlMaster socket dies on a network
  outage, a Pi reboot, or after `ControlPersist` expires — then a human must re-auth.
  Keepalives prevent *idle* death only.
- **Cache gotcha** (`bindingdb_processed.csv`) is not yet automated in the runner.
- **First real task not chosen yet** — see the parent chat. Leading candidate:
  wire the working `mdcath_cuda_loader/` into a real training step.
