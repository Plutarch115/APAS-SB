# mdCATH streaming dataloader (torchmd-net, no download)

Stream the [mdCATH dataset](https://huggingface.co/datasets/compsciencelab/mdCATH)
directly from Hugging Face using **HDF5 range reads** — you never download the
1.5 GB-per-domain files (let alone the 3.3 TB whole set). Built on the **official
torchmd-net `MDCATH`** class, with **sampling under your control**.

## How it avoids downloading (verified)
mdCATH files are HDF5. Opening a *remote* file object (`HfFileSystem`) instead of
a local path makes h5py issue HTTP range requests and fetch only the b-tree
metadata + the exact chunks for the frame you index.

Measured here on `12asA00` (a **1508.9 MB** file):
```
bytes to read metadata:    0.046 MB
bytes for 1 more frame:    0.122 MB
full coords+forces traj:  53.8 MB   <- what a naive read would pull
file total size:        1508.9 MB   <- what a download would pull
```
So one frame ≈ **0.17 MB over the wire** instead of 1.5 GB.

## What's reused vs. added
- **Reused unchanged:** `torchmdnet_mdcath.py` is the upstream torchmd-net module,
  vendored verbatim so you don't need a full torchmd-net install. If you have it
  installed, `mdcath_streaming.py` auto-imports `torchmdnet.datasets.mdcath.MDCATH`
  instead of the vendored copy.
- **Added:** `mdcath_streaming.py`:
  - `StreamingMDCATH(MDCATH)` — subclass overriding ONLY the file-access seams:
    - `download()` -> no-op (nothing bulk-fetched)
    - `_ensure_source_file()` -> fetch the ~186 MB `mdcath_source.h5` index once
      (metadata index, not trajectories) or stream it
    - `calculate_dataset_size()` -> remote metadata stat, no local files
    - `process_specific_group()` -> read the frame slice from a cached, streamed
      h5py handle (range reads), reusing the upstream read logic
  - `PerSequenceBatchSampler` — YOUR sampling policy (see below)

## Files
- `mdcath_streaming.py` — streaming subclass + custom sampler + runnable `demo()`
- `torchmdnet_mdcath.py` — vendored upstream torchmd-net MDCATH (unmodified)
- `test_sampler.py` — offline unit test of the sampling-control surface
- `environment.yaml` / `requirements.txt` — CUDA 11.8 / V100 setup

## Sampling control
The base class's `self.idx` is a fixed flat conformer catalog — itself a sampling
policy you didn't choose. This module hands it to you:

```python
ds = StreamingMDCATH(root="./idx_cache", pdb_list=["12asA00","1r9lA02"],
                     temperatures=["348"], numAtoms=None)   # numAtoms=None: no size filter

catalog = ds.build_catalog()          # list of (pdb, file, temp, replica, conf_idx)
by_dom  = ds.group_indices_by("domain")        # {domain: [dataset_idx,...]}
by_dt   = ds.group_indices_by("domain_temp")   # keep temperature constant too
```
Then drive a `DataLoader` with any sampler you write over those indices. Provided:

```python
from torch_geometric.loader import DataLoader
from mdcath_streaming import StreamingMDCATH, PerSequenceBatchSampler

sampler = PerSequenceBatchSampler(
    ds, batch_size=32,
    group_key="domain_temp",     # 'domain' | 'temp' | 'replica' | 'domain_temp'
    shuffle=True, drop_last=False,
    max_frames_per_group=None,   # optional per-group cap per epoch
    seed=0,
)
loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)

device = torch.device("cuda")
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)               # reshuffle deterministically
    for data in loader:                    # torch_geometric Batch
        data = data.to(device, non_blocking=True)
        # data.z (ΣN,)  data.pos (ΣN,3)  data.neg_dy (ΣN,3)  data.batch (ΣN,)
        ...
```
Every batch stays within one sequence (constant protein / atom count), which is
what "per-sequence frame sampling" buys you. Want a totally different policy
(weighted by trajectory, temperature-balanced, a fixed subset)? Build your own
`batch_sampler` over `ds.build_catalog()` — the streaming layer is independent of
how you order indices.

## Environment (V100, CUDA 11.8)
PyTorch cu118 wheels bundle their CUDA runtime; you only need the NVIDIA driver
(you have 11.8). torch_geometric is required (upstream MDCATH extends its Dataset).

**Conda:**
```bash
conda env create -f environment.yaml
conda activate mdcath
```
**Or venv + pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
**Verify GPU:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.0.1+cu118 True Tesla V100-...
```
**Optional but recommended** — set an HF token to lift anonymous rate limits
(streaming makes many small range requests):
```bash
export HF_TOKEN=hf_xxx    # or: huggingface-cli login
```

## Run the demo (streams live, downloads nothing but the index)
```bash
python mdcath_streaming.py --root ./idx_cache --pdb-list 12asA00 \
    --temperatures 348 --batch-size 8 --num-atoms 999999 --max-batches 3
```
Observed output (this was actually run):
```
domains kept: ['12asA00']  conformers: 2220
batches/epoch: 278
batch 0: z=(40728,) pos=(40728, 3) neg_dy=(40728, 3) batch=(40728,) on cpu
...
done -- nothing but the index + touched frames was transferred.
```
(40728 = 8 frames x 5091 atoms.) On your box "cpu" becomes "cuda:0".

Note: `12asA00` has 5091 protein atoms, above the upstream default
`numAtoms=5000`, so it's filtered out unless you pass `numAtoms=None`
(`--num-atoms 999999` in the demo). Most domains are smaller.

## Offline test (no GPU, no network)
```bash
python test_sampler.py
```
Verifies `group_indices_by` and `PerSequenceBatchSampler` (grouping, per-sequence
batch purity, coverage, drop_last, per-group cap, determinism) against a fake
catalog.

## Verification status (honest)
Run on an aarch64 Raspberry Pi (no GPU) with CPU torch 2.13 + pyg 2.8:
- ✅ streaming primitive proven — 1 frame from a 1.5 GB file = ~0.17 MB transferred
- ✅ `StreamingMDCATH` runs end-to-end against **live** HF data, 3 per-sequence
  batches with correct shapes, exit 0
- ✅ `test_sampler.py` passes (12 checks)
- ⚠️ NOT exercised here: actual CUDA staging (no GPU on the dev box). `.to("cuda")`
  is the standard path; the `torch.cuda.is_available()` check + demo on your V100
  are the first real GPU smoke test. Multi-worker streaming (`num_workers>0`) uses
  per-process handle caches (pid-guarded) but was validated at `num_workers=0`;
  try a small `num_workers` first on your box.
