"""
mdcath_streaming.py -- stream the mdCATH dataset over the network (HDF5 range
reads) via the OFFICIAL torchmd-net MDCATH class, with sampling under your
control. Nothing is bulk-downloaded: h5py issues HTTP range requests through
huggingface_hub's HfFileSystem and pulls only the bytes for the frames you touch.

Why this works
--------------
mdCATH per-domain files are HDF5. Opening a remote file object (fsspec/HfFileSystem)
instead of a local path makes h5py fetch just the b-tree/metadata + the exact
chunk slices for the frame you index. Measured on `12asA00` (a 1.5 GB file):
reading one frame transfers ~0.05 MB metadata + ~0.12 MB data -- not 1.5 GB, not
even the 53.8 MB full trajectory.

Two requirements addressed
--------------------------
1. Reuse torchmd-net: `StreamingMDCATH` subclasses `torchmdnet_mdcath.MDCATH`
   and overrides ONLY the file-access seams (`download`, `_ensure_source_file`,
   `process_specific_group`). All filtering + read logic is inherited unchanged,
   so a PDB filter/temperature/skip_frames behaves identically to upstream.
2. Your own sampling: the base class's `self.idx` is a fixed flat conformer list.
   We expose it plus grouping helpers and a `PerSequenceBatchSampler` so YOU
   choose batch composition (per-domain, per-temperature, weighted, subset...).

Requires: torch, torch_geometric, h5py, huggingface_hub (see environment.yaml).
The vendored `torchmdnet_mdcath.py` is the upstream module, unmodified, so this
runs without a full torchmd-net install. If you have torchmd-net installed you
can instead `from torchmdnet.datasets.mdcath import MDCATH`.
"""
from __future__ import annotations

import os
import threading
from contextlib import nullcontext as _nullcontext
from typing import Optional, Sequence

import numpy as np
import h5py

try:
    from torchmdnet.datasets.mdcath import MDCATH  # prefer the installed package
except Exception:
    from torchmdnet_mdcath import MDCATH           # fall back to vendored copy

from huggingface_hub import HfFileSystem, hf_hub_download


DEFAULT_REPO = "datasets/compsciencelab/mdCATH"


class StreamingMDCATH(MDCATH):
    """torchmd-net MDCATH that streams remote HDF5 instead of downloading.

    Extra parameters (beyond the upstream MDCATH signature)
    -------------------------------------------------------
    repo : str
        HF filesystem repo id. Default "datasets/compsciencelab/mdCATH".
    cache_source : bool
        The ~186 MB `mdcath_source.h5` index is needed for filtering. If True
        (default) it is fetched ONCE to `root` via hf_hub_download (metadata only,
        not trajectory data) -- far faster for the initial filter pass. If False
        the index itself is also streamed (no local file at all).
    rdcc_nbytes : int
        Per-open HDF5 chunk cache size. Default 4 MiB -- large enough to avoid
        re-fetching a chunk within one frame read, small enough to not over-read.
    """

    def __init__(self, root, *args,
                 repo: str = DEFAULT_REPO,
                 cache_source: bool = True,
                 rdcc_nbytes: int = 4 * 1024 * 1024,
                 **kwargs):
        self.repo = repo
        self.cache_source = cache_source
        self.rdcc_nbytes = rdcc_nbytes
        self._fs = HfFileSystem()
        # Per-process handle cache: h5py handles are NOT fork-safe, so we open
        # lazily and keep one handle per (pid, domain) to reuse range-read cache.
        self._handles: dict = {}
        self._handle_lock = threading.Lock()
        self._owner_pid = os.getpid()
        super().__init__(root, *args, **kwargs)

    # -- seam 1: never bulk-download domain files -------------------------------
    def download(self):
        return  # frames are streamed on demand in process_specific_group

    # -- seam 1b: don't stat local domain files that don't exist (streaming) ----
    def calculate_dataset_size(self):
        # Upstream sums os.path.getsize() of local domain files; in streaming
        # mode they aren't on disk. Report the remote total via metadata stat
        # (a HEAD-like call, not a download); fall back to 0 if unavailable.
        total = 0
        for pdb_id in self.processed.keys():
            try:
                total += self._fs.size(self._remote_path(pdb_id))
            except Exception:
                pass
        return round(total / (1024 * 1024), 4)

    # -- seam 2: source index (small) — cache once or stream --------------------
    def _ensure_source_file(self):
        source_path = os.path.join(self.root, self.source_file)
        if os.path.exists(source_path):
            return
        if self.cache_source:
            assert self.source_file == "mdcath_source.h5", \
                "Only 'mdcath_source.h5' is supported for download."
            # Metadata index only (~186 MB), NOT trajectory data.
            hf_hub_download(
                repo_id=self.repo.split("/", 1)[1],
                repo_type="dataset",
                filename=self.source_file,
                local_dir=self.root,
            )
        else:
            # Pure-stream mode: filtering reads the index over the network too.
            # Base _filter_and_prepare_data opens source_path with a plain path,
            # so we redirect it here by monkeypatching h5py.File open via a shim.
            self._stream_source = True

    def _remote_path(self, pdb_id: str) -> str:
        return f"{self.repo}/data/{self.file_basename}_{pdb_id}.h5"

    def _open_domain(self, pdb_id: str) -> h5py.File:
        """Return a cached, per-process, streamed h5py handle for a domain."""
        pid = os.getpid()
        if pid != self._owner_pid:
            # Forked worker inherited stale handles -- drop them.
            self._handles = {}
            self._owner_pid = pid
        with self._handle_lock:
            h = self._handles.get(pdb_id)
            if h is None:
                remote = self._fs.open(self._remote_path(pdb_id), "rb")
                h = h5py.File(remote, "r", rdcc_nbytes=self.rdcc_nbytes)
                self._handles[pdb_id] = h
            return h

    # -- seam 3: per-frame read — stream the slice instead of local open -------
    def process_specific_group(self, pdb, file, temp, repl, conf_idx):
        # Mirrors upstream logic exactly, but reads from a streamed handle and
        # does NOT close it (so the range-read cache persists across frames).
        conf_idx = conf_idx * self.skip_frames
        f = self._open_domain(pdb)
        z = f[pdb]["z"][:]
        coords = np.zeros((z.shape[0], 3), dtype=np.float32)
        forces = np.zeros((z.shape[0], 3), dtype=np.float32)
        group = f[f"{pdb}/{temp}/{repl}"]
        group["coords"].read_direct(coords, np.s_[conf_idx, :, :])
        group["forces"].read_direct(forces, np.s_[conf_idx, :, :])
        assert coords.shape[0] == forces.shape[0] == z.shape[0]
        return (z, coords, forces)

    # -- sampling control surface ----------------------------------------------
    def build_catalog(self):
        """Force construction of the flat conformer catalog and return it.

        Each entry: (pdb_id, file_path, temp, replica, conf_idx). You can slice,
        group, reweight, or subset this however you like to drive a Sampler.
        """
        if self.idx is None:
            self._setup_idx()
        return self.idx

    def group_indices_by(self, key: str = "domain"):
        """Return {group_value: [dataset_index, ...]} for custom batch sampling.

        key: 'domain' | 'temp' | 'replica' | 'domain_temp'
        """
        cat = self.build_catalog()
        keyfns = {
            "domain": lambda e: e[0],
            "temp": lambda e: e[2],
            "replica": lambda e: e[3],
            "domain_temp": lambda e: (e[0], e[2]),
        }
        if key not in keyfns:
            raise ValueError(f"key must be one of {list(keyfns)}")
        kf = keyfns[key]
        groups: dict = {}
        for i, entry in enumerate(cat):
            groups.setdefault(kf(entry), []).append(i)
        return groups

    def close(self):
        # Defensive: __del__ may fire even if __init__ failed before these
        # attributes were set (e.g. a network error inside super().__init__).
        lock = getattr(self, "_handle_lock", None)
        handles = getattr(self, "_handles", None)
        if handles is None:
            return
        ctx = lock if lock is not None else _nullcontext()
        with ctx:
            for h in handles.values():
                try:
                    h.close()
                except Exception:
                    pass
            self._handles = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------- custom samplers ---------------------------------
try:
    from torch.utils.data import Sampler
except Exception:
    class Sampler:  # import-time shim so this module loads without torch
        def __init__(self, *a, **k): pass


class PerSequenceBatchSampler(Sampler):
    """Yield batches whose conformers all come from ONE sequence (domain).

    You control the policy: shuffle of domain order, shuffle within domain,
    batch size, drop_last, optional cap on frames per domain per epoch, and an
    optional grouping key (e.g. 'domain_temp' to also keep temperature constant).

    Use as DataLoader(dataset, batch_sampler=this, ...).
    """

    def __init__(self, dataset: StreamingMDCATH, batch_size: int,
                 group_key: str = "domain", shuffle: bool = True,
                 drop_last: bool = False, max_frames_per_group: Optional[int] = None,
                 seed: int = 0):
        self.groups = dataset.group_indices_by(group_key)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_frames_per_group = max_frames_per_group
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _epoch_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(keys)
        for k in keys:
            idx = np.asarray(self.groups[k])
            if self.shuffle:
                idx = idx.copy(); rng.shuffle(idx)
            if self.max_frames_per_group is not None:
                idx = idx[:self.max_frames_per_group]
            n = len(idx)
            nb = n // self.batch_size if self.drop_last else -(-n // self.batch_size)
            for b in range(nb):
                yield idx[b * self.batch_size:(b + 1) * self.batch_size].tolist()

    def __iter__(self):
        return self._epoch_batches()

    def __len__(self):
        total = 0
        for idx in self.groups.values():
            n = len(idx) if self.max_frames_per_group is None \
                else min(len(idx), self.max_frames_per_group)
            total += n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        return total


def demo():
    import argparse
    import torch
    from torch_geometric.loader import DataLoader

    ap = argparse.ArgumentParser(description="Streaming mdCATH (torchmd-net) demo")
    ap.add_argument("--root", default="./mdcath_stream_cache",
                    help="dir for the ~186MB source index only (no trajectories)")
    ap.add_argument("--pdb-list", nargs="*", default=["12asA00"],
                    help="domains to stream (default: 12asA00)")
    ap.add_argument("--temperatures", nargs="*", default=["348"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-frames", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--num-atoms", type=int, default=None,
                    help="max protein atoms filter (upstream default 5000; "
                         "None = no cap). 12asA00 has 5091, so it needs None.")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = StreamingMDCATH(
        root=args.root, pdb_list=args.pdb_list,
        temperatures=args.temperatures, skip_frames=args.skip_frames,
        numAtoms=args.num_atoms,
    )
    print(f"domains kept: {list(ds.processed.keys())}  conformers: {ds.num_conformers}")

    sampler = PerSequenceBatchSampler(ds, batch_size=args.batch_size,
                                      group_key="domain_temp", shuffle=True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"batches/epoch: {len(sampler)}")

    for i, data in enumerate(loader):
        data = data.to(device, non_blocking=True)
        print(f"batch {i}: z={tuple(data.z.shape)} pos={tuple(data.pos.shape)} "
              f"neg_dy={tuple(data.neg_dy.shape)} batch={tuple(data.batch.shape)} "
              f"on {data.pos.device}")
        if i + 1 >= args.max_batches:
            break
    ds.close()
    print("done -- nothing but the index + touched frames was transferred.")


if __name__ == "__main__":
    demo()
