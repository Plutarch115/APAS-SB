"""
Sequence / SMILES featurizers for the PEARL binding-affinity pipeline.

Protein sequences are embedded with ESM2 (loaded via the ``esm`` / fair-esm
package) and ligand SMILES with MolFormer (``ibm/MoLFormer-XL-both-10pct`` via
HuggingFace ``transformers``). Both produce *per-token* embeddings (one vector
per residue / per SMILES token), which the (Mock)PEARL trunk consumes as a
variable-length sequence of feature vectors.

Embeddings are expensive, so every result is cached to disk as a float16 ``.npy``
keyed by a hash of (model signature + text). This lets a one-time precompute
pass populate the cache on GPU, after which DataLoader workers only read files
(safe with num_workers > 0 and no CUDA in workers).

Notes on the environment:
- MolFormer is loaded with ``trust_remote_code=True`` pinned to a stable
  revision that is compatible with transformers 4.53.
- ``transformers`` is prevented from importing a (possibly broken) torchvision
  by marking it unavailable before any vision utils load. In the project venv
  torchvision is simply absent, so this is a harmless belt-and-suspenders guard.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Tokenizers are used before DataLoader forks workers; disable their internal
# parallelism to avoid noisy fork warnings / potential deadlocks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# ESM2 (protein) featurizer
# ---------------------------------------------------------------------------

# Embedding dimension per fair-esm model name (used before the model is loaded).
ESM2_DIMS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


class ESM2Featurizer:
    """Per-residue ESM2 embeddings via fair-esm, with a disk cache."""

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        max_length: int = 1022,          # ESM2 positional limit (excl. BOS/EOS)
        cache_dir: Optional[str] = None,
        dtype=np.float16,
    ):
        if model_name not in ESM2_DIMS:
            raise ValueError(
                f"Unknown ESM2 model '{model_name}'. Known: {sorted(ESM2_DIMS)}"
            )
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._batch_converter = None
        self._repr_layer = None

    @property
    def dim(self) -> int:
        return ESM2_DIMS[self.model_name]

    def _ensure_model(self):
        if self._model is not None:
            return
        import esm  # fair-esm

        model, alphabet = getattr(esm.pretrained, self.model_name)()
        model = model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        self._repr_layer = model.num_layers

    def cache_path(self, sequence: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_hash(sequence)}.npy"

    def get_cached(self, sequence: str) -> Optional[np.ndarray]:
        p = self.cache_path(sequence)
        if p is not None and p.exists():
            return np.load(p)
        return None

    @torch.no_grad()
    def embed_batch(self, sequences: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """Return one [L, dim] float array per input sequence (cache-aware)."""
        results: List[Optional[np.ndarray]] = [None] * len(sequences)
        todo = []
        for i, seq in enumerate(sequences):
            cached = self.get_cached(seq)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)

        if todo:
            self._ensure_model()
            for start in range(0, len(todo), batch_size):
                idxs = todo[start:start + batch_size]
                # Truncate overly long sequences to the model's positional limit.
                data = [(str(j), sequences[j][: self.max_length]) for j in idxs]
                _, _, toks = self._batch_converter(data)
                toks = toks.to(self.device)
                out = self._model(toks, repr_layers=[self._repr_layer])
                reps = out["representations"][self._repr_layer]  # [B, T, D]
                for k, j in enumerate(idxs):
                    L = len(sequences[j][: self.max_length])
                    # Strip BOS (index 0) and take exactly L residue tokens.
                    emb = reps[k, 1:1 + L].float().cpu().numpy().astype(self.dtype)
                    results[j] = emb
                    p = self.cache_path(sequences[j])
                    if p is not None:
                        np.save(p, emb)

        return [r for r in results]  # type: ignore[return-value]

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def release(self):
        """Drop the loaded model and free GPU memory (cache stays on disk)."""
        self._model = None
        self._batch_converter = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MolFormer (ligand) featurizer
# ---------------------------------------------------------------------------

# Stable MolFormer revision compatible with transformers 4.53 (newer HEAD needs
# an API absent in that version).
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"

# Known MolFormer hidden sizes (lets callers get `.dim` without loading weights).
MOLFORMER_DIMS = {
    "ibm/MoLFormer-XL-both-10pct": 768,
    "ibm-research/MoLFormer-XL-both-10pct": 768,
}


def _neutralize_torchvision():
    """Stop transformers from importing a broken/absent torchvision."""
    try:
        import transformers.utils.import_utils as iu
        iu._torchvision_available = False
        iu.is_torchvision_available = lambda: False
    except Exception:
        pass


class MolFormerFeaturizer:
    """Per-token MolFormer embeddings via transformers, with a disk cache."""

    def __init__(
        self,
        model_name: str = "ibm/MoLFormer-XL-both-10pct",
        revision: str = MOLFORMER_REVISION,
        device: Optional[str] = None,
        max_length: int = 202,
        cache_dir: Optional[str] = None,
        dtype=np.float16,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None
        self._dim = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Fast path: known models expose their hidden size without loading.
            if self.model_name in MOLFORMER_DIMS:
                return MOLFORMER_DIMS[self.model_name]
            self._ensure_model()
        return self._dim

    def _ensure_model(self):
        if self._model is not None:
            return
        _neutralize_torchvision()
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision
        )
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision,
            deterministic_eval=True,
        ).eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._dim = int(model.config.hidden_size)

    def cache_path(self, smiles: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_hash(smiles)}.npy"

    def get_cached(self, smiles: str) -> Optional[np.ndarray]:
        p = self.cache_path(smiles)
        if p is not None and p.exists():
            return np.load(p)
        return None

    @torch.no_grad()
    def embed_batch(self, smiles_list: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Return one [L, dim] float array per SMILES (cache-aware)."""
        results: List[Optional[np.ndarray]] = [None] * len(smiles_list)
        todo = []
        for i, smi in enumerate(smiles_list):
            cached = self.get_cached(smi)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)

        if todo:
            self._ensure_model()
            tok = self._tokenizer
            for start in range(0, len(todo), batch_size):
                idxs = todo[start:start + batch_size]
                batch_smiles = [smiles_list[j] for j in idxs]
                enc = tok(
                    batch_smiles, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._model(**enc)
                hidden = out.last_hidden_state  # [B, T, D]
                attn = enc["attention_mask"]
                input_ids = enc["input_ids"]
                for k, j in enumerate(idxs):
                    # Keep real (non-pad) tokens, drop special tokens.
                    ids = input_ids[k].tolist()
                    special = tok.get_special_tokens_mask(
                        ids, already_has_special_tokens=True
                    )
                    keep = [
                        t for t in range(len(ids))
                        if attn[k, t].item() == 1 and special[t] == 0
                    ]
                    if not keep:  # degenerate SMILES -> keep all real tokens
                        keep = [t for t in range(len(ids)) if attn[k, t].item() == 1]
                    emb = hidden[k, keep].float().cpu().numpy().astype(self.dtype)
                    results[j] = emb
                    p = self.cache_path(smiles_list[j])
                    if p is not None:
                        np.save(p, emb)

        return [r for r in results]  # type: ignore[return-value]

    def embed(self, smiles: str) -> np.ndarray:
        return self.embed_batch([smiles])[0]

    def release(self):
        """Drop the loaded model and free GPU memory (cache stays on disk)."""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fixed-length packing helper (used by the dataset / collate)
# ---------------------------------------------------------------------------

def pad_or_truncate(emb: np.ndarray, max_len: int, dim: int):
    """
    Pack a variable-length [L, D] embedding into a fixed [max_len, D] block and
    a boolean mask [max_len] (True = real token). Truncates if longer.
    """
    features = np.zeros((max_len, dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=bool)
    if emb is None or emb.shape[0] == 0:
        return features, mask
    L = min(emb.shape[0], max_len)
    features[:L] = emb[:L].astype(np.float32)
    mask[:L] = True
    return features, mask
