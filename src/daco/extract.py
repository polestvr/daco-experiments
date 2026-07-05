"""Frozen BEATs embedding extraction with on-disk caching.

Clips are grouped by exact sample length so batches need no padding mask;
DCASE machines use a single fixed clip length (10 s or 12 s at 16 kHz).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backbones.beats.BEATs import BEATs, BEATsConfig

from .data import Clip


def load_beats(ckpt_path: Path, device: str = "cuda") -> BEATs:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _embed_batch(model, wavs: np.ndarray, device: str) -> np.ndarray:
    if hasattr(model, "embed_batch"):        # generic embedder (embedders.py)
        return model.embed_batch(wavs)
    source = torch.from_numpy(wavs).float().to(device)
    out = model.extract_features(source)     # bare BEATs module (run_e1/e2/e3)
    feats = out[0] if isinstance(out, tuple) else out   # (B, T, D)
    return feats.mean(dim=1).cpu().numpy()              # clip-level: mean over time


def _mono(wav: np.ndarray, channel: int | None) -> np.ndarray:
    """Reduce multi-channel audio: fixed channel index, or mean mixdown."""
    if wav.ndim == 1:
        return wav
    return wav[:, channel] if channel is not None else wav.mean(axis=1)


@torch.no_grad()
def embed_clips(model: BEATs, clips: list[Clip], device: str = "cuda",
                batch_size: int = 8, channel: int | None = None) -> np.ndarray:
    lengths = []
    for c in clips:
        info = sf.info(str(c.path))
        if info.samplerate != 16000:
            raise ValueError(f"expected 16 kHz audio, got {info.samplerate}: {c.path}")
        lengths.append(info.frames)
    lengths = np.asarray(lengths)

    dim = None
    embeddings: dict[int, np.ndarray] = {}
    for length in np.unique(lengths):
        idxs = np.flatnonzero(lengths == length)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            wavs = np.stack([
                _mono(sf.read(str(clips[i].path), dtype="float32")[0], channel)
                for i in batch_idx])
            emb = _embed_batch(model, wavs, device)
            dim = emb.shape[1]
            for i, e in zip(batch_idx, emb):
                embeddings[i] = e

    X = np.zeros((len(clips), dim), dtype=np.float32)
    for i, e in embeddings.items():
        X[i] = e
    return X


def cache_path_for(clips: list[Clip], cache_dir: Path, tag: str,
                   model_tag: str = "") -> Path:
    """Deterministic cache location for (model_tag, tag, exact file list)."""
    key = hashlib.sha1((model_tag + "\n" + "\n".join(
        str(c.path) for c in clips)).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{tag}_{key}.npz"


def cached_embeddings(model: BEATs, clips: list[Clip], cache_dir: Path,
                      tag: str, device: str = "cuda",
                      batch_size: int = 8, model_tag: str = "",
                      channel: int | None = None) -> np.ndarray:
    """Embed `clips`, caching by (model_tag, tag, file list) so reruns are
    free and a different checkpoint can never silently reuse stale vectors.
    For multi-channel audio, `channel` selects one channel (None = mixdown);
    encode the choice in `tag` so cache entries stay distinct."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for(clips, cache_dir, tag, model_tag)

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        if list(data["files"]) == [str(c.path) for c in clips]:
            return data["X"]

    X = embed_clips(model, clips, device=device, batch_size=batch_size,
                    channel=channel)
    np.savez_compressed(
        cache_file, X=X,
        files=np.array([str(c.path) for c in clips]),
        domain=np.array([c.domain for c in clips]),
        split=np.array([c.split for c in clips]),
        label=np.array([-1 if c.label is None else c.label for c in clips]),
    )
    return X
