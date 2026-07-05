"""Frozen audio backbone embedders behind one interface.

Each embedder exposes embed_batch(wavs: np.ndarray (B, samples) @16 kHz)
-> np.ndarray (B, D) clip-level embeddings (time-mean pooled), plus a
`tag` used to key the on-disk embedding cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BEATsEmbedder:
    tag_prefix = "beats"

    def __init__(self, ckpt: Path, device: str = "cuda"):
        from backbones.beats.BEATs import BEATs, BEATsConfig
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt_data["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt_data["model"])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        # bare checkpoint stem, matching the cache entries written by run_e1/e2/e3
        self.tag = Path(ckpt).stem

    @torch.no_grad()
    def embed_batch(self, wavs: np.ndarray) -> np.ndarray:
        source = torch.from_numpy(wavs).float().to(self.device)
        out = self.model.extract_features(source)
        feats = out[0] if isinstance(out, tuple) else out
        return feats.mean(dim=1).cpu().numpy()


class EATEmbedder:
    """EAT via the authors' Hugging Face export (no fairseq needed).

    Input per the model card: Kaldi fbank (128 mel bins, 10 ms hop, htk
    compat, hanning, dither 0) of the mean-subtracted 16 kHz waveform,
    padded/truncated to 1024 frames, normalized (mel + 4.268) / (4.569 * 2),
    shaped (B, 1, 1024, 128). Utterance embedding = the CLS token
    (extract_features(...)[:, 0]), 768-d — the authors' designated
    utterance-level representation.
    """
    tag_prefix = "eat"
    TARGET_LEN = 1024
    NORM_MEAN, NORM_STD = -4.268, 4.569

    def __init__(self, hf_name: str = "worstchan/EAT-base_epoch30_pretrain",
                 device: str = "cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.tag = f"{self.tag_prefix}_{hf_name.split('/')[-1]}"

    def _fbank(self, wav: torch.Tensor) -> torch.Tensor:
        import torchaudio.compliance.kaldi as ta_kaldi
        wav = wav - wav.mean()
        mel = ta_kaldi.fbank(wav.unsqueeze(0), htk_compat=True,
                             sample_frequency=16000, use_energy=False,
                             window_type="hanning", num_mel_bins=128,
                             dither=0.0, frame_shift=10)
        n = mel.shape[0]
        if n < self.TARGET_LEN:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, self.TARGET_LEN - n))
        else:
            mel = mel[:self.TARGET_LEN]
        return (mel - self.NORM_MEAN) / (self.NORM_STD * 2)

    @torch.no_grad()
    def embed_batch(self, wavs: np.ndarray) -> np.ndarray:
        mels = torch.stack([self._fbank(torch.from_numpy(w).float())
                            for w in wavs])
        mels = mels.unsqueeze(1).to(self.device)      # (B, 1, 1024, 128)
        feats = self.model.extract_features(mels)      # (B, 513, 768)
        return feats[:, 0].cpu().numpy()               # CLS utterance token


class PANNsEmbedder:
    """PANNs Cnn14_16k (2048-d penultimate embedding)."""
    tag_prefix = "panns"

    def __init__(self, ckpt: Path, device: str = "cuda"):
        from backbones.panns.models import Cnn14_16k
        self.model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160,
                               mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.tag = f"{self.tag_prefix}_{Path(ckpt).stem}"

    @torch.no_grad()
    def embed_batch(self, wavs: np.ndarray) -> np.ndarray:
        source = torch.from_numpy(wavs).float().to(self.device)
        out = self.model(source, None)
        return out["embedding"].cpu().numpy()


def load_embedder(name: str, ckpt: str | None, device: str = "cuda"):
    if name == "beats":
        return BEATsEmbedder(Path(ckpt).expanduser(), device)
    if name == "eat":
        return EATEmbedder(ckpt or "worstchan/EAT-base_epoch30_pretrain", device)
    if name == "panns":
        return PANNsEmbedder(Path(ckpt).expanduser(), device)
    raise ValueError(f"unknown backbone {name!r}")
