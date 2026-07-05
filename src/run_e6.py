"""E6 — efficiency measurements for the paper.

Per backbone: parameter count, embedding-extraction throughput (measured on
real dev clips), peak VRAM during extraction. Plus the marginal cost of one
full DACo configuration cycle (base scores + LOO + calibration + criterion)
on cached embeddings, which is the quantity that makes the ablation grid
cheap.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import Config, _knn_scores, cv_balance_criterion
from daco.embedders import load_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", type=Path, required=True)
    ap.add_argument("--beats-ckpt", type=Path, required=True)
    ap.add_argument("--panns-ckpt", type=Path, required=True)
    ap.add_argument("--n-clips", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.dev_root.expanduser()
                                 / "fan" / "train" / "*.wav")))[:args.n_clips]
    wavs = np.stack([sf.read(f, dtype="float32")[0] for f in files])

    print(f"=== E6 efficiency (RTX 4070 Ti SUPER, {args.n_clips} clips of "
          f"{wavs.shape[1]/16000:.0f}s) ===")
    print(f"{'backbone':8s} {'params(M)':>9s} {'dim':>5s} {'ms/clip':>8s} "
          f"{'clips/s':>8s} {'peakVRAM(MB)':>12s}")

    for name, ckpt in (("beats", str(args.beats_ckpt)),
                       ("eat", None), ("panns", str(args.panns_ckpt))):
        emb = load_embedder(name, ckpt, args.device)
        params = sum(p.numel() for p in emb.model.parameters()) / 1e6
        torch.cuda.reset_peak_memory_stats()
        emb.embed_batch(wavs[:8])                       # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        X = np.concatenate([emb.embed_batch(wavs[i:i + 8])
                            for i in range(0, len(wavs), 8)])
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 2**20
        print(f"{name:8s} {params:9.1f} {X.shape[1]:5d} "
              f"{dt / len(wavs) * 1000:8.1f} {len(wavs) / dt:8.1f} {vram:12.0f}")
        del emb
        torch.cuda.empty_cache()

    # marginal cost of one configuration cycle on cached embeddings (CPU)
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(1000, 768)).astype(np.float32)
    X_test = rng.normal(size=(200, 768)).astype(np.float32)
    dom = np.array(["source"] * 990 + ["target"] * 10)
    cfg = Config("conf-soft-m0-k1", "soft", "conformal", 0, 1)
    t0 = time.time()
    loo = loo_knn_scores(X_train, 1)
    cal = DACoCalibrator("soft", "conformal", 0).fit(X_train, dom, loo)
    _ = cal.transform(_knn_scores(X_test, X_train, 1),
                      cal.domain_weights(X_test))
    t_score = time.time() - t0
    t0 = time.time()
    _ = cv_balance_criterion(X_train, dom, cfg, n_splits=10)
    t_crit = time.time() - t0
    print(f"\nper-machine config cycle on cached embeddings: "
          f"score+calibrate {t_score*1000:.0f} ms, criterion (10 splits) "
          f"{t_crit*1000:.0f} ms")


if __name__ == "__main__":
    main()
