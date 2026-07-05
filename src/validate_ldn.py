"""Validate the LDN reimplementation against published numbers.

Wilkinghoff et al. (TASLP, TR2026-010) Table IV reports, for raw BEATs
embeddings with ratio-based LDN (K=1) under the OFFICIAL DCASE metric:
DCASE2023 dev 64.8 / eval 67.6 / hmean 66.2.

Their exact configuration (Sec. IV-C):
  - BEATs iter3 (AudioSet pre-trained, no fine-tuning)
  - embedding = temporal mean of the patch-token grid, flattened:
    tokens (T', 8, 768) -> mean over T' -> 8x768 = 6144-d
  - distance = MSE (not cosine)
  - reference set = all 1000 training samples (no domain labels)
  - ratio LDN, K=1: score(x) = min_y MSE(x,y) / dens(y),
    dens(y) = min_{y' != y} MSE(y, y')

This script reproduces exactly that configuration on DCASE 2023 and prints
dev/eval official scores next to the published ones, plus the un-normalized
min-MSE baseline for the delta.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.data import list_clips
from daco.embedders import BEATsEmbedder
from daco.extract import cached_embeddings
from daco.metrics import machine_metrics, official_score

DEV7 = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
EVAL_2023 = ["Vacuum", "ToyTank", "ToyNscale", "ToyDrone",
             "bandsaw", "grinder", "shaker"]


class BEATs6144Embedder(BEATsEmbedder):
    """Wilkinghoff-style embedding: temporal mean over the time-patch axis,
    frequency-patch axis kept and flattened -> 8 x 768 = 6144-d."""

    @torch.no_grad()
    def embed_batch(self, wavs: np.ndarray) -> np.ndarray:
        source = torch.from_numpy(wavs).float().to(self.device)
        out = self.model.extract_features(source)
        feats = out[0] if isinstance(out, tuple) else out    # (B, T'*8, 768)
        B, N, D = feats.shape
        assert N % 8 == 0, f"token count {N} not divisible by 8"
        grid = feats.reshape(B, N // 8, 8, D)                # (B, T', 8, 768)
        return grid.mean(dim=1).reshape(B, 8 * D).cpu().numpy()


def sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances (MSE up to the constant 1/D,
    which cancels in the LDN ratio and is AUC-irrelevant for the baseline)."""
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2023")
    ap.add_argument("--ckpt", default=str(
        Path.home() / "models/beats/BEATs_iter3_plus_AS2M.pt"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "data/embeddings_multi_years")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev_root = Path.home() / f"data/dcase{args.year}t2/dev/raw"
    eval_root = Path.home() / f"data/dcase{args.year}t2/eval/raw"
    gt_root = Path.home() / f"tools/dcase{args.year}_task2_evaluator"
    eval_machines = EVAL_2023 if args.year == "2023" else None

    emb = BEATs6144Embedder(Path(args.ckpt).expanduser(), args.device)
    emb.tag = "beats6144_" + Path(args.ckpt).stem

    results = {}
    for split, mlist, root, gt in (("dev", DEV7, dev_root, None),
                                   ("eval", eval_machines, eval_root, gt_root)):
        per_machine = {}
        for machine in mlist:
            t0 = time.time()
            clips = list_clips(root, machine, gt_root=gt)
            train = [c for c in clips if c.split == "train"]
            test = [c for c in clips if c.split == "test"]
            X_tr = cached_embeddings(emb, train, args.cache_dir,
                                     f"{args.year}_{machine}_train",
                                     args.device, model_tag=emb.tag)
            X_te = cached_embeddings(emb, test, args.cache_dir,
                                     f"{args.year}_{machine}_test",
                                     args.device, model_tag=emb.tag)
            labels = np.array([c.label for c in test])
            domains = np.array([c.domain for c in test])

            D_rr = sq_dists(X_tr, X_tr)
            np.fill_diagonal(D_rr, np.inf)
            dens = np.maximum(D_rr.min(axis=1), 1e-30)       # K=1 LOO density
            D_tr = sq_dists(X_te, X_tr)
            baseline = D_tr.min(axis=1)                       # min-MSE
            ldn = (D_tr / dens[None, :]).min(axis=1)          # ratio, K=1

            per_machine[machine] = {
                "baseline": machine_metrics(baseline, labels, domains),
                "ldn": machine_metrics(ldn, labels, domains)}
            print(f"  {machine} ({split}): {time.time()-t0:.0f}s", flush=True)

        for system in ("baseline", "ldn"):
            results[(split, system)] = official_score(
                {m: v[system] for m, v in per_machine.items()}) * 100

    print(f"\n=== LDN validation, DCASE {args.year}, BEATs-6144 + MSE ===")
    print(f"{'system':22s} {'dev':>7s} {'eval':>7s}")
    for system in ("baseline", "ldn"):
        print(f"{system + ' (min-MSE)' if system == 'baseline' else 'ratio LDN K=1':22s}"
              f" {results[('dev', system)]:7.2f} {results[('eval', system)]:7.2f}")
    print("published (TASLP Tab. IV): dev 64.8, eval 67.6 (ratio LDN K=1)")


if __name__ == "__main__":
    main()
