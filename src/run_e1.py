"""E1 — frozen-embedding baseline on the DCASE 2025 Task 2 development set.

Extracts (cached) BEATs embeddings, scores the dev test clips with
training-free backends, and reports per-machine AUC_s / AUC_t / pAUC plus the
official harmonic-mean score, side by side per backend.

Usage (WSL):
    python src/run_e1.py --data-root ~/data/dcase2025t2/dev/raw \
        --ckpt ~/models/beats/BEATs_iter3_plus_AS2M.pt \
        --cache-dir ~/data/dcase2025t2/embeddings/beats_iter3p_as2m
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.backends import GMMBackend, KNNBackend, SelectiveMahalanobis
from daco.data import DEV_MACHINES, list_clips
from daco.extract import cached_embeddings, load_beats
from daco.metrics import machine_metrics, official_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--machines", nargs="*", default=DEV_MACHINES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "e1_dev.csv")
    args = ap.parse_args()

    model = load_beats(args.ckpt.expanduser(), args.device)
    print(f"loaded BEATs from {args.ckpt}", flush=True)

    rows = []
    per_backend: dict[str, dict[str, dict[str, float]]] = {}

    for machine in args.machines:
        t0 = time.time()
        clips = list_clips(args.data_root.expanduser(), machine)
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]
        if not train or not test:
            raise RuntimeError(f"{machine}: found {len(train)} train / {len(test)} test clips")

        X_train = cached_embeddings(model, train, args.cache_dir.expanduser(),
                                    f"{machine}_train", args.device, args.batch_size,
                                    model_tag=args.ckpt.expanduser().stem)
        X_test = cached_embeddings(model, test, args.cache_dir.expanduser(),
                                   f"{machine}_test", args.device, args.batch_size,
                                   model_tag=args.ckpt.expanduser().stem)

        X_src = X_train[[i for i, c in enumerate(train) if c.domain == "source"]]
        X_tgt = X_train[[i for i, c in enumerate(train) if c.domain == "target"]]
        labels = np.array([c.label for c in test])
        domains = np.array([c.domain for c in test])

        backends = {
            "knn1": KNNBackend(k=1).fit(X_train),
            "sel-maha": SelectiveMahalanobis().fit(X_src, X_tgt),
            "gmm2": GMMBackend(n_components=2).fit(X_train),
        }
        for name, backend in backends.items():
            m = machine_metrics(backend.score(X_test), labels, domains)
            per_backend.setdefault(name, {})[machine] = m
            rows.append({"machine": machine, "backend": name,
                         **{k: round(v * 100, 2) for k, v in m.items()}})

        print(f"{machine}: {len(train)} train / {len(test)} test, "
              f"{time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n=== E1: frozen BEATs + training-free backends (dev set, %) ===")
    print(df.pivot(index="machine", columns="backend",
                   values=["auc_source", "auc_target", "pauc"]).to_string())
    print("\nOfficial harmonic-mean score:")
    for name, machines in per_backend.items():
        print(f"  {name}: {official_score(machines) * 100:.2f}")
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
