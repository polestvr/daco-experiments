"""E2 — DACo main result: per-latent-domain conformal calibration of frozen
BEATs + kNN anomaly scores on the DCASE 2025 Task 2 development set.

Variants: raw kNN; DACo conformal with prior-strength sweep m in
{0, 10, 30, 100, 300} (m traces the source/target balance frontier:
m=0 is pure per-domain FPR equalization, m->inf recovers the raw ranking);
per-domain median-ratio normalization; and an oracle-assignment diagnostic.

Everything runs on cached embeddings; a full dev cycle is seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.backends import KNNBackend
from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.data import DEV_MACHINES, list_clips
from daco.extract import cached_embeddings, load_beats
from daco.metrics import machine_metrics, official_score

VARIANTS: list[tuple[str, str | None, str | None, float]] = [
    ("raw", None, None, 0),
    ("daco-m0", "soft", "conformal", 0),
    ("daco-m10", "soft", "conformal", 10),
    ("daco-m30", "soft", "conformal", 30),
    ("daco-m100", "soft", "conformal", 100),
    ("daco-m300", "soft", "conformal", 300),
    ("ratio-soft", "soft", "ratio", 0),
    ("daco-m30-orc", "oracle", "conformal", 30),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--machines", nargs="*", default=DEV_MACHINES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--knn-k", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results" / "e2_dev.csv")
    args = ap.parse_args()

    model = load_beats(args.ckpt.expanduser(), args.device)

    rows = []
    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    assign_rows = []

    for machine in args.machines:
        clips = list_clips(args.data_root.expanduser(), machine)
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]

        X_train = cached_embeddings(model, train, args.cache_dir.expanduser(),
                                    f"{machine}_train", args.device,
                                    model_tag=args.ckpt.expanduser().stem)
        X_test = cached_embeddings(model, test, args.cache_dir.expanduser(),
                                   f"{machine}_test", args.device,
                                   model_tag=args.ckpt.expanduser().stem)

        train_domains = np.array([c.domain for c in train])
        test_domains = np.array([c.domain for c in test])
        labels = np.array([c.label for c in test])

        base_scores = KNNBackend(k=args.knn_k).fit(X_train).score(X_test)
        loo = loo_knn_scores(X_train, k=args.knn_k)

        for name, assignment, method, m in VARIANTS:
            if name == "raw":
                scores = base_scores
            else:
                cal = DACoCalibrator(assignment, method, prior_strength=m).fit(
                    X_train, train_domains, loo)
                w = cal.domain_weights(
                    X_test,
                    oracle_domains=test_domains if assignment == "oracle" else None)
                scores = cal.transform(base_scores, w)
            res = machine_metrics(scores, labels, test_domains)
            per_variant.setdefault(name, {})[machine] = res
            rows.append({"machine": machine, "variant": name,
                         **{k: round(v * 100, 2) for k, v in res.items()}})

        # Stage-1 diagnostic: hard-assignment accuracy on test NORMALS
        # (anomalies can legitimately sit far from both banks)
        w_hard = DACoCalibrator("hard").fit(X_train, train_domains, loo) \
            .domain_weights(X_test)
        pred_target = w_hard[:, 1] > 0.5
        normal = labels == 0
        acc = float((pred_target[normal] == (test_domains[normal] == "target")).mean())
        assign_rows.append({"machine": machine, "normal_assign_acc": round(acc * 100, 1)})
        print(f"{machine}: done (normal-clip domain assignment acc {acc*100:.1f}%)",
              flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    for metric in ("auc_source", "auc_target", "pauc"):
        print(f"\n=== {metric} (%) ===")
        print(df.pivot(index="machine", columns="variant", values=metric)
              [[v for v, *_ in VARIANTS]].to_string())

    print("\n=== Balance frontier (means over machines) + official score ===")
    print(f"  {'variant':13s} {'mean_AUC_s':>10s} {'mean_AUC_t':>10s} "
          f"{'mean_pAUC':>9s} {'omega':>6s}")
    for name, *_ in VARIANTS:
        pm = per_variant[name]
        ms = np.mean([v["auc_source"] for v in pm.values()]) * 100
        mt = np.mean([v["auc_target"] for v in pm.values()]) * 100
        mp = np.mean([v["pauc"] for v in pm.values()]) * 100
        print(f"  {name:13s} {ms:10.2f} {mt:10.2f} {mp:9.2f} "
              f"{official_score(pm) * 100:6.2f}")

    print("\n=== Stage-1 latent-domain assignment (test normals, %) ===")
    print(pd.DataFrame(assign_rows).to_string(index=False))
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
