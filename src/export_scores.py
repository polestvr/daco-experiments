"""Export per-clip anomaly scores in the official evaluator's format.

Writes teams/<team>/<system>/anomaly_score_<machine>_section_00_test.csv and
decision_result_..._test.csv for the requested configuration, so results can
be cross-checked with github.com/nttcslab/dcase2025_task2_evaluator.

The decision threshold is the label-free 90th percentile of the training LOO
scores mapped through the same calibration (only F1-type metrics depend on it;
AUC/pAUC do not).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator, loo_knn_scores
from daco.criteria import Config, _knn_scores
from daco.data import EVAL_MACHINES, list_clips
from daco.extract import cached_embeddings, load_beats


def parse_config(name: str) -> Config:
    if name.startswith("raw-k"):
        return Config(name, None, None, 0, int(name.split("-k")[1]))
    kind, assign, *rest = name.split("-")
    k = int(rest[-1][1:])
    if kind == "conf":
        return Config(name, assign, "conformal", float(rest[0][1:]), k)
    return Config(name, assign, {"ratio": "ratio", "zscore": "zscore"}[kind], 0, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. conf-soft-m0-k2, raw-k1")
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--teams-dir", type=Path, required=True)
    ap.add_argument("--team", default="daco")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    model = load_beats(args.ckpt.expanduser(), args.device)
    model_tag = args.ckpt.expanduser().stem
    out = args.teams_dir.expanduser() / args.team / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    for machine in EVAL_MACHINES:
        clips = list_clips(args.eval_root.expanduser(), machine,
                           gt_root=args.gt_root.expanduser())
        train = [c for c in clips if c.split == "train"]
        test = [c for c in clips if c.split == "test"]
        X_train = cached_embeddings(model, train, args.cache_dir.expanduser(),
                                    f"{machine}_train", args.device,
                                    model_tag=model_tag)
        X_test = cached_embeddings(model, test, args.cache_dir.expanduser(),
                                   f"{machine}_test", args.device,
                                   model_tag=model_tag)
        train_domains = np.array([c.domain for c in train])
        base = _knn_scores(X_test, X_train, cfg.k)
        loo = loo_knn_scores(X_train, k=cfg.k)
        if cfg.assignment is None:
            scores, thr = base, float(np.percentile(loo, 90))
        else:
            cal = DACoCalibrator(cfg.assignment, cfg.method,
                                 prior_strength=cfg.prior_strength).fit(
                X_train, train_domains, loo)
            scores = cal.transform(base, cal.domain_weights(X_test))
            cal_train = cal.transform(loo, cal.domain_weights(X_train))
            thr = float(np.percentile(cal_train, 90))

        with open(out / f"anomaly_score_{machine}_section_00_test.csv", "w") as f:
            for c, s in zip(test, scores):
                f.write(f"{c.path.name},{s:.10f}\n")
        with open(out / f"decision_result_{machine}_section_00_test.csv", "w") as f:
            for c, s in zip(test, scores):
                f.write(f"{c.path.name},{int(s > thr)}\n")
        print(f"{machine}: exported {len(test)} scores", flush=True)

    print(f"scores for {cfg.name} written to {out}")


if __name__ == "__main__":
    main()
