"""Export per-clip anomaly scores for the E8 (DCASE 2026) systems in the
official evaluator's format.

Writes teams/daco-<policy>/<system>/anomaly_score_<machine>_section_00_test.csv
and decision_result_..._test.csv directly into the checkout of
github.com/nttcslab/dcase2026_task2_evaluator, so every E8 headline number can
be re-scored end-to-end by the official evaluator (same hygiene standard as the
2025 exports in export_scores.py).

Exports, per channel policy, exactly the systems quoted in section VI-F:
the frozen guarded pick, the frozen dev-Omega pick, and the fixed default.
Embeddings come from the E8 cache (same keys as run_e8_2026.py), so no GPU
forward passes are needed after E8 has run.

The decision threshold is the label-free 90th percentile of the training-bank
scores mapped through the same transform (only F1-type metrics depend on it;
AUC/pAUC and the official score do not).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import loo_knn_scores
from daco.data import list_clips
from daco.embedders import load_embedder
from daco.extract import cached_embeddings
from run_e45 import build_extended_configs, config_scores_ext
from run_e8_2026 import EVAL_MACHINES_2026

# Frozen systems per channel policy (PREREGISTRATION.md; sanity-checked by
# run_e8_2026.py): guarded pick, dev-Omega pick, fixed default.
SYSTEMS = {
    "ch0": ["ratio-hard-k2", "raw-k1", "conf-soft-m0-k1"],
    "mix": ["conf-soft-m0-k1", "conf-hard-m300-k1"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/models/beats/BEATs_iter3_plus_AS2M.pt")
    ap.add_argument("--eval-root", type=Path,
                    default=Path.home() / "data/dcase2026t2/eval/raw")
    ap.add_argument("--gt-root", type=Path,
                    default=Path.home() / "tools/dcase2026_task2_evaluator")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path.home() / "data/embeddings_2026")
    ap.add_argument("--teams-dir", type=Path, default=None,
                    help="default: <gt-root>/teams")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine when the E8 embedding cache is warm")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    gt_root = args.gt_root.expanduser()
    teams_dir = (args.teams_dir or gt_root / "teams").expanduser()
    embedder = load_embedder("beats", args.ckpt, args.device)
    cfg_by_name = {c.name: c for c in build_extended_configs()}

    for policy, channel in (("ch0", 0), ("mix", None)):
        configs = [cfg_by_name[n] for n in SYSTEMS[policy]]
        outs = {c.name: teams_dir / f"daco-{policy}" / c.name for c in configs}
        for o in outs.values():
            o.mkdir(parents=True, exist_ok=True)

        for machine in EVAL_MACHINES_2026:
            clips = list_clips(eval_root, machine, gt_root=gt_root)
            train = [c for c in clips if c.split == "train"]
            test = [c for c in clips if c.split == "test"]
            X_train = cached_embeddings(
                embedder, train, args.cache_dir.expanduser(),
                f"2026eval_{machine}_train_{policy}", args.device,
                model_tag=embedder.tag, channel=channel)
            X_test = cached_embeddings(
                embedder, test, args.cache_dir.expanduser(),
                f"2026eval_{machine}_test_{policy}", args.device,
                model_tag=embedder.tag, channel=channel)
            train_domains = np.array([c.domain for c in train])
            loo_knn_by_k = {k: loo_knn_scores(X_train, k=k) for k in (1, 2, 4)}

            for cfg in configs:
                scores = config_scores_ext(X_train, train_domains, X_test,
                                           cfg, loo_knn_by_k, {})
                # label-free threshold: train bank pushed through the same
                # transform (self-match makes it slightly conservative; only
                # F1-type metrics depend on it)
                train_scores = config_scores_ext(X_train, train_domains,
                                                 X_train, cfg, loo_knn_by_k, {})
                thr = float(np.percentile(train_scores, 90))
                out = outs[cfg.name]
                with open(out / f"anomaly_score_{machine}_section_00_test.csv",
                          "w") as f:
                    for c, s in zip(test, scores):
                        f.write(f"{c.path.name},{s:.10f}\n")
                with open(out / f"decision_result_{machine}_section_00_test.csv",
                          "w") as f:
                    for c, s in zip(test, scores):
                        f.write(f"{c.path.name},{int(s > thr)}\n")
            print(f"[{policy}] {machine}: exported "
                  f"{len(test)} clips x {len(configs)} systems", flush=True)

    print(f"per-clip scores written under {teams_dir}/daco-<policy>/")


if __name__ == "__main__":
    main()
