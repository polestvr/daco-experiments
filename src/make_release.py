"""Build the release bundle for the paper (repo release/ + Zenodo deposit).

Contents:
  release/scores/<config>/<split>/anomaly_score_<machine>_section_00_test.csv
      per-clip scores (evaluator format) for every configuration that appears
      in a paper table, dev and eval machines, BEATs backbone
      + decision_result_*.csv (label-free 90th-percentile threshold)
  release/manifest.json      all 51 grid configurations with exact parameters
                             + criterion protocol + data provenance
  release/checksums.txt      SHA-256 of the three backbone checkpoints
  release/environment.txt    python/torch versions + pip freeze
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator
from daco.criteria import _base_and_loo
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cache_path_for
from run_e45 import build_extended_configs

REPO = Path(__file__).resolve().parent.parent
RELEASE = REPO / "release"
CACHE = Path.home() / "data/dcase2025t2/embeddings/multi"
MODEL_TAG = "BEATs_iter3_plus_AS2M"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"
EVAL_ROOT = Path.home() / "data/dcase2025t2/eval/raw"
GT_ROOT = Path.home() / "tools/dcase2025_task2_evaluator"

TABLE_CONFIGS = ["raw-k1", "conf-soft-m0-k1", "conf-soft-m0-k2",
                 "conf-hard-m0-k1", "conf-hard-m0-k2", "conf-soft-m0-ldnK1",
                 "conf-soft-m30-ldnK1", "ldn-ratio-K1", "ldn-ratio-K16",
                 "ldn-diff-K1", "ldn-diff-K16", "zscore-soft-k1",
                 "ratio-soft-k1"]

CKPTS = {
    "BEATs_iter3_plus_AS2M.pt":
        Path.home() / "models/beats/BEATs_iter3_plus_AS2M.pt",
    "Cnn14_16k_mAP=0.438.pth":
        Path.home() / "models/panns/Cnn14_16k_mAP=0.438.pth",
}
CKPT_URLS = {
    "BEATs_iter3_plus_AS2M.pt":
        "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt",
    "Cnn14_16k_mAP=0.438.pth":
        "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
    "EAT-base_epoch30_pretrain (HF repo)":
        "https://huggingface.co/worstchan/EAT-base_epoch30_pretrain",
}


def load(machine, split):
    root = DEV_ROOT if machine in DEV_MACHINES else EVAL_ROOT
    gt = None if machine in DEV_MACHINES else GT_ROOT
    clips = [c for c in list_clips(root, machine, gt_root=gt)
             if c.split == split]
    d = np.load(cache_path_for(clips, CACHE, f"{machine}_{split}", MODEL_TAG),
                allow_pickle=True)
    return clips, d["X"], d["domain"]


def main() -> None:
    cfg_by = {c.name: c for c in build_extended_configs()}

    # ---- per-clip scores -------------------------------------------------
    for name in TABLE_CONFIGS:
        cfg = cfg_by[name]
        for machine in DEV_MACHINES + EVAL_MACHINES:
            split_dir = "dev" if machine in DEV_MACHINES else "eval"
            out = RELEASE / "scores" / name / split_dir
            out.mkdir(parents=True, exist_ok=True)
            train, X_tr, dom_tr = load(machine, "train")
            test, X_te, _ = load(machine, "test")
            base, loo = _base_and_loo(X_tr, X_te, cfg)
            if cfg.assignment is None:
                scores, cal_train = base, loo
            else:
                cal = DACoCalibrator(cfg.assignment, cfg.method,
                                     prior_strength=cfg.prior_strength).fit(
                    X_tr, dom_tr, loo)
                scores = cal.transform(base, cal.domain_weights(X_te))
                cal_train = cal.transform(loo, cal.domain_weights(X_tr))
            thr = float(np.percentile(cal_train, 90))
            with open(out / f"anomaly_score_{machine}_section_00_test.csv",
                      "w") as f:
                for c, s in zip(test, scores):
                    f.write(f"{c.path.name},{s:.10f}\n")
            with open(out / f"decision_result_{machine}_section_00_test.csv",
                      "w") as f:
                for c, s in zip(test, scores):
                    f.write(f"{c.path.name},{int(s > thr)}\n")
        print(f"scores exported: {name}", flush=True)

    # ---- manifest --------------------------------------------------------
    manifest = {
        "grid_configurations": [asdict(c) for c in build_extended_configs()],
        "criterion_protocol": {"S_splits": 10, "holdout_fraction": 0.5,
                               "seed_primary": 0,
                               "seed_stability_range": "0-9",
                               "statistic": "two-sample KS between calibrated "
                                            "held-out source and target "
                                            "normal scores"},
        "viability_veto": "exclude bottom decile of configurations by "
                          "development omega, then argmin criterion",
        "data": {
            "dev": "zenodo.org/records/15097779",
            "additional_train": "zenodo.org/records/15392814",
            "eval_test": "zenodo.org/records/15519362",
            "eval_ground_truth":
                "github.com/nttcslab/dcase2025_task2_evaluator",
        },
        "checkpoints": CKPT_URLS,
    }
    RELEASE.joinpath("manifest.json").write_text(
        json.dumps(manifest, indent=2))

    # ---- checksums -------------------------------------------------------
    lines = []
    for label, p in CKPTS.items():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {label}  ({CKPT_URLS[label]})")
    hf = Path.home() / ".cache/huggingface/hub"
    for st in sorted(hf.glob(
            "models--worstchan--EAT-base_epoch30_pretrain/**/model.safetensors")):
        h = hashlib.sha256(st.read_bytes()).hexdigest()
        lines.append(f"{h}  EAT-base_epoch30_pretrain/model.safetensors")
    RELEASE.joinpath("checksums.txt").write_text("\n".join(lines) + "\n")

    # ---- environment -----------------------------------------------------
    import torch
    env = [f"python {sys.version.split()[0]}",
           f"torch {torch.__version__}",
           f"cuda_available {torch.cuda.is_available()}", ""]
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    RELEASE.joinpath("environment.txt").write_text(
        "\n".join(env) + freeze)
    print(f"release bundle at {RELEASE}")


if __name__ == "__main__":
    main()
