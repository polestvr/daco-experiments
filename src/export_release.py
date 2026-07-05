"""Build the release/ tree for the Zenodo DOI deposit.

release/
  scores/<config>/<split>/anomaly_score_<machine>_section_00_test.csv
      per-clip scores (evaluator format) for every configuration that appears
      in a table of the paper, on both dev and eval machines (BEATs backbone)
  manifest.json         all grid configurations with exact parameters
  checkpoints.json      checkpoint identifiers, URLs, SHA-256
  environment.txt       pip freeze + versions
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from daco.calibrate import DACoCalibrator
from daco.criteria import Config, _base_and_loo
from daco.data import DEV_MACHINES, EVAL_MACHINES, list_clips
from daco.extract import cache_path_for
from run_e45 import build_extended_configs

RELEASE = Path(__file__).resolve().parent.parent / "release"
CACHE = Path.home() / "data/dcase2025t2/embeddings/multi"
MODEL_TAG = "BEATs_iter3_plus_AS2M"
DEV_ROOT = Path.home() / "data/dcase2025t2/dev/raw"
EVAL_ROOT = Path.home() / "data/dcase2025t2/eval/raw"
GT_ROOT = Path.home() / "tools/dcase2025_task2_evaluator"

TABLE_CONFIGS = [
    "raw-k1", "conf-soft-m0-k1", "conf-soft-m0-k2", "conf-hard-m0-k1",
    "conf-hard-m0-k2", "conf-soft-m0-ldnK1", "conf-soft-m30-ldnK1",
    "ldn-ratio-K1", "ldn-ratio-K16", "ldn-diff-K1", "ldn-diff-K16",
    "zscore-soft-k1", "ratio-soft-k1",
]


def load_clips(machine, split):
    root = DEV_ROOT if machine in DEV_MACHINES else EVAL_ROOT
    gt = None if machine in DEV_MACHINES else GT_ROOT
    return [c for c in list_clips(root, machine, gt_root=gt)
            if c.split == split]


def load_X(clips, machine, split):
    return np.load(cache_path_for(clips, CACHE, f"{machine}_{split}",
                                  MODEL_TAG), allow_pickle=True)["X"]


def main() -> None:
    cfg_by = {c.name: c for c in build_extended_configs()}

    # ---- per-clip scores for every table configuration ---------------------
    for machine in DEV_MACHINES + EVAL_MACHINES:
        split_lbl = "dev" if machine in DEV_MACHINES else "eval"
        train = load_clips(machine, "train")
        test = load_clips(machine, "test")
        X_tr, X_te = load_X(train, machine, "train"), load_X(test, machine, "test")
        dom_tr = np.array([c.domain for c in train])
        for name in TABLE_CONFIGS:
            cfg = cfg_by[name]
            base, loo = _base_and_loo(X_tr, X_te, cfg)
            if cfg.assignment is None:
                scores = base
            else:
                cal = DACoCalibrator(cfg.assignment, cfg.method,
                                     prior_strength=cfg.prior_strength).fit(
                    X_tr, dom_tr, loo)
                scores = cal.transform(base, cal.domain_weights(X_te))
            out = RELEASE / "scores" / name / split_lbl
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"anomaly_score_{machine}_section_00_test.csv",
                      "w") as f:
                for c, s in zip(test, scores):
                    f.write(f"{c.path.name},{s:.10f}\n")
        print(f"{machine}: {len(TABLE_CONFIGS)} configs exported", flush=True)

    # ---- configuration manifest --------------------------------------------
    manifest = [{"name": c.name, "base": c.base, "density_K": c.density_K,
                 "assignment": c.assignment, "method": c.method,
                 "prior_strength": c.prior_strength, "k": c.k,
                 "in_paper_tables": c.name in TABLE_CONFIGS}
                for c in build_extended_configs()]
    (RELEASE / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # ---- checkpoint pins ----------------------------------------------------
    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    home = Path.home()
    eat_files = list((home / ".cache/huggingface/hub").glob(
        "models--worstchan--EAT-base_epoch30_pretrain/snapshots/*/model.safetensors"))
    ckpts = {
        "beats": {"file": "BEATs_iter3_plus_AS2M.pt",
                  "source": "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt",
                  "license": "MIT",
                  "sha256": sha256(home / "models/beats/BEATs_iter3_plus_AS2M.pt")},
        "panns": {"file": "Cnn14_16k_mAP=0.438.pth",
                  "source": "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
                  "license": "CC-BY-4.0",
                  "sha256": sha256(home / "models/panns/Cnn14_16k_mAP=0.438.pth")},
        "eat": {"file": "model.safetensors",
                "source": "https://huggingface.co/worstchan/EAT-base_epoch30_pretrain",
                "license": "MIT",
                "sha256": sha256(eat_files[0]) if eat_files else "NOT-FOUND"},
    }
    (RELEASE / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))

    # ---- environment ---------------------------------------------------------
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    import torch
    header = (f"python {sys.version.split()[0]}\n"
              f"torch {torch.__version__} cuda {torch.version.cuda}\n"
              f"gpu RTX 4070 Ti SUPER 16GB / WSL2 Ubuntu 24.04\n---\n")
    (RELEASE / "environment.txt").write_text(header + freeze)
    print("release/ complete")


if __name__ == "__main__":
    main()
