"""Generate the per-machine appendix table (LaTeX) for the paper.

Per machine (BEATs backbone): AUC_s / AUC_t / pAUC for raw kNN, DACo
(conf-soft m=0 k=1), and the guarded pick (DACo on LDN-ratio K=1), on both
dev and eval machine sets. Writes the tabular body to the paper repo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONFIGS = [("raw-k1", "raw"), ("conf-soft-m0-k1", "daco"),
           ("conf-soft-m0-ldnK1", "stack")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pm = pd.read_csv(args.results / "e45_beats_per_machine.csv")
    lines = []
    for split in ("dev", "eval"):
        sub = pm[pm.split == split]
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{'development' if split == 'dev' else 'evaluation'} machines}}}}\\")
        for machine in sorted(sub.machine.unique(), key=str.lower):
            cells = [machine.replace("_", r"\_")]
            for cfg, _ in CONFIGS:
                r = sub[(sub.machine == machine) & (sub.config == cfg)].iloc[0]
                cells += [f"{r.auc_source:.1f}", f"{r.auc_target:.1f}",
                          f"{r.pauc:.1f}"]
            lines.append(" & ".join(cells) + r" \\")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"written {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
