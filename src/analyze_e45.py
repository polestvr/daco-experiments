"""Guarded selection analysis over the per-backbone grids.

Selection rule: restrict to configurations whose development omega is within
delta of the development-best (a coarse viability screen using only labeled
DEVELOPMENT data — legal pre-challenge; it removes degenerate score maps that
a pure balance criterion cannot see), then pick the configuration minimizing
the label-free criterion computed on the development machines. Reports
sensitivity to delta.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--deltas", nargs="*", type=float, default=[2, 3, 5, 8])
    args = ap.parse_args()

    rows = []
    for bb in ("beats", "eat", "panns"):
        g = pd.read_csv(args.results / f"e45_{bb}_grid.csv")
        raw = g[g.config == "raw-k1"].iloc[0]
        best_dev = g.loc[g.dev_omega.idxmax()]
        oracle = g.loc[g.eval_omega.idxmax()]
        print(f"=== {bb} ===")
        print(f"  raw-k1                    eval {raw.eval_omega:6.2f}")
        print(f"  by dev omega: {best_dev.config:22s} eval {best_dev.eval_omega:6.2f}")
        for delta in args.deltas:
            viable = g[g.dev_omega >= g.dev_omega.max() - delta]
            pick = viable.loc[viable.crit_dev.idxmin()]
            print(f"  guarded crit d={delta:<3g}: {pick.config:20s} "
                  f"eval {pick.eval_omega:6.2f}  (viable {len(viable)}/{len(g)})")
            rows.append({"backbone": bb, "delta": delta, "pick": pick.config,
                         "eval_omega": pick.eval_omega})
        print(f"  oracle:       {oracle.config:22s} eval {oracle.eval_omega:6.2f}")

    pd.DataFrame(rows).to_csv(args.results / "e45_guarded_selection.csv",
                              index=False)
    print(f"\nwritten {args.results}/e45_guarded_selection.csv")


if __name__ == "__main__":
    main()
