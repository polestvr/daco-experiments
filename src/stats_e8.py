"""Clustered statistics for the E8 (DCASE 2026) forward test, matching the
apparatus used for the 2023/2024/2025 cross-year study (stats_yearly.py):

  * family-block bootstrap CIs and family-mean exact permutation p for
    rho(criterion, eval Omega) and rho(dev Omega, eval Omega), per channel
    policy, over the 51-config 2026 grid;
  * machine-level jackknife CIs for the decisive Omega deltas
    (guarded - dev pick, guarded - fixed default) over the five 2026
    evaluation machines.

Reads results/e8_2026_grid.csv, results/e8_2026_per_machine_<policy>.csv and
the frozen dev grid results/prereg2026_grid.csv (for the pick identities).
Writes results/stats_e8_2026.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_e8_2026 import FIXED_DEFAULT, guarded_pick
from stats_yearly import (RESULTS, block_bootstrap_ci, family_mean_perm_p,
                          family_of, jackknife_delta)


def main() -> None:
    grid_all = pd.read_csv(RESULTS / "e8_2026_grid.csv")
    dev_all = pd.read_csv(RESULTS / "prereg2026_grid.csv")
    out = {}

    for policy in ("ch0", "mix"):
        grid = grid_all[grid_all.policy == policy].copy()
        grid["family"] = grid.config.map(family_of)
        pm = pd.read_csv(RESULTS / f"e8_2026_per_machine_{policy}.csv")
        machines = sorted(pm.machine.unique())

        dev = dev_all[dev_all.policy == policy][
            ["config", "dev_omega", "crit_dev"]]
        g_pick = guarded_pick(dev)
        d_pick = dev.loc[dev.dev_omega.idxmax()].config

        out[policy] = {
            "guarded_pick": g_pick,
            "dev_pick": d_pick,
            "fixed_default": FIXED_DEFAULT,
            "rho_criterion": {
                "block_ci95": block_bootstrap_ci(grid, "crit_dev", -1),
                **family_mean_perm_p(grid, "crit_dev", -1),
            },
            "rho_dev": {
                "block_ci95": block_bootstrap_ci(grid, "dev_omega", 1),
                **family_mean_perm_p(grid, "dev_omega", 1),
            },
            "delta_guarded_vs_dev":
                jackknife_delta(pm, g_pick, d_pick, machines),
            "delta_guarded_vs_default":
                jackknife_delta(pm, g_pick, FIXED_DEFAULT, machines),
        }

        r = out[policy]
        print(f"\n===== E8 2026 clustered stats [{policy}] =====")
        print(f"  rho(-criterion): block CI {r['rho_criterion']['block_ci95']}"
              f"  family-mean rho {r['rho_criterion']['rho_family_means']}"
              f"  perm p(1s/2s) {r['rho_criterion']['p_one_sided']}"
              f"/{r['rho_criterion']['p_two_sided']}")
        print(f"  rho(dev)       : block CI {r['rho_dev']['block_ci95']}"
              f"  family-mean rho {r['rho_dev']['rho_family_means']}"
              f"  perm p(1s/2s) {r['rho_dev']['p_one_sided']}"
              f"/{r['rho_dev']['p_two_sided']}")
        print(f"  guarded-dev    : {r['delta_guarded_vs_dev']['delta']:+.2f}"
              f"  CI {r['delta_guarded_vs_dev']['ci95']}")
        print(f"  guarded-default: {r['delta_guarded_vs_default']['delta']:+.2f}"
              f"  CI {r['delta_guarded_vs_default']['ci95']}")

    path = RESULTS / "stats_e8_2026.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
