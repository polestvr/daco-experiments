# Pre-registration: DCASE 2026 Task 2 forward test

**Date frozen:** 2026-07-03

**Timeline context.** At freezing time the DCASE 2026 evaluation *audio*
was already public (Zenodo record 20437238, published 2026-06-01) and the
team leaderboard had been announced (2026-06-30; 175 submissions, top
official score 70.24, MSE baseline 59.80). What did **not** exist at
freezing time -- and still does not as of 2026-07-05 -- is the **per-clip
evaluation ground truth** (anomaly and domain labels), which is what
offline scoring requires. The frozen configuration therefore could not
have been evaluated, directly or indirectly, at freezing time.
Verification rests on the SHA-256 artifact hashes below and on an
independent, immutable Zenodo archive of this repository snapshot
(version v1.0.0, DOI [10.5281/zenodo.21210904](https://doi.org/10.5281/zenodo.21210904)).

## Input

- DCASE 2026 Challenge Task 2 **Development** dataset, Zenodo record
  [19336329](https://zenodo.org/records/19336329) (7 machine types: ToyCar,
  ToyCarEmu, bearingEmu, fan, gearboxEmu, sliderEmu, valveEmu; two-channel
  16 kHz audio).
- Backbone: BEATs iter3+ AS2M (frozen; SHA-256 pinned in
  `release/checkpoints.json`).
- Pipeline: the 51-configuration grid, CV domain-balance criterion
  (S=10 splits, seed 0), and bottom-decile dev-Ω viability veto, exactly as
  released in `src/` (`src/run_prereg2026.py`).

## Channel policy

The 2026 data are two-channel (microphones at different distances from the
machine; channel semantics are not specified in the dataset description).
**Primary policy: channel 0.** Sensitivity policy: mean mixdown of both
channels. Both were fixed before selection was run.

## Frozen selections (output of `src/run_prereg2026.py`)

| Policy | Frozen configuration | dev Ω | criterion |
|---|---|---|---|
| **ch0 (primary)** | **`ratio-hard-k2`** (per-domain median-ratio, hard assignment, k=2) | 54.78 | 0.3997 |
| mix (sensitivity) | `conf-soft-m0-k1` (per-domain quantile calibration, m=0, soft, k=1) | 54.31 | 0.4042 |

Reference points frozen alongside: dev-Ω picks are `raw-k1` (57.05, ch0) and
`conf-hard-m300-k1` (56.13, mix); the fixed a-priori default is
`conf-soft-m0-k1` under both policies.

## Artifact hashes (SHA-256)

```
0670fed7ba6e40a9fa6ee8d180579e218d99015121d6caff701c1a55d98bc877  results/prereg2026_grid.csv
d26cddf725cf0810d100bb6d19dc9a6884ae9e78fd1cadfa5112314adb5b4321  results/prereg2026_per_machine_ch0.csv
7e1357f16bc023c91a4a56dd888d43ac0881d26fc849d099e29f2f6add6b5b6e  results/prereg2026_per_machine_mix.csv
```

## Evaluation plan (once the 2026 evaluation ground truth is released)

1. Extract BEATs embeddings for the 2026 evaluation machines under the same
   channel policies; no re-selection, no re-tuning of any kind.
2. Score the two frozen configurations above and the frozen reference points
   (dev-Ω picks, fixed default, raw kNN) with the official evaluator.
3. Report, per policy: the official Ω of the frozen pick vs the dev-Ω pick
   and the fixed default, plus rho(criterion, eval Ω) and
   rho(dev Ω, eval Ω) across the full 51-configuration grid -- extending the
   three-year transfer study (2023/2024/2025, `results/rep_*_grid.csv`,
   `results/e45_beats_grid.csv`) by a fourth, genuinely prospective point.
4. Report the outcome **whatever it is**; the hypothesis at stake is that
   label-free domain-balance selection transfers to unseen machines at least
   as reliably as development-set score.
