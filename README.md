# DACo — Training-Free Model Selection and Domain-Aware Score Calibration for First-Shot Anomalous Sound Detection

Code and released artifacts for the experiments in the paper of the same
title. DACo is the method, evaluated on DCASE Challenge Task 2 (2023--2026).

The method is a training-free post-hoc layer over frozen audio embeddings:
per-domain quantile calibration with a pooling strength *m* (tracing the
source/target balance frontier) and a label-free cross-validated
domain-balance criterion for configuration selection, paired with a
development-side viability veto. All experiments run on a single consumer
GPU; every score cycle after embedding extraction takes ~0.15 s per machine
on CPU.

## Repository layout

```
src/
  daco/                 core library
    data.py             DCASE filename parsing, eval ground-truth injection
    extract.py          embedding extraction with on-disk cache
    embedders.py        BEATs / EAT / PANNs behind one interface
    backends.py         kNN, selective Mahalanobis, GMM scorers
    calibrate.py        LOO scores, per-domain quantile maps, shrinkage
    criteria.py         CV domain-balance criterion, separability
    ldn.py              local-density normalization (reimplementation)
    metrics.py          official DCASE metrics (differential-tested)
  run_e1.py run_e2.py run_e3.py run_e45.py run_e6.py   experiments E1-E6 (DCASE 2025)
  run_replication.py            E7: DCASE 2023 / 2024 replication
  run_prereg2026.py             frozen DCASE 2026 selection
  paper_stats.py                clustered bootstrap CIs, permutation tests
  stats_intervals.py            jackknife / clip-level CIs, partials
  stats_yearly.py               per-year clustered statistics
  check_seed_stability.py       criterion seed stability
  check_fpr_coverage.py         empirical per-domain FPR validation
  test_metrics_vs_evaluator.py  differential test vs official evaluator
  dump_diagnostics.py           per-machine diagnostics
  export_scores.py              evaluator-format per-clip score export
  analyze_e45.py                guarded-selection analysis
  make_figures.py, make_appendix_table.py   paper figures and tables
  backbones/            vendored model code (BEATs, PANNs; MIT)
scripts/
  download_data.sh      datasets, checkpoints, evaluator repositories
  fetch_2023_2024.sh    2023/2024 datasets only
results/                all CSV/JSON artifacts behind every number in the paper
release/                per-clip scores (evaluator format) for every table row,
                        configuration manifest, checkpoint SHA-256 pins,
                        environment lock
PREREGISTRATION.md      frozen DCASE 2026 forward test (configuration + hashes)
```

## Setup

Python >= 3.11 with an NVIDIA GPU (16 GB is ample; extraction peaks < 1 GB).

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # ~20 GB: DCASE 2023-2026 + checkpoints
```

Datasets and checkpoints live outside the repository (default `~/data`,
`~/models`, `~/tools`); exact Zenodo records and SHA-256 pins are in
`release/checkpoints.json` and the paper's Data Availability section.

## Reproducing the paper

| Result | Command |
|---|---|
| E1 baselines (Table: baseline reproduction) | `python src/run_e1.py --data-root ~/data/dcase2025t2/dev/raw --ckpt ~/models/beats/BEATs_iter3_plus_AS2M.pt --cache-dir <cache>` |
| E2 balance frontier (Fig. 1) | `python src/run_e2.py ...same args...` |
| E3 transfer study (Fig. 3, Table I) | `python src/run_e3.py --dev-root ... --eval-root ~/data/dcase2025t2/eval/raw --gt-root ~/tools/dcase2025_task2_evaluator --ckpt ... --cache-dir <cache>` |
| E4/E5 backbones + LDN (Tables II, III) | `python src/run_e45.py --backbone {beats,eat,panns} ...` |
| E6 efficiency (Table V) | `python src/run_e6.py ...` |
| E7 replication (Table IV) | `python src/run_replication.py --year {2023,2024}` |
| 2026 frozen selection | `python src/run_prereg2026.py` |
| Every statistic quoted in the text | `python src/paper_stats.py && python src/stats_intervals.py && python src/stats_yearly.py` |
| Official-evaluator verification | `python src/export_scores.py --config <name> ...` then run `dcase2025_task2_evaluator.py` on `release/scores/` |
| Figures / appendix table | `python src/make_figures.py --fig-dir <paper>/figures` |

Everything is deterministic given the criterion seed (0; stability over
seeds 0-9 reported by `check_seed_stability.py`). A full 51-configuration x
15-machine study runs in under four minutes per backbone on cached
embeddings.

## Main results (DCASE 2025 evaluation set, official score)

| Selection rule | Eval score |
|---|---|
| best-on-development configuration (standard practice) | 55.83 |
| label-free criterion + viability veto | **61.05** |
| official baselines (Selective Mahalanobis / MSE) | 56.51 / 54.43 |

Development score never predicts evaluation score (three challenge years);
the criterion's demonstrated transfer is concentrated in 2025 -- see the
paper for the clustered-uncertainty analysis and the scope of the claim.

## Pre-registered DCASE 2026 forward test

`PREREGISTRATION.md` freezes the selected configuration (per channel
policy), the veto rule, criterion seeds, and SHA-256 hashes of the selection
artifacts -- committed before the per-clip 2026 evaluation ground truth
became available. The frozen pick will be scored with the official evaluator
as soon as the ground truth is published, and the outcome reported whatever
it is.

## Third-party code and licenses

- `src/backbones/beats/` -- BEATs model code from
  [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) (MIT).
- `src/backbones/panns/` -- PANNs model code from
  [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (MIT).
- EAT is loaded from the authors' Hugging Face export
  (`worstchan/EAT-base_epoch30_pretrain`, MIT).
- Local-density normalization is reimplemented from the published equations
  of Wilkinghoff et al.; the AGPL reference implementation was not used.
- DCASE datasets are CC BY-NC-SA 4.0 (Zenodo); evaluation ground truth from
  the official `nttcslab/dcase202{3,4,5}_task2_evaluator` repositories.

Project code is released under the MIT License (see `LICENSE`).
