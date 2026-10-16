# Recloser-day experiment suite

Repeat the city-day experiment families on the nine-recloser dataset: algorithm baselines, frozen versus
monthly training, 12/24/all-history windows, feature combinations, calibration modes, five warning rules,
per-unit results and uncertainty-mask sensitivity. Keep all previous experiments unchanged.

The primary target is a recorded interruption affecting a named recloser service area on D, not necessarily
failure of its physical recloser. Weather/vegetation follow the existing frozen announcement-derived proxy
mapping; no official GIS or equipment inventory is invented. Mandatory equipment history remains 7/14 days.

Run from `E:\PALECO\final` with the already installed V3 environment:

```powershell
& '.\thesis_pipeline_v3\.venv\Scripts\python.exe' -X utf8 '.\thesis_pipeline_recloser_suite\test_suite.py'
& '.\thesis_pipeline_v3\.venv\Scripts\python.exe' -X utf8 '.\thesis_pipeline_recloser_suite\run_pipeline.py' --run-id recloser_suite_first
```

`--until 2` prepares labels/features/cohorts; `--resume` continues an unfinished run. Every model has a
verified checkpoint. Completed runs are immutable; code/config changes require a new run ID. No new
downloads or installations. The suite runs nine scientific safeguards before the requested full experiment.

| Step | Script | Saved evidence |
|---|---|---|
| 1 | 01_audit_sources.py | Input/dependency/code hashes, environment, original artifact preservation, source support |
| 2 | 02_features_and_cohorts.py | Per-recloser panel, exact dictionary/families, spatial audit, frozen/monthly role memberships |
| 3 | 03_frozen_development.py | Four earlier-quarter trials, fitted models, calibration, all warnings and predictions |
| 4 | 04_select_and_fit_frozen.py | Earlier-only frozen ranking/choices, final pre-2026 fitting and fixed thresholds |
| 5 | 05_monthly_development.py | Warmup/2025 monthly models, window comparisons, earlier out-of-sample calibration pools |
| 6 | 06_select_and_evaluate_monthly.py | Earlier-only monthly/feature choices, sequential 2026 predictions, thresholds and sensitivity |
| 7 | 07_evaluate_frozen_and_compare.py | Same-row frozen evaluation, per-unit scores, original references, date-block intervals |
| 8 | 08_verify_and_report.py | 687 model reloads, fitting-only checks, aggregate score verification, preservation, figures/report |

There are 27 frozen and 23 monthly base procedures (687 refits;2,748 inner tuning trials). Broad frozen
baselines include logistic, spline logistic, XGBoost, CatBoost and random forest. Monthly windows compare
logistic/XGBoost; all feature groups are compared for frozen and 12-month monthly XGBoost. This follows
the earlier experiment designs, not every possible model x feature x window combination.

Read [METHODOLOGY.md](METHODOLOGY.md). The final report is
`runs/<run-id>/08_report/THESIS_RECLOSER_SUITE_REPORT.md`. Primary test: 2,098 eligible unit-days/57 positives;
sensitivity: 2,376 rows/57 positives. The target-day uncertainty mask is retrospective, not operational.
All 2026 results remain exploratory. Same-D historical weather and own-month vegetation retain the
approved retrospective definition and do not establish operational day-ahead input availability.
