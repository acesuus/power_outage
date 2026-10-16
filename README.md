# Puerto Princesa power outage research data and recloser candidates

Research snapshot of the **XGBoost/history frozen and monthly candidates** from
`thesis_pipeline_recloser_suite/runs/recloser_suite_first`. This is the experiment
with **57 positive recloser-days**, not the later 2024 split experiment.

## Recorded comparison

Same 2,098 eligible recloser-days, including 57 recorded positive recloser-days
on 41 dates, January 1 to September 21, 2026. Raw scores, earlier-learned FPR20 policy:

| Training approach | Detected /57 | False warnings | Precision |
|---|---:|---:|---:|
| Frozen, expanding history | 44 | 1,140 | 3.7% |
| Monthly, last 12 months | 13 | 623 | 2.0% |
| Monthly, last 24 months | 17 | 649 | 2.6% |
| Monthly, expanding history | 2 | 175 | 1.1% |

The monthly 12-month candidate with its earlier balanced-accuracy rule detects
42/57 but produces 1,255 false warnings. FPR20 is learned on earlier calibration
data; it does **not** guarantee a 20% false-positive rate on later data. These
are not fixed 0.30 or 0.50 cutoffs. The frozen raw FPR20 cutoff is 0.04223041981458664.

These are exploratory candidate comparisons. The examined 2026 scores do not
establish the history candidate as the development-selected main model or as a
validated deployment model.

## Find the files

Original folder layout is retained so code imports, provenance paths and saved
model objects remain usable. Start with:

| Content | Location |
|---|---|
| Exact interruption log used (1,369 posts) | [paleco_unscheduled_puerto_princesa_city_only.csv](paleco_unscheduled_puerto_princesa_city_only.csv) |
| Earlier cleaned and upstream logs | `paleco_unscheduled_puerto_princesa_cleaned.csv`, `paleco_unscheduled_puerto_princesa.csv` |
| Original city weather reference | [daily_weather_data.csv](daily_weather_data.csv) |
| Per-barangay daily historical weather | [weather_data/barangay_daily_weather.csv](weather_data/barangay_daily_weather.csv) |
| Barangay coordinates, mappings and download method | `weather_data/` |
| Monthly vegetation | [vegetation_features_monthly.csv](vegetation_features_monthly.csv) |
| Equipment flags and dictionary | `equipment_analysis/` |
| Daily rainfall aggregates and download manifest | `thesis_pipeline_v3/data/rainfall/` |
| Engineered feature panel, dictionaries and cohort membership | `thesis_pipeline_recloser_suite/runs/recloser_suite_first/02_features/` |
| Original label evidence and barangay-to-recloser proxy mapping | `thesis_pipeline_recloser/runs/recloser9_validated_first/01_audit/` and `02_features/` |
| Final frozen history model and cutoffs | `thesis_pipeline_recloser_suite/runs/recloser_suite_first/04_frozen_selection_fit/models/xgboost__history__all/` |
| Monthly fitted candidates | `thesis_pipeline_recloser_suite/runs/recloser_suite_first/06_monthly_evaluation/models/YYYY-MM-01/xgboost__history__{12m,24m,all}/` |
| Focused comparison, monthly thresholds and counts | [Focused RESULTS.md](thesis_pipeline_recloser_suite/analysis/recloser_suite_first/monthly_xgboost_history/RESULTS.md) |
| Data guide and feature timing | [docs/DATA_AND_FEATURES.md](docs/DATA_AND_FEATURES.md) |
| Reproduction steps | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| CSV row counts and field lists | [docs/csv_catalog.json](docs/csv_catalog.json) |

There are **77 fitted history model bundles**: four frozen development fits,
one final frozen fit, 45 monthly development/warmup fits and 27 monthly evaluation
fits. Other algorithms' binaries are omitted. Original broad-suite score tables
and reports are retained for context and can mention candidates outside this
export. Original completion manifests describe the original full suite; use the
package manifest for this curated export's contents.

## Verify or refit

Use Python 3.12 (the original runtime) and the recorded versions in `requirements.txt`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python verify_package.py
```

Verification checks exact source hashes, reloads all 77 models, reproduces saved
probabilities and warning decisions, and recounts the table above. The committed
[verification result](docs/package_verification.json) records the packaging check.
The five frozen fits were also freshly retrained as an entrypoint check; their
selected trials and final probabilities matched the archived run exactly
([refit check](docs/frozen_refit_check.json)).

Optional refits in new directories:

```bash
python train_history.py --approach frozen --output refits/frozen_history
python train_history.py --approach monthly --output refits/monthly_history
```

Daily weather and monthly vegetation are included as project datasets and were
used by other feature families. **This history candidate uses neither.** Its nine
raw inputs are recloser identity, calendar seasonality, recent outage history and
equipment-fault history. No target-day weather is needed to score this candidate.

The source log contains publicly reported interruptions, not a complete utility
operations register. A no-record day is not proof of uninterrupted service. The
primary evaluation's uncertainty mask is retrospective, and the spatial mapping
is a text-derived proxy, not official electrical topology. See the data guide.

Source and copied-artifact bytes are listed in `docs/source_manifest.json`;
`SHA256SUMS.json` covers the complete published package except itself and `.git`.
No new license is assigned to third-party source data in this snapshot.
