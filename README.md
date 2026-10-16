# Puerto Princesa power outage research data and recloser candidates

Featured model: **frozen XGBoost/history with the recent-outage trend**, using
expanding historical training and no monthly refitting. Its source is the
targeted history feature experiment on the same **57 positive recloser-days**.
Original experiment data and archived model records remain available below.

## Recent-outage-trend model results

Same 2,098 eligible recloser-days, including 57 recorded positive recloser-days
on 41 dates, January 1 to September 21, 2026. Raw scores, earlier-learned FPR20 policy:

| Training approach | AUC-ROC | F1-score | Precision | Recall | Brier score |
|---|---:|---:|---:|---:|---:|
| Frozen XGBoost/history + recent-outage trend | 0.6343 | 0.0768 | 4.07% | 66.67% | 0.026588 |

**[Full metric report and calibration analysis](reports/recent_outage_trend/RESULTS.md)**
includes reliability diagrams, raw versus earlier-calibrated probabilities, Brier
reference comparisons and all saved frozen warning policies. Under FPR20 the trend
model detects **38 of 57** positive rows, with **895 false warnings**, 19 missed
positives and 1,146 true negatives.

FPR20 is learned on earlier calibration
data; it does **not** guarantee a 20% false-positive rate on later data. These
are not fixed 0.30 or 0.50 cutoffs. This model's raw FPR20 cutoff is **0.04553172364830971**.

This candidate is featured at the user's request after exploratory evaluation.
It was not the feature set selected by the earlier development comparison.
The AUC increase over the original baseline is small and uncertain; the
[full report](reports/recent_outage_trend/RESULTS.md#original-baseline-comparison)
retains that comparison and its detection/false-warning tradeoff.

## Added recent-outage trend

The model uses the original nine inputs plus:

```text
outage_rate_recent3d_minus_previous4d =
    outage_days_prev3d / 3
    - (outage_days_prev7d - outage_days_prev3d) / 4
```

It compares the daily outage rate in the last three days with the preceding
four days, using counts available by D-1 20:00 PHT. Positive values indicate
more concentrated recent recorded outages. It adds no new observations.
The [feature list](models/recent_outage_trend/features.json) documents all ten inputs.

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
| **Featured trend model, cutoffs and evaluation predictions** | [models/recent_outage_trend/](models/recent_outage_trend/) |
| Archived original baseline model and cutoffs | `thesis_pipeline_recloser_suite/runs/recloser_suite_first/04_frozen_selection_fit/models/xgboost__history__all/` |
| Monthly fitted candidates | `thesis_pipeline_recloser_suite/runs/recloser_suite_first/06_monthly_evaluation/models/YYYY-MM-01/xgboost__history__{12m,24m,all}/` |
| Focused comparison, monthly thresholds and counts | [Focused RESULTS.md](thesis_pipeline_recloser_suite/analysis/recloser_suite_first/monthly_xgboost_history/RESULTS.md) |
| Featured model metrics, calibration and Brier analysis | [Trend metric report](reports/recent_outage_trend/RESULTS.md) |
| Archived original baseline metric report | [Baseline report](reports/candidate_metrics/RESULTS.md) |
| Data guide and feature timing | [docs/DATA_AND_FEATURES.md](docs/DATA_AND_FEATURES.md) |
| Reproduction steps | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| CSV row counts and field lists | [docs/csv_catalog.json](docs/csv_catalog.json) |

The featured package includes **five trend-model fits**: four earlier chronological
development fits and one final frozen fit. The archive contains **77 original
history model bundles**: four frozen development fits,
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
python verify_recent_outage_trend.py
# Optional: verify the original archived experiments too.
python verify_package.py
```

The trend verifier reloads its five fits, reproduces probabilities and warning
decisions, and recounts the featured table above. See the
[trend verification result](reports/recent_outage_trend/verification.json).
The archive verifier checks exact source hashes and reloads the original 77 fits.
The committed
[verification result](docs/package_verification.json) records the packaging check.
The five frozen fits were also freshly retrained as an entrypoint check; their
selected trials and final probabilities matched the archived run exactly
([refit check](docs/frozen_refit_check.json)).
The five featured trend fits were also freshly retrained and reproduced their
selected trials, probabilities and warnings exactly
([trend refit check](reports/recent_outage_trend/refit_verification.json)).

Optional refits in new directories:

```bash
python train_history.py --approach frozen --output refits/frozen_history
python train_history.py --approach monthly --output refits/monthly_history
python train_recent_outage_trend.py --output refits/recent_outage_trend
```

Daily weather and monthly vegetation are included as project datasets and were
used by other feature families. **The featured trend candidate uses neither.** Its ten
raw inputs are recloser identity, calendar seasonality, recent outage history and
equipment-fault history, including the derived recent-outage trend. No target-day
weather is needed to score this candidate.

The source log contains publicly reported interruptions, not a complete utility
operations register. A no-record day is not proof of uninterrupted service. The
primary evaluation's uncertainty mask is retrospective, and the spatial mapping
is a text-derived proxy, not official electrical topology. See the data guide.

Source and copied-artifact bytes are listed in `docs/source_manifest.json`;
`SHA256SUMS.json` covers the complete published package except itself and `.git`.
No new license is assigned to third-party source data in this snapshot.
