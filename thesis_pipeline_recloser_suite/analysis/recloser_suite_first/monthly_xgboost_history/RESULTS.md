# Monthly retraining: XGBoost with history inputs

These experiments were already completed within the broad recloser suite. This focused review extracts their saved results and independently recounts them. No model was refitted and no threshold or window was selected from the examined 2026 results.

## Same evaluation population

Nine reclosers, 2,098 eligible recloser-days, 57 recorded positive recloser-days on 41 dates, January 1–September 21, 2026. Each warning concerns one recloser on one day.

## Primary comparison: raw scores and earlier-learned FPR20 rules

| Training procedure | Detected /57 | False warnings | Missed | Recall | Precision | Actual FPR | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen, expanding history | 44 | 1140 | 13 | 77.2% | 3.7% | 55.9% | 0.0420 |
| Monthly, 12-month window | 13 | 623 | 44 | 22.8% | 2.0% | 30.5% | 0.0362 |
| Monthly, 24-month window | 17 | 649 | 40 | 29.8% | 2.6% | 31.8% | 0.0273 |
| Monthly, expanding history | 2 | 175 | 55 | 3.5% | 1.1% | 8.6% | 0.0334 |

Monthly retraining reduces warnings under these earlier FPR20 rules, but also misses many more positives. These are observed operating points, not a comparison at matched future warning burden. FPR20 constrains earlier calibration, not the future false-positive rate.

## Different warning rules for the 12-month model

| Rule learned earlier | Detected /57 | False warnings | Recall | Precision |
|---|---:|---:|---:|---:|
| fpr20 | 13 | 623 | 22.8% | 2.0% |
| fpr10 | 5 | 264 | 8.8% | 1.9% |
| balanced_accuracy | 42 | 1255 | 73.7% | 3.2% |
| f1 | 35 | 1047 | 61.4% | 3.2% |
| fixed050 | 0 | 0 | 0.0% | undefined |

The 12-month model can detect 42/57 under the earlier balanced-accuracy rule, but that produces 1,255 false warnings. It therefore does not demonstrate keeping the frozen model’s high detection count while reducing false warnings. All three monthly windows issue no warnings at a raw fixed 0.50 cutoff.

## Training windows and inputs

- **12 months:** refit each month on the recent eligible 12-month history.
- **24 months:** refit each month on the recent eligible 24-month history.
- **Expanding:** refit each month on all eligible history from September 30, 2022.
- Each monthly fit uses earlier outcomes available before the target month and a 14-day incident-maturity buffer. Earlier completed 2026 months may enter later monthly fits after availability/maturity checks. Current/future target-month labels never enter fitting or calibration.
- Cutoffs are learned anew each month from genuinely earlier out-of-sample prediction pools. They are not a single 0.30 cutoff and need not equal the frozen cutoff.
- The history family has nine raw inputs, listed below. It uses no weather or vegetation. Training-window length and the daily history-feature lookback are separate choices.

- `recloser_id`
- `day_of_week`
- `day_of_year_sin`
- `day_of_year_cos`
- `outage_days_prev3d`
- `outage_days_prev7d`
- `days_since_last_outage_capped7`
- `equipment_fault_days_prev7d`
- `equipment_fault_days_prev14d`

The outage lookbacks remain 3/7 days and equipment lookbacks 7/14 days. A 12-month training window does not change those lookbacks into 12-month feature accumulations.

## Monthly FPR20 cutoffs, raw score scale

| Target month | 12-month history | 24-month history | Expanding history |
|---|---:|---:|---:|
| 2026-01 | 0.027708 | 0.032731 | 0.042340 |
| 2026-02 | 0.027708 | 0.035045 | 0.040078 |
| 2026-03 | 0.030104 | 0.035045 | No-warning rule |
| 2026-04 | 0.031147 | 0.035045 | No-warning rule |
| 2026-05 | 0.030832 | 0.041301 | 0.050171 |
| 2026-06 | 0.030832 | 0.047382 | 0.095134 |
| 2026-07 | 0.030832 | No-warning rule | 0.095134 |
| 2026-08 | No-warning rule | 0.076815 | 0.125747 |
| 2026-09 | 0.032060 | 0.076815 | No-warning rule |

## Interpretation and thesis reporting

The current evidence does not show that monthly XGBoost/history solves the false-warning problem. Under FPR20 the monthly 24-month window detects the most of the three examined monthly windows (17), while 12 months has higher AP (0.0362). Those are different criteria; no 2026 winner replaces the registered main procedure.

Within the three raw XGBoost/history windows, earlier 2025 FPR20 development recall was 18/89 for 24 months versus 17/89 for 12 months and expanding history. The suite-wide monthly selection remains logistic/full/expanding with intercept adjustment; this focused comparison does not replace it.

All 2026 results remain exploratory because this period was examined during earlier experiments. No-record rows are not verified uninterrupted power; primary eligibility uses a retrospective reporting mask. The later source audit did not change these 57 evaluation positives, and these original classifiers were not retrained on the audited source copy.

## Navigation

- [Every window, score scale and warning rule](all_window_and_warning_results.csv)
- [Month-by-month results](monthly_results.csv)
- [Actual monthly cutoffs](monthly_cutoffs.csv)
- [Earlier development results](earlier_development_results.csv)
- [Per-recloser results](per_recloser_results.csv)
- [Fitting end dates](fitting_dates.csv)
- [Verification](verification.json)
- [Complete training methodology](../../../METHODOLOGY.md)
- [January 2026 12-month model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__12m/model.joblib)
- [January 2026 24-month model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__24m/model.joblib)
- [January 2026 expanding model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__all/model.joblib)

Later month-specific models are stored under the corresponding target-month folder in `runs/recloser_suite_first/06_monthly_evaluation/models/`. There is a model per month and training window, not one fixed monthly model for the whole evaluation period.
