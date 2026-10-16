# Recloser-day experiment suite

The city-day experiment families were repeated for nine named recloser service areas using unchanged recloser labels, frozen geographic proxies and the exact earlier evaluation population. There are 2,098 eligible evaluated unit-days, 57 positive recloser-days on 41 dates, and nine reclosers. A no-record row is not verified uninterrupted power.

Earlier frozen probability-loss selection: **xgboost__vegetation__all__monotone_sigmoid**. Earlier monthly warning-oriented selection: **logistic__full__all__intercept_adjusted**. Monthly pooled development FPR allowance met: **True**. These choices are recorded before their 2026 evaluation, but 2026 was already examined before designing this suite.

## Main comparisons

| Procedure | Detected /57 | False warnings | Missed | Precision | Recall | FPR | AP | AUC | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen / xgboost__vegetation__all__monotone_sigmoid | 35 | 1063 | 22 | 3.2% | 61.4% | 52.1% | 0.0356 | 0.5593 | 0.1267 |
| Frozen / random_forest__full__all__monotone_sigmoid | 30 | 894 | 27 | 3.2% | 52.6% | 43.8% | 0.0322 | 0.5405 | 0.1297 |
| Frozen / random_forest__full__all__raw | 30 | 894 | 27 | 3.2% | 52.6% | 43.8% | 0.0322 | 0.5405 | 0.1296 |
| Frozen / xgboost__full__all__raw | 32 | 1010 | 25 | 3.1% | 56.1% | 49.5% | 0.0297 | 0.5414 | 0.1325 |
| Monthly / logistic__full__all__intercept_adjusted | 10 | 438 | 47 | 2.2% | 17.5% | 21.5% | 0.0259 | 0.5046 | 0.1292 |
| Monthly / xgboost__history__12m__raw | 13 | 623 | 44 | 2.0% | 22.8% | 30.5% | 0.0362 | 0.5715 | 0.1251 |
| Monthly / xgboost__full__12m__raw | 9 | 372 | 48 | 2.4% | 15.8% | 18.2% | 0.0279 | 0.5147 | 0.1259 |

All rows above use FPR20 thresholds learned on earlier data. Future FPR can exceed 20%. Frozen and monthly procedures are selected under different earlier criteria, as in the city experiments; do not attribute differences solely to updating. Their evaluation rows and labels match exactly.

## Scope of experiments

- Frozen algorithm baseline: logistic regression, small spline logistic regression, XGBoost, CatBoost and random forest on history, all-weather, compact-full and original 88-feature families.
- Frozen XGBoost feature combinations: rainfall, wind, temperature/humidity, vegetation, all-weather, local weather-vegetation products, dry-to-wet and both additions.
- Monthly logistic/XGBoost: history/full x 12-month/24-month/expanding histories. Monthly XGBoost additionally tests every feature combination at 12 months and the legacy 88-input family across all three windows. This repeats experiment families, not every possible Cartesian product.
- Frozen raw/nonnegative-sigmoid and monthly raw/intercept-only probability comparisons.
- FPR20, FPR10, balanced-accuracy, F1 and fixed-0.50 warning rules for every probability procedure.
- All-no-record-zero sensitivity with unchanged fitted models, plus per-recloser and source-support audits.

There are 27 frozen and 23 monthly base procedures, 135 frozen and 552 monthly model bundles (687 total), with 2,748 four-trial inner fits. No new source download or package installation occurs.

## Feature definitions

| Family | Inputs |
|---|---:|
| history | 9 |
| full | 37 |
| rainfall | 22 |
| wind | 14 |
| temperature_humidity | 11 |
| vegetation | 13 |
| all_weather | 30 |
| full_local | 39 |
| full_dry | 39 |
| full_local_dry | 41 |
| legacy_full | 88 |

Compact history includes recloser identity, categorical weekday, annual sine/cosine, earlier outage counts 3/7, recency capped at seven, and mandatory equipment counts 7/14. Full adds mapped weather summaries/histories, monthly vegetation and interactions. Local variants replace cross-area rain/gust x vegetation products with means/maxima of products formed within the same barangay before recloser aggregation. Dry-to-wet fractions use 3/7 preceding dry days within each mapped barangay. Weather histories never exceed seven days. Prior wet-day counts retain the original recloser mapped-mean-rain definition (>=0.1 mm); the city any-site definition is not silently substituted.

The legacy 88-feature family retains the earlier feature catalogue, including fitting-only per-unit P90 stress, fault/repair subtypes, shared-other-recloser histories, and original remote-sensing coverage proxy. Its new models use the new common unweighted algorithm/calibration policy. Original fitted legacy models and their F1 rules are separately preserved as references; these are not the same procedure as refitting the 88 inputs under modern suite settings.

## Spatial and label limits

The nine names are Buncag, Chariot, Factor, Kalikasan, Lacsamana, Mendoza, PENRO, PILTEL and San Manuel. The existing pre-development mapping has 33 recloser/barangay links across 24 distinct barangays. Weights remain equal within each recloser proxy. Shared weather grid values are retained; no distinct weather, asset ages, electrical connectivity or service-area polygons are invented. Same-month vegetation is a geographic proxy, not measured branch contact or verified equipment exposure.

The label is a recorded interruption affecting part of a named service area, not necessarily failure of the physical recloser. Existing assignment exclusions and publication-date fallbacks remain unchanged. Primary fitting/scoring excludes otherwise-zero rows on dates with notices lacking an extracted recloser, while preserving positives. That mask uses target-day reporting evidence and is retrospective/conditional, not an operational prediction-time filter. Sensitivity includes all 2,376 unit-days and the same 57 positives.

## Chronological workflow

Frozen models reuse four original 2025 development origins and five-role cohorts. Final refit/calibration use only pre-2026 data. The model, preprocessor, calibrator and thresholds stay fixed throughout 2026. Frozen selection minimizes mean earlier outer-quarter log loss, then higher AP and ID. A fitted sigmoid slope zero yields a constant procedure and must not be presented as useful detection.

Monthly origins run October 2024–September 2026, with three initial calibration-pool months. At M use records published by M-1 20:00 PHT and incident maturity through M-15. Tuning uses the preceding three months; initial fit ends 14 days before tuning starts. Refit uses the full mature 12/24/all-history window. Calibration uses genuinely earlier monthly predictions joined on recloser-day row ID; no current-model in-sample scores. January–December 17, 2025 are scored for selection. December 18–31 raw predictions remain available for later calibration without entering selection outcomes. Later 2026 updates may use completed earlier 2026 outcomes after publication/maturity checks, never current/future target-month labels.

Highest pooled 2025 recall with pooled FPR<=20%; higher mean monthly AP, lower loss, ID ties. If none feasible lowest FPR then AP/loss/ID; flag allowance unmet.

XGBoost 12m raw feature replacement: FPR<=20%, AP better than full in >=7/12 months, no worse mean monthly loss; highest recall, AP, lower loss, ID ties. Else keep full reference.

The 7-of-12 feature screen is a disclosed experimental consistency criterion, not an industry standard. All feature candidate scores are retained even when not selected. Monthly intercept adjustment preserves within-month ranking and, with corresponding retuned thresholds, alert decisions; it is not independent evidence of improved detection.

## All raw frozen procedures at the primary FPR20 rule

| Procedure | Detected /57 | False warnings | Missed | Precision | Recall | FPR | AP | AUC | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic__history__all__raw | 38 | 1089 | 19 | 3.4% | 66.7% | 53.4% | 0.0339 | 0.5758 | 0.1280 |
| logistic__all_weather__all__raw | 34 | 1097 | 23 | 3.0% | 59.6% | 53.7% | 0.0320 | 0.5616 | 0.1282 |
| logistic__full__all__raw | 32 | 1047 | 25 | 3.0% | 56.1% | 51.3% | 0.0321 | 0.5634 | 0.1286 |
| logistic__legacy_full__all__raw | 30 | 886 | 27 | 3.3% | 52.6% | 43.4% | 0.0350 | 0.5828 | 0.1295 |
| spline_logistic__history__all__raw | 38 | 1089 | 19 | 3.4% | 66.7% | 53.4% | 0.0339 | 0.5758 | 0.1280 |
| spline_logistic__all_weather__all__raw | 30 | 1056 | 27 | 2.8% | 52.6% | 51.7% | 0.0330 | 0.5609 | 0.1286 |
| spline_logistic__full__all__raw | 31 | 1056 | 26 | 2.9% | 54.4% | 51.7% | 0.0334 | 0.5606 | 0.1290 |
| spline_logistic__legacy_full__all__raw | 29 | 916 | 28 | 3.1% | 50.9% | 44.9% | 0.0369 | 0.5825 | 0.1300 |
| xgboost__history__all__raw | 44 | 1140 | 13 | 3.7% | 77.2% | 55.9% | 0.0420 | 0.6305 | 0.1270 |
| xgboost__all_weather__all__raw | 40 | 1191 | 17 | 3.2% | 70.2% | 58.4% | 0.0343 | 0.5721 | 0.1277 |
| xgboost__full__all__raw | 32 | 1010 | 25 | 3.1% | 56.1% | 49.5% | 0.0297 | 0.5414 | 0.1325 |
| xgboost__legacy_full__all__raw | 25 | 736 | 32 | 3.3% | 43.9% | 36.1% | 0.0289 | 0.5376 | 0.1334 |
| catboost__history__all__raw | 45 | 1360 | 12 | 3.2% | 78.9% | 66.6% | 0.0326 | 0.5574 | 0.1319 |
| catboost__all_weather__all__raw | 36 | 959 | 21 | 3.6% | 63.2% | 47.0% | 0.0356 | 0.5776 | 0.1268 |
| catboost__full__all__raw | 34 | 987 | 23 | 3.3% | 59.6% | 48.4% | 0.0338 | 0.5680 | 0.1283 |
| catboost__legacy_full__all__raw | 29 | 835 | 28 | 3.4% | 50.9% | 40.9% | 0.0419 | 0.5773 | 0.1276 |
| random_forest__history__all__raw | 28 | 674 | 29 | 4.0% | 49.1% | 33.0% | 0.0353 | 0.5973 | 0.1275 |
| random_forest__all_weather__all__raw | 29 | 903 | 28 | 3.1% | 50.9% | 44.2% | 0.0298 | 0.5415 | 0.1296 |
| random_forest__full__all__raw | 30 | 894 | 27 | 3.2% | 52.6% | 43.8% | 0.0322 | 0.5405 | 0.1296 |
| random_forest__legacy_full__all__raw | 30 | 914 | 27 | 3.2% | 52.6% | 44.8% | 0.0334 | 0.5622 | 0.1304 |
| xgboost__rainfall__all__raw | 37 | 966 | 20 | 3.7% | 64.9% | 47.3% | 0.0392 | 0.5972 | 0.1272 |
| xgboost__wind__all__raw | 40 | 1187 | 17 | 3.3% | 70.2% | 58.2% | 0.0391 | 0.6109 | 0.1271 |
| xgboost__temperature_humidity__all__raw | 39 | 1058 | 18 | 3.6% | 68.4% | 51.8% | 0.0362 | 0.6091 | 0.1276 |
| xgboost__vegetation__all__raw | 35 | 1063 | 22 | 3.2% | 61.4% | 52.1% | 0.0356 | 0.5593 | 0.1345 |
| xgboost__full_local__all__raw | 28 | 884 | 29 | 3.1% | 49.1% | 43.3% | 0.0297 | 0.5407 | 0.1342 |
| xgboost__full_dry__all__raw | 31 | 1011 | 26 | 3.0% | 54.4% | 49.5% | 0.0310 | 0.5518 | 0.1330 |
| xgboost__full_local_dry__all__raw | 38 | 1030 | 19 | 3.6% | 66.7% | 50.5% | 0.0360 | 0.5728 | 0.1276 |

## All raw monthly procedures at the primary FPR20 rule

| Procedure | Detected /57 | False warnings | Missed | Precision | Recall | FPR | AP | AUC | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic__full__12m__raw | 10 | 357 | 47 | 2.7% | 17.5% | 17.5% | 0.0418 | 0.5555 | 0.1312 |
| logistic__full__24m__raw | 12 | 566 | 45 | 2.1% | 21.1% | 27.7% | 0.0326 | 0.5580 | 0.1328 |
| logistic__full__all__raw | 10 | 438 | 47 | 2.2% | 17.5% | 21.5% | 0.0352 | 0.5945 | 0.1283 |
| logistic__history__12m__raw | 7 | 394 | 50 | 1.7% | 12.3% | 19.3% | 0.0330 | 0.5869 | 0.1279 |
| logistic__history__24m__raw | 12 | 448 | 45 | 2.6% | 21.1% | 22.0% | 0.0397 | 0.6084 | 0.1265 |
| logistic__history__all__raw | 13 | 538 | 44 | 2.4% | 22.8% | 26.4% | 0.0371 | 0.6030 | 0.1276 |
| xgboost__all_weather__12m__raw | 7 | 319 | 50 | 2.1% | 12.3% | 15.6% | 0.0290 | 0.5118 | 0.1258 |
| xgboost__full__12m__raw | 9 | 372 | 48 | 2.4% | 15.8% | 18.2% | 0.0279 | 0.5147 | 0.1259 |
| xgboost__full__24m__raw | 4 | 380 | 53 | 1.0% | 7.0% | 18.6% | 0.0290 | 0.4708 | 0.1325 |
| xgboost__full__all__raw | 8 | 347 | 49 | 2.3% | 14.0% | 17.0% | 0.0241 | 0.4515 | 0.1311 |
| xgboost__full_dry__12m__raw | 5 | 327 | 52 | 1.5% | 8.8% | 16.0% | 0.0287 | 0.5264 | 0.1255 |
| xgboost__full_local__12m__raw | 8 | 327 | 49 | 2.4% | 14.0% | 16.0% | 0.0293 | 0.5131 | 0.1259 |
| xgboost__full_local_dry__12m__raw | 12 | 406 | 45 | 2.9% | 21.1% | 19.9% | 0.0296 | 0.5390 | 0.1254 |
| xgboost__history__12m__raw | 13 | 623 | 44 | 2.0% | 22.8% | 30.5% | 0.0362 | 0.5715 | 0.1251 |
| xgboost__history__24m__raw | 17 | 649 | 40 | 2.6% | 29.8% | 31.8% | 0.0273 | 0.5011 | 0.1294 |
| xgboost__history__all__raw | 2 | 175 | 55 | 1.1% | 3.5% | 8.6% | 0.0334 | 0.5157 | 0.1279 |
| xgboost__legacy_full__12m__raw | 8 | 286 | 49 | 2.7% | 14.0% | 14.0% | 0.0283 | 0.5249 | 0.1255 |
| xgboost__legacy_full__24m__raw | 4 | 338 | 53 | 1.2% | 7.0% | 16.6% | 0.0271 | 0.4878 | 0.1317 |
| xgboost__legacy_full__all__raw | 7 | 433 | 50 | 1.6% | 12.3% | 21.2% | 0.0260 | 0.4749 | 0.1295 |
| xgboost__rainfall__12m__raw | 8 | 379 | 49 | 2.1% | 14.0% | 18.6% | 0.0326 | 0.5429 | 0.1250 |
| xgboost__temperature_humidity__12m__raw | 10 | 607 | 47 | 1.6% | 17.5% | 29.7% | 0.0307 | 0.5189 | 0.1265 |
| xgboost__vegetation__12m__raw | 20 | 734 | 37 | 2.7% | 35.1% | 36.0% | 0.0365 | 0.5808 | 0.1240 |
| xgboost__wind__12m__raw | 16 | 590 | 41 | 2.6% | 28.1% | 28.9% | 0.0265 | 0.4819 | 0.1261 |

## Cutoffs, uncertainty and reproducibility

Frozen thresholds are saved once per procedure; monthly thresholds are saved per origin. The 0.50 reference is explicit and is not substituted for the earlier-selected policies. F1 now uses the separate earlier calibration period; the original legacy reference used its earlier tuning F1 threshold and is labelled accordingly. Earlier FPR allowances are not guarantees of future FPR. Precision is stored as zero by the inherited metric convention if no warnings exist; empirical warning precision is undefined in that case. No-warning accuracy is 97.3% on the primary cohort while recall is zero.

Date-block bootstrap resamples seven calendar days with all corresponding available recloser rows together (300 replicates). It does not treat nine units on one date as independent weather observations. Intervals are exploratory and conditional on these fitted/selected procedures, not full selection uncertainty. All model trials, preprocessing, selected rounds, role memberships, probability modes, thresholds, per-unit metrics, support audits, primary/sensitivity predictions and original reference results are saved.

All 2026 results remain exploratory; these dates were repeatedly examined and motivated the suite. Historical target-day weather and own-month vegetation do not establish operational day-ahead input availability. Reporting completeness is unverified. Different city/barangay/recloser units and eligibility must not be compared as though they have the same target prevalence or independently observed events.

References: [thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
