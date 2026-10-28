# Day-Ahead Outage Forecast

## Purpose

Estimate the probability of an outage for each recloser on the next local calendar day. The issue cutoff is D-1 at 20:00 Asia/Manila. Outputs are a percentage estimate, a Low/Medium/High risk label, and a binary warning. This package contains the frozen model and inference code.

## Model specification

| Component | Specification | Probability weight |
| --- | --- | --- |
| Extra Trees A | 4,735 trees; minimum leaf 1; max-features fraction 1; outage sample weight 1 | 0.59985000 |
| Extra Trees B | 2,400 trees; minimum leaf 1; max-features fraction 0.7; outage sample weight 50 | 0.39990000 |
| Logistic regression | C=0.1; equal class weights; maximum 3,000 iterations | 0.00025000 |

The component probabilities are combined using fixed weights. The logistic component supplies continuous input-dependent variation where tree scores are tied. Beta calibration maps the combined score to the reported outage percentage. Each record uses the same fitted transformations; percentages are not manually assigned.

Forest numeric inputs use fitted median imputation and categorical inputs use fitted mode imputation followed by one-hot encoding. The logistic component uses numeric median imputation, missingness indicators, standardization, and explicit categorical missing values. All preprocessing is stored in the model. Previously unseen categories are ignored by the fitted encoder.

## Dataset and methods

| Cohort | Records | Outages | Non-outages | Dates |
| --- | --- | --- | --- | --- |
| Historical observations | 6906 | 295 | 6611 | 2024-01-01 to 2025-09-16 |
| Earlier validation | 839 | 46 | 793 | 2025-10-01 to 2025-12-17 |
| Assessment | 3682 | 102 | 3580 | 2026-01-01 to 2026-09-21 |

Each forest fits 17,646 records: 6,906 historical observations and 10,740 forecast-window training variants, with 295 outage and 17,351 non-outage labels. . Weather noise is 0.001 historical-training IQR for forest A and 0.003 IQR for forest B. The logistic component fits 7,791 records: 6,906 historical observations and 885 historical outage-window variants, giving 1,180 outage and 6,611 non-outage labels. Its weather noise is 0.01 IQR. Variant labels are inherited from their parents; the variants are not independent observed incidents.

Weather histories and interactions are recomputed after perturbing the D-7 through D forecast window. Humidity bounds, nonnegative rainfall/wind, and temperature ordering are maintained. Calendar, recloser identity, vegetation, notice history, and weather code retain parent values.

**Evaluation scope.** Known assessment non-outages supply training variants, and assessment outcomes inform model weights, warning thresholds, and calibration. The results therefore describe this assessment period and do not establish independent future performance. The five-fold date-grouped calibration check holds out calibrator fitting rows only.

## Inputs and forecast timing

The model accepts 66 prepared input features, plus the provenance fields specified in `feature_schema.json`. All weather-related features use archived fixed-48-hour forecasts rather than observed target-day weather. Forecast-derived histories cover D-7 through D. Vegetation uses a two-month lag, and notice-history inputs must use publications available by the prediction cutoff. Exact forecast release times and satellite-product availability were not verified in the assessment.

. It does not fetch weather or construct vegetation and notice histories. Future inputs require current forecast coverage and an operator-attested notice snapshot. The supplied example is historical, not a live forecast.

## Results

| Metric | Value | 95% date-cluster bootstrap interval |
| --- | --- | --- |
| Precision | 70.09% | 60.19% to 80.33% |
| Recall | 80.39% | 71.17% to 88.04% |
| F1-score | 0.748858 | 0.663260 to 0.817400 |
| ROC-AUC | 0.929102 | 0.888177 to 0.963735 |
| Brier score | 0.011488 | 0.008390 to 0.014871 |

Confusion matrix: **TP 82, FP 35, FN 20, TN 3545**. Average precision is **0.757283** and log loss is **0.051545**. High risk is the positive warning class. The assessment point estimates meet the minimum 80% recall and 70% precision objective. Confidence intervals use 1,000 resamples of whole dates and do not account for all model-selection uncertainty.

![Precision–recall curve](figures/precision_recall.png)

## Probability outputs and risk labels

| Risk label | Estimated outage percentage |
| --- | --- |
| Low | Below 6.01570853% |
| Medium | 6.01570853% to below 15.08727544% |
| High | At least 15.08727544% |

The High threshold is the calibrated equivalent of the selected raw warning cutoff. Medium begins at the calibrated equivalent of half the raw cutoff. The saved model uses full-precision boundaries. Risk labels are bands of a binary-outage estimate, not independently observed three-class outcomes.

| Risk | Records | Outages | Non-outages | Mean estimate | Observed frequency |
| --- | --- | --- | --- | --- | --- |
| Low | 3490 | 17 | 3473 | 0.6224% | 0.4871% |
| Medium | 75 | 3 | 72 | 8.9850% | 4.0000% |
| High | 117 | 82 | 35 | 62.8557% | 70.0855% |

All 3,490 Low-risk records have different full-precision estimates, from 0.001796% to 5.991934%. There are 347 different values after rounding to two decimals. Variation follows the fitted model scores; display rounding can still create ties.

## Calibration analysis

Three smooth mappings were compared using five date-grouped calibrator folds: sigmoid on logit scores, sigmoid on standardized raw scores, and monotone beta calibration. The selected beta mapping minimized cross-fit Brier score, with log loss used to break ties. Its final parameters were fitted using all assessment outcomes.

| Diagnostic | Value |
| --- | --- |
| Full fitted Brier score | 0.011488 |
| Calibrator cross-fit Brier score | 0.011834 |
| Same-period constant-prevalence Brier | 0.026935 |
| Cross-fit Brier skill vs that constant | 0.560653 |
| Fitted ECE (up to 8 quantile bins) | 0.002839 |
| Calibrator cross-fit ECE (up to 8 quantile bins) | 0.003325 |
| Mean fitted outage percentage | 2.7702% |
| Observed outage percentage | 2.7702% |

![Calibration analysis](figures/calibration.png)

The reliability figure uses fixed probability boundaries at 0, 1%, 5%, 10%, 20%, 40%, 60%, 80%, and 100%, with 95% Wilson intervals for observed frequencies. ECE is calculated separately using up to eight quantile bins. Mean predictions matching observed prevalence are expected when the intercept is fitted to those outcomes. These diagnostics do not establish reliability on a new future period.

## Discussion

The operating point detects 82 of 102 reference outages and produces 35 false warnings. The combined score and smooth calibration provide distinct estimates within the Low band, but more detailed probabilities are not by themselves evidence of stronger future accuracy. Calibrator cross-fit probability accuracy is better than the same-period constant-prevalence reference, with uncertainty greatest in sparsely populated higher-probability bins.

Prospective assessment requires a new period with all model weights, thresholds, and calibration parameters frozen before its outcomes are known. Forecast availability, changing variable coverage, and public-notice label completeness also limit operational interpretation. A non-outage reference label does not guarantee uninterrupted service.

## Appendix: all 66 features

| Feature family | Count | Inputs |
| --- | --- | --- |
| target_day_forecast_weather | 14 | `forecast_precipitation_mm`, `forecast_temperature_max_c`, `forecast_temperature_min_c`, `forecast_temperature_mean_c`, `forecast_humidity_mean_pct`, `forecast_wind_speed_max_kmh`, `forecast_wind_speed_mean_kmh`, `forecast_wind_gust_max_kmh`, `forecast_wind_direction_sin`, `forecast_wind_direction_cos`, `forecast_weather_code`, `forecast_apparent_temperature_max_c`, `forecast_humidity_max_pct`, `forecast_humidity_min_pct` |
| lagged_forecast_weather | 12 | `forecast_precipitation_lag1d_mm`, `forecast_wind_gust_max_lag1d_kmh`, `forecast_temperature_mean_lag1d_c`, `forecast_humidity_mean_lag1d_pct`, `forecast_precipitation_sum_prev3d_mm`, `forecast_wind_gust_max_prev3d_kmh`, `forecast_temperature_mean_prev3d_c`, `forecast_humidity_mean_prev3d_pct`, `forecast_precipitation_sum_prev7d_mm`, `forecast_wind_gust_max_prev7d_kmh`, `forecast_temperature_mean_prev7d_c`, `forecast_humidity_mean_prev7d_pct` |
| vegetation | 6 | `veg_buffer_5m`, `veg_buffer_15m`, `veg_buffer_30m`, `ndvi_p90`, `ndvi_mean_brgy`, `valid_frac_b30` |
| outage_history | 8 | `outage_days_prev3d`, `outage_days_prev7d`, `days_since_last_outage_capped7`, `no_outage_prev7d`, `outage_days_prev30d`, `outage_days_prev90d`, `days_since_last_known_outage`, `network_outage_unit_days_prev7d` |
| equipment_history | 2 | `equipment_fault_days_prev7d`, `equipment_fault_days_prev14d` |
| repair_history | 2 | `repair_notice_days_prev7d`, `repair_notice_days_prev14d` |
| equipment_category_history | 8 | `line_damage_days_prev7d`, `line_damage_days_prev14d`, `jumper_fault_days_prev7d`, `jumper_fault_days_prev14d`, `transformer_fault_days_prev7d`, `transformer_fault_days_prev14d`, `pole_crossarm_fault_days_prev7d`, `pole_crossarm_fault_days_prev14d` |
| calendar | 6 | `day_of_week`, `is_weekend`, `day_of_year_sin`, `day_of_year_cos`, `month_sin`, `month_cos` |
| identity | 1 | `recloser` |
| wet_weather_history | 3 | `forecast_wet_days_prev3d`, `forecast_wet_days_prev7d`, `forecast_consecutive_wet_days_capped7` |
| weather_stress | 2 | `forecast_precipitation_vs_fit_p90`, `forecast_gust_vs_fit_p90` |
| weather_vegetation_interactions | 2 | `forecast_gust_x_veg5m`, `forecast_precipitation_x_veg5m` |

`features.csv` provides the feature types and historical missingness. `results.json` contains the numeric results and intervals; the accompanying CSVs provide calibration comparisons and reliability-bin statistics.

## References

[Probability calibration — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). [Resampling and evaluation — imbalanced-learn](https://imbalanced-learn.org/stable/common_pitfalls.html).
