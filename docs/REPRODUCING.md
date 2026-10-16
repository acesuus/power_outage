# Reproduction and documented experiment steps

1. **Inspect sources.** Read the data guide and original run's
   `01_audit/input_hashes.json`, source reports and assignment evidence. The
   package includes every file named by that input-hash manifest with original
   directory paths. `docs/source_manifest.json` maps copied files to their
   workspace-relative source path, SHA-256 and size.
2. **Inspect labels and features.** The reference run contains labels, assignment
   evidence, spatial proxy mappings and original features. The suite's
   `02_features/recloser_feature_panel.csv` contains the final engineered panel;
   `families.json` restricts history candidates to nine raw inputs. Saved NPZ
   cohort archives identify the exact fit/tune/refit/calibration/outer rows.
3. **Review chronological development.** Four frozen development origins are
   January, April, July and October 2025. Each XGBoost candidate tunes depth
   (2/3) and L2 regularization (10/30) on earlier records, with early stopping.
   Frozen trial selection minimizes tuning log loss, using AP to resolve ties.
   Records are in `03_frozen_development/*/xgboost__history__all/`.
4. **Review final frozen training.** The final refit uses eligible expanding
   history from September 30, 2022 through September 16, 2025. Calibration uses
   October 1 through December 17, 2025. The model and warnings are fixed before
   the 2026 evaluation. It does not refit monthly. Consult cohort summary for
   the exact row counts, exclusions and all role dates.
5. **Review monthly training.** Three independent windows: 12 months, 24 months,
   expanding. Development/warmup origins run October 2024 through December 2025;
   evaluation origins run January through September 2026. Each month's fitting
   uses earlier eligible published records and a 14-day incident-maturity buffer.
   Tuning chooses AP, then log loss. Earlier out-of-sample predictions populate
   a calibration pool; the cutoff updates each month. Later monthly fits may
   use earlier 2026 outcomes once eligible. Current target-month labels are
   excluded from fitting and calibration. Calibration files and raw pools are
   supplied, including the initial three warmup months.
6. **Review warning policies.** Raw and calibrated scores are retained. FPR20,
   FPR10, balanced accuracy and F1 rules are learned earlier; fixed 0.50 is a
   separate rule. FPR20 seeks recall subject to the earlier calibration FPR
   constraint; it is not a future FPR guarantee. An infinite cutoff is an
   explicit no-warning rule. Some original JSON artifacts use Python's Infinity
   serialization; Python's JSON reader handles it. Monthly thresholds are in
   the focused analysis and individual model folders.
7. **Recount and reload.** Run `python verify_package.py`. It verifies all copied
   files, all 77 fitted model probabilities, saved warning decisions, and the
   four FPR20 operating points plus the monthly 12-month balanced-accuracy point.
   No retraining occurs during verification. The packaging verification report
   is included in `docs/package_verification.json`.
8. **Optional refitting.** `train_history.py --approach frozen --output refits/new_frozen`
   recreates four development fits and the final frozen fit using archived
   original functions and cohorts. `--approach monthly --output refits/new_monthly`
   recreates all three monthly windows, including warmup and earlier calibration
   pools. It never overwrites an existing directory. Use recorded dependency
   versions and a new run directory. All 77 archived fits were checked by model
   reload. The five frozen fits were additionally retrained with the focused
   entrypoint: selected trials matched, and final raw probabilities matched
   exactly. See `docs/frozen_refit_check.json`. The 72 monthly fits were not
   retrained during packaging.

The original numbered broad-suite scripts are archived for methodological
traceability. Their full-run audit expects thousands of artifacts from other
earlier experiments that are not part of this focused export. Running the
original `run_pipeline.py` end to end is therefore not the supported entrypoint
here; use `verify_package.py` and `train_history.py`. Broad-suite reports and
manifests are historical evidence, not a declaration that all other candidate
binaries were exported.

The focused refit entrypoint reuses saved engineered features and cohorts. It
does not redownload weather or regenerate textual interruption labels. Feature
generation logic and weather-download code are included for inspection; changing
their inputs requires a new separately documented experiment.

## Requested evaluation metrics and calibration

`python report_candidate_metrics.py` independently recomputes AUC-ROC, F1-score,
precision, recall and Brier score for all four history candidates, both saved
probability modes and all five warning policies. It checks all 40 metric rows
against the archived scores. It also produces reliability diagrams, quantile-bin
calibration data, ECE and a Brier comparison against earlier calibration-rate
probabilities. The report is `reports/candidate_metrics/RESULTS.md`. It does not
fit a new calibrator or change an existing warning rule using evaluation labels.
