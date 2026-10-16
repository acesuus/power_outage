# Recent-outage-trend candidate

`model.joblib` is the final frozen XGBoost model featured in the repository README.
Use the root `verify_recent_outage_trend.py` to load and verify it with the bundled
source modules and the recorded Python dependencies. The bundle includes its
preprocessor, calibration mapping and warning rules.

- `features.json`: ten raw inputs and the added trend definition.
- `selection.json`, `trials.csv`: original earlier-data tuning and selected rounds.
- `calibration_and_warnings.json`: earlier-fitted calibration and all warning rules.
- `predictions.csv`: all 2,098 evaluation rows, raw/calibrated probabilities and warnings.
- `development/`: four earlier chronological fits and their predictions/records.
- `provenance.json`: exact source paths and SHA-256 values for copied artifacts.
- `source_experiment_verification.json`: review of the larger local feature experiment.

The source review describes the full 120-model experiment. This focused upload
contains only the five trend fits. It does not claim that all other feature
variants were uploaded. The user chose this featured candidate after inspecting
exploratory 2026 results; this is not an untouched-test selection claim.
