# Day-Ahead Outage Forecast

Standalone, frozen inference package. Copy this entire folder together; no development folders or dashboard are required.

## Clone with Git LFS

The fitted model is stored using Git LFS. Install Git LFS before cloning, then run:

```powershell
git lfs install
git clone https://github.com/acesuus/power_outage.git
cd power_outage
git lfs pull
```

## Run

Python 3.13 was used for verification.

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.csv --output predictions.csv
python verify.py
```

Use a new output filename. Supply the 66 prepared features and provenance columns in `feature_schema.json`; do not supply outcome labels. The saved pipelines perform their own imputation and encoding. Missing core forecast variables and future-dated source information are rejected.

Future rows require an operator-attested notice snapshot collected within 24 hours before the issue cutoff, current forecast inputs, and supported recloser names from `model_manifest.json`. This package scores prepared features and does not retrieve or construct raw-source inputs. `example_input.csv` is historical and covers the three risk labels.

## Files

- `model.joblib`: fitted classifiers, preprocessing, calibration, and risk cutoffs.
- `forecast_model.py`: standalone scoring and input checks.
- `predict.py`: CSV prediction command.
- `feature_schema.json`: 66 model inputs and provenance fields.
- `model_manifest.json`: model identity, supported reclosers, software, and integrity hash.
- `requirements.txt`: inference dependencies.
- `example_input.csv`, `example_output.csv`: executable example and expected output.
- `verify.py`: local example replay and output checks.
- `documentation/THESIS_REPORT.md`: thesis methods, results, discussion, feature list, and evaluation scope.

Assessment precision is 70.09%, recall 80.39%, ROC-AUC 0.9291, and fitted Brier score 0.011488. Assessment records informed training variants, selection, and calibration, so these values are not independently validated future performance. See the thesis methods note for the evaluation protocol.
