# Puerto Princesa pooled recloser-day experiment

This separate thesis experiment evaluates the nine recloser names appearing in each year 2022–2026 in the final 1,369-post city-only source: Buncag, Chariot, Factor, Kalikasan, Lacsamana, Mendoza, PENRO, PILTEL and San Manuel. It does not overwrite the original barangay experiments or the older root-level recloser panel.

The model predicts a **recorded unscheduled interruption affecting part of a named recloser service area on day D**, given historical weather for D, monthly vegetation and previously published history. It does not predict the physical recloser hardware necessarily failing. It is a retrospective experiment; already-seen 2026 is exploratory.

Barangay weather and vegetation become spatial exposure proxies through a frozen announcement-derived mapping learned before development tests. Equal barangay averages describe typical exposure; spatial maxima/ranges describe more exposed parts. No official service-area polygons, network lengths or fabricated recloser coordinates are used. Read [methodology](docs/METHODOLOGY.md) and [feature catalog](FEATURE_CATALOG.md).

Run with the existing V3 environment:

```powershell
& '.\thesis_pipeline_v3\.venv\Scripts\python.exe' -X utf8 '.\thesis_pipeline_recloser\run_pipeline.py' --run-id my_recloser_run
```

Eight meaningful safeguards run before eight separate scripts. Each stage has its own logs, status, inputs/outputs and documented purpose in [stage guides](docs/STAGES.md). Add `--until 3` to inspect labels/features/cohorts first, then use `--resume` to continue. Completed runs are not overwritten. A code/data change requires a new run ID.

1. Audit original sources and previous artifacts; construct labels and assignment review.
2. Freeze area proxies and construct 88 candidate inputs.
3. Freeze chronological development and evaluation cohorts.
4. Tune/refit/calibrate learners on four 2025 development periods.
5. Select the procedure using development periods only and report paired feature effects.
6. Fit final learned models and a frequency benchmark before 2026.
7. Evaluate already-seen 2026, date-block uncertainty and negative-label sensitivity.
8. Verify reproducibility/preservation and write `08_report/THESIS_RECLOSER_REPORT.md`.

Primary fitting excludes otherwise-zero rows on dates with posts that name no extracted recloser, preserving positives. This conservative uncertainty policy changes the population; results cannot be directly compared with barangay AP as evidence that one geography is superior. A frozen-model sensitivity includes all no-record zeros. Missing/incorrect public reports remain possible.

The completed first comparison is `runs/recloser9_validated_first/`. Read its [measured report](runs/recloser9_validated_first/08_report/THESIS_RECLOSER_REPORT.md) and [results interpretation](analysis/recloser9_validated_first/RESULTS_INTERPRETATION.md). Four final calibration fits have negative slopes; the interpretation distinguishes raw ranking from calibrated ranking and records why these remain weak warning models. `runs/recloser9_first/` is the preserved preparation attempt that stopped at cohort checks before training.

Reproduce the completed-output review without retraining:

```powershell
& '.\thesis_pipeline_v3\.venv\Scripts\python.exe' -X utf8 '.\thesis_pipeline_recloser\analysis\review_completed_run.py' '.\thesis_pipeline_recloser\runs\recloser9_validated_first'
```
