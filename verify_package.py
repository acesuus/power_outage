"""Verify source bytes, reload all 77 candidate fits and recount requested results."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'thesis_pipeline_recloser_suite'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sums = ROOT / 'SHA256SUMS.json'
    if sums.exists():
        for name, expected in json.loads(sums.read_text(encoding='utf-8')).items():
            with (ROOT / name).open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, name
    import joblib
    import numpy as np
    from suitelib.common import csv_read, read_json
    from suitelib.data import load, frozen_roles
    from suitelib.modeling import raw_predict, score
    from suitelib.stages import monthly_data
    from sklearn.metrics import confusion_matrix

    manifest = read_json(ROOT / 'docs/source_manifest.json')
    for name, item in manifest.items():
        with (ROOT / name).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256'], name
    run = ROOT / 'thesis_pipeline_recloser_suite/runs/recloser_suite_first'
    config = read_json(run / 'config.json')
    frame = load(run)
    final = csv_read(run / '07_frozen_evaluation/predictions.csv')
    monthly = csv_read(run / '06_monthly_evaluation/predictions.csv')
    assert len(final) == len(monthly) == 2098
    assert int(final.outage_label.sum()) == int(monthly.outage_label.sum()) == 57
    assert final[final.outage_label.eq(1)].date.nunique() == 41
    assert final.row_id.equals(monthly.row_id)
    assert int(frame.outage_label.sum()) == 418 and len(frame) == 13473
    assert len(csv_read(ROOT / 'paleco_unscheduled_puerto_princesa_city_only.csv')) == 1369
    from lib.features import history_features
    reference = ROOT / 'thesis_pipeline_recloser/runs/recloser9_validated_first'
    evidence = csv_read(reference / '01_audit/recloser_outage_evidence.csv')
    flags = csv_read(ROOT / 'equipment_analysis/all_row_equipment_flags.csv')
    rebuilt = history_features(frame, evidence, flags)
    history_fields = ['outage_days_prev3d','outage_days_prev7d','days_since_last_outage_capped7',
                      'equipment_fault_days_prev7d','equipment_fault_days_prev14d']
    compared = frame[['psgc','date']+history_fields].merge(rebuilt[['psgc','date']+history_fields],
                           on=['psgc','date'], validate='one_to_one', suffixes=('_saved','_rebuilt'))
    for col in history_fields:
        np.testing.assert_array_equal(compared[col+'_saved'], compared[col+'_rebuilt'])
    models = sorted(run.rglob('model.joblib'))
    assert len(models) == 77
    worst_error = 0.0
    for path in models:
        bundle = joblib.load(path)
        assert bundle['spec']['kind'] == 'xgboost' and bundle['spec']['family'] == 'history'
        folder = path.parent
        if bundle['protocol'] == 'monthly':
            phase = 'development' if '05_monthly_development' in path.parts else 'evaluation'
            origin = folder.parent.name
            roles, target, _ = monthly_data(run, frame, origin, phase, bundle['spec']['window'])
            expected = csv_read(folder / 'raw_predictions.csv').raw_probability.to_numpy()
        else:
            origin = folder.parent.name if '03_frozen_development' in path.parts else config['test_start']
            roles = frozen_roles(frame, origin, config)
            target = roles['outer']
            saved = csv_read(folder / 'predictions.csv') if '03_frozen_development' in path.parts else final
            expected = saved[bundle['spec']['base_id'] + '__raw_probability'].to_numpy()
        got = raw_predict(bundle, target)
        error = float(np.max(np.abs(got - expected)))
        assert error < 1e-12, str(path)
        worst_error = max(worst_error, error)
        # Reload and reproduce all warning decisions and both score modes,
        # including frozen development and the monthly development models.
        if 'calibrator' in bundle:
            saved_path = folder / 'predictions.csv'
            saved = csv_read(saved_path) if saved_path.exists() else final
            scored_target = frame.set_index('row_id', drop=False).loc[saved.row_id].copy()
            reproduced, _ = score(bundle, scored_target, origin, config)
            for col in reproduced:
                if col.endswith('_warning'):
                    np.testing.assert_array_equal(reproduced[col], saved[col])
                elif col.endswith('_probability'):
                    np.testing.assert_allclose(reproduced[col], saved[col], rtol=0, atol=1e-12)

    comparison = []
    for name, pred, prefix, policy, expected_tp, expected_fp in [
        ('Frozen expanding', final, 'xgboost__history__all__raw', 'fpr20', 44, 1140),
        ('Monthly 12 months', monthly, 'xgboost__history__12m__raw', 'fpr20', 13, 623),
        ('Monthly 24 months', monthly, 'xgboost__history__24m__raw', 'fpr20', 17, 649),
        ('Monthly expanding', monthly, 'xgboost__history__all__raw', 'fpr20', 2, 175),
        ('Monthly 12 months', monthly, 'xgboost__history__12m__raw', 'balanced_accuracy', 42, 1255),
    ]:
        tn, fp, fn, tp = map(int, confusion_matrix(pred.outage_label, pred[prefix+'_'+policy+'_warning'], labels=[0,1]).ravel())
        assert (tp, fp) == (expected_tp, expected_fp)
        comparison.append(dict(approach=name, policy=policy, rows=len(pred), positives=tp+fn,
                               tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp)))
    result = dict(status='passed', source_files_verified=len(manifest), model_bundles_reloaded=len(models),
                  publication_aware_history_fields_rebuilt=history_fields,
                  maximum_probability_reload_error=worst_error, comparison=comparison)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
