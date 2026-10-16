"""Refit the focused history candidates in a NEW output directory.

Uses archived original fitting functions and the exact saved cohort definitions.
Does not overwrite the archived experiment or reselect a model using 2026 scores.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'thesis_pipeline_recloser_suite'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', required=True, choices=['frozen', 'monthly'])
    parser.add_argument('--output', required=True, type=Path, help='A new output directory')
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'Choose a NEW output directory; archived runs must remain unchanged.'
    import joblib
    import pandas as pd
    from suitelib.common import csv_write, json_write, read_json
    from suitelib.data import load, frozen_roles
    from suitelib.modeling import fit_model, attach, score, raw_predict
    from suitelib import stages
    source = ROOT / 'thesis_pipeline_recloser_suite/runs/recloser_suite_first'
    config = read_json(source / 'config.json')
    output.mkdir(parents=True)
    shutil.copy2(source / 'config.json', output / 'config.json')
    shutil.copytree(source / '02_features', output / '02_features')
    frame = load(output)
    if args.approach == 'frozen':
        # Four chronological development fits, then the single final frozen fit.
        spec = dict(kind='xgboost', family='history', window='all', base_id='xgboost__history__all')
        all_metrics = []
        for origin in config['development_origins'] + [config['test_start']]:
            final = origin == config['test_start']
            folder = output / ('04_frozen_selection_fit/models' if final else '03_frozen_development/'+origin) / spec['base_id']
            roles = frozen_roles(frame, origin, config)
            bundle = fit_model(roles, config, spec, folder, 'frozen')
            attach(bundle, roles['calibrate'], config)
            pred, metrics = score(bundle, roles['outer'], origin, config)
            csv_write(pred, folder / 'predictions.csv')
            json_write(folder / 'scores.json', metrics)
            csv_write(roles['calibrate'][['row_id','date','recloser','outage_label']].assign(
                raw_probability=raw_predict(bundle, roles['calibrate'])), folder / 'calibration.csv')
            json_write(folder / 'calibration_and_warnings.json', dict(calibrator=bundle['calibrator'].summary(), warnings=bundle['warnings']))
            joblib.dump(bundle, folder / 'model.joblib')
            stages.complete(folder)
            all_metrics += metrics
        csv_write(pd.DataFrame(all_metrics), output / 'frozen_metrics.csv')
    else:
        # Narrow the existing original orchestration to exactly the three
        # requested history candidates. All fitting/calibration code is unchanged.
        stages.monthly_specs = lambda: [dict(kind='xgboost', family='history', window=w,
                         base_id='xgboost__history__'+w) for w in ['12m','24m','all']]
        stages.monthly_predictions(output, config, 'development')
        pred, monthly = stages.monthly_predictions(output, config, 'evaluation')
        csv_write(stages.aggregate(pred, monthly, config), output / 'overall_metrics.csv')
    print('Completed focused refit: '+str(output))

if __name__ == '__main__':
    main()
