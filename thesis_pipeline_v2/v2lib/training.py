"""Nested fitting/tuning/refitting/calibration; outer outcomes never choose settings."""
from itertools import product
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lib.modeling import frequency_fit, frequency_predict, xgb_model, metrics, freeze_threshold, SigmoidCalibrator
from .common import csv_write, json_write
from .preprocessing import Transformer


def predict_raw(bundle, frame):
    if bundle['kind'] == 'frequency':
        return frequency_predict(bundle['model'], frame)
    matrix = bundle['transformer'].transform(frame, scale=bundle['kind'] == 'logistic')
    return bundle['model'].predict_proba(matrix)[:, 1]


def predict_final(bundle, frame):
    raw = predict_raw(bundle, frame)
    final = raw if bundle['calibrator'] is None else bundle['calibrator'].predict(raw)
    return raw, final, (raw >= bundle['raw_threshold']).astype(int)


def search_xgboost(cohorts, config, family, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fit, tune = cohorts['fit'], cohorts['tune']
    prep = Transformer(config, family).fit(fit)
    X, V = prep.transform(fit), prep.transform(tune)
    s, records, best = config['xgboost'], [], {}
    grid = list(product(s['max_depth'], s['min_child_weight'], s['reg_lambda'], s['class_weight_modes']))
    for i, (depth, child, l2, weight) in enumerate(grid):
        params = {'max_depth': depth, 'min_child_weight': child, 'reg_lambda': l2, 'class_weight_mode': weight}
        model = xgb_model(config, params, label=fit.outage_label.to_numpy())
        model.fit(X, fit.outage_label, eval_set=[(V, tune.outage_label)], verbose=False)
        probability = model.predict_proba(V)[:, 1]
        score = metrics(tune.outage_label, probability)
        record = {'trial': i+1, **params, 'trees': int(model.best_iteration+1), **score}
        records.append(record)
        for criterion in ['log_loss', 'average_precision']:
            rank = (score['log_loss'], -score['average_precision'], i) if criterion == 'log_loss' else (-score['average_precision'], score['log_loss'], i)
            if criterion not in best or rank < best[criterion]['rank']:
                best[criterion] = {'rank': rank, 'parameters': params, 'trees': int(model.best_iteration+1),
                    'raw_threshold': freeze_threshold(tune.outage_label, probability), 'tune_metrics': score}
        csv_write(pd.DataFrame(records), folder / 'trials.csv')
        if (i+1) % 8 == 0:
            print(f'  {folder.name}: XGBoost trial {i+1}/{len(grid)}', flush=True)
    json_write(folder / 'inner_preprocessing.json', prep.summary())
    for value in best.values():
        value.pop('rank')
    json_write(folder / 'inner_selections.json', best)
    return best


def search_logistic(cohorts, config, family, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fit, tune = cohorts['fit'], cohorts['tune']
    prep = Transformer(config, family).fit(fit)
    X, V = prep.transform(fit, scale=True), prep.transform(tune, scale=True)
    records, best = [], None
    for C in config['logistic_C']:
        model = LogisticRegression(C=C, max_iter=3000, random_state=config['seed'])
        model.fit(X, fit.outage_label)
        assert model.n_iter_.max() < model.max_iter, 'Resolve logistic convergence before comparison.'
        p = model.predict_proba(V)[:, 1]
        score = metrics(tune.outage_label, p)
        records.append({'C': C, **score})
        rank = (score['log_loss'], -score['average_precision'])
        if best is None or rank < best['rank']:
            best = {'rank': rank, 'C': C, 'raw_threshold': freeze_threshold(tune.outage_label, p), 'tune_metrics': score}
    csv_write(pd.DataFrame(records), folder / 'logistic_trials.csv')
    best.pop('rank')
    json_write(folder / 'logistic_selection.json', best)
    return best


def refit_calibrate(cohorts, config, kind, family, selected, folder):
    folder.mkdir(parents=True, exist_ok=True)
    refit, calibration = cohorts['refit'], cohorts['calibrate']
    if kind == 'frequency':
        bundle = {'kind': kind, 'model': frequency_fit(refit, config['frequency_prior_days']),
                  'raw_threshold': selected['raw_threshold'], 'transformer': None}
    else:
        prep = Transformer(config, family).fit(refit)
        if kind == 'logistic':
            model = LogisticRegression(C=selected['C'], max_iter=3000, random_state=config['seed'])
            model.fit(prep.transform(refit, scale=True), refit.outage_label)
            assert model.n_iter_.max() < model.max_iter
        else:
            model = xgb_model(config, selected['parameters'], trees=selected['trees'], early=False,
                              label=refit.outage_label.to_numpy())
            model.fit(prep.transform(refit), refit.outage_label, verbose=False)
            model.save_model(folder / 'native_xgboost.json')
        bundle = {'kind': kind, 'model': model, 'raw_threshold': selected['raw_threshold'], 'transformer': prep}
        json_write(folder / 'refit_preprocessing.json', prep.summary())
    raw = predict_raw(bundle, calibration)
    calibrator = None if kind == 'frequency' else SigmoidCalibrator().fit(raw, calibration.outage_label)
    bundle['calibrator'] = calibrator
    bundle['mapped_threshold'] = bundle['raw_threshold'] if calibrator is None else float(calibrator.predict(np.array([bundle['raw_threshold']]))[0])
    bundle['family'] = family
    bundle['selection'] = selected
    joblib.dump(bundle, folder / 'model.joblib')
    cp = raw if calibrator is None else calibrator.predict(raw)
    predictions = calibration[['row_id', 'date', 'psgc', 'outage_label']].copy()
    predictions['raw_probability'], predictions['probability'] = raw, cp
    csv_write(predictions, folder / 'calibration_predictions.csv')
    json_write(folder / 'fit_calibration_report.json', {
        'kind': kind, 'family': family, 'fit_rows': len(refit), 'fit_positives': int(refit.outage_label.sum()),
        'calibration_rows': len(calibration), 'calibration_positives': int(calibration.outage_label.sum()),
        'calibrator': calibrator.summary() if calibrator else {'method': 'frequency unchanged'},
        'raw_threshold': bundle['raw_threshold'], 'mapped_threshold': bundle['mapped_threshold'],
        'outer_labels_used_for_fitting': False})
    return bundle


def frequency_selection(cohorts, config):
    fit, tune = cohorts['fit'], cohorts['tune']
    m = frequency_fit(fit, config['frequency_prior_days'])
    p = frequency_predict(m, tune)
    return {'raw_threshold': freeze_threshold(tune.outage_label, p), 'tune_metrics': metrics(tune.outage_label, p)}


def search_signature(cohorts, family, config):
    """Reuse only identical initial fit+tune data, feature family and search configuration."""
    import json
    h = hashlib.sha256()
    for role in ['fit', 'tune']:
        h.update(cohorts[role].row_id.to_numpy(dtype=np.int64).tobytes())
    h.update(family.encode())
    h.update(json.dumps(config['xgboost'], sort_keys=True).encode())
    h.update(str(config['seed']).encode())
    return h.hexdigest()
