from itertools import product
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from .common import csv_write, json_write
from .data import FAMILIES, FORBIDDEN
from citylib.modeling import MonotoneCalibrator, metrics, freeze_threshold
from monthlib.modeling import InterceptAdjustment
from lib.modeling import freeze_threshold as f1_threshold

STRESS = {'precipitation_vs_fit_p90': 'precipitation_mm', 'gust_vs_fit_p90': 'wind_gust_max_kmh'}
SMOOTH = ['rain_mean_mm', 'rain_max_mm', 'rain_hourly_peak_max_mm', 'gust_max_kmh',
          'temperature_mean_c', 'humidity_mean_pct', 'veg5_mean', 'veg5_spatial_p90',
          'precipitation_mm', 'wind_gust_max_kmh', 'veg_buffer_5m']


class Transformer:
    def __init__(self, family, kind):
        self.family, self.kind = family, kind
        self.fields = list(FAMILIES[family])
        assert set(self.fields).isdisjoint(FORBIDDEN)

    def augment(self, frame):
        if not self.p90: return frame
        result = frame.copy()
        for field, raw in STRESS.items():
            if field in self.fields:
                result[field] = result[raw] / result.psgc.map(self.p90[raw]).fillna(self.global_p90[raw])
        return result

    def categories(self, f):
        return pd.DataFrame({n: f[n].fillna('__MISSING__').astype(str) if n in ['psgc', 'recloser_id'] else
                            f[n].map(lambda x: '__MISSING__' if pd.isna(x) else str(int(x)))
                             for n in self.categorical}, index=f.index)

    def numbers(self, f):
        values = f[self.numeric].fillna(self.medians)
        pieces = [values[self.linear].to_numpy(dtype=float)]
        if self.spline is not None: pieces.append(self.spline.transform(values[self.smooth]))
        if self.indicators: pieces.append(f[self.indicators].isna().to_numpy(dtype=float))
        return np.concatenate(pieces, axis=1)

    def fit(self, frame):
        self.p90, self.global_p90 = {}, {}
        for field, raw in STRESS.items():
            if field in self.fields:
                self.p90[raw] = frame.groupby('psgc')[raw].quantile(.9).clip(lower=1).to_dict()
                self.global_p90[raw] = max(float(frame[raw].quantile(.9)), 1)
        f = self.augment(frame)
        self.categorical = [n for n in ['psgc', 'recloser_id', 'day_of_week', 'weather_code'] if n in self.fields]
        names = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in names if f[n].isna().all()]
        self.numeric = [n for n in names if n not in self.dropped]
        self.medians = f[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if f[n].isna().any()]
        self.smooth = [n for n in SMOOTH if n in self.numeric and f[n].nunique() >= 3] if self.kind == 'spline_logistic' else []
        self.linear = [n for n in self.numeric if n not in self.smooth]
        self.spline = SplineTransformer(n_knots=3, degree=2, knots='uniform', include_bias=False,
                         extrapolation='constant').fit(f[self.smooth].fillna(self.medians)) if self.smooth else None
        self.numeric_names = self.linear + (list(self.spline.get_feature_names_out(self.smooth)) if self.spline else [])
        self.numeric_names += [n + '_missing' for n in self.indicators]
        values = self.numbers(f)
        self.scaler = StandardScaler().fit(values) if self.kind in ['logistic', 'spline_logistic'] else None
        self.encoder = None if self.kind == 'catboost' else OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(self.categories(f))
        self.names = self.numeric_names + (self.categorical if self.encoder is None else list(self.encoder.get_feature_names_out(self.categorical)))
        return self

    def transform(self, frame):
        f = self.augment(frame)
        numeric = self.numbers(f)
        if self.scaler is not None: numeric = self.scaler.transform(numeric)
        assert np.isfinite(numeric).all()
        if self.kind == 'catboost':
            return pd.concat([pd.DataFrame(numeric, columns=self.numeric_names, index=f.index), self.categories(f)], axis=1)
        return np.concatenate([numeric, self.encoder.transform(self.categories(f))], axis=1)

    def summary(self):
        return dict(family=self.family, kind=self.kind, fields=self.fields, medians=self.medians,
                    categorical_fields=self.categorical, missing_indicators=self.indicators, dropped=self.dropped,
                    encoded_names=self.names, p90=self.p90, global_p90=self.global_p90,
                    scaler_mean=[] if self.scaler is None else self.scaler.mean_.tolist(),
                    spline_knots=[s.t.tolist() for s in self.spline.bsplines_] if self.spline else [], fitting_only=True)


def candidates(kind):
    if kind in ['logistic', 'spline_logistic']: return [dict(C=c) for c in [.01, .1, 1, 10]]
    if kind == 'random_forest': return [dict(depth=d, leaf=l) for d, l in product([6, 10], [10, 20])]
    return [dict(depth=d, l2=l) for d, l in product([2, 3], [10, 30])]


def create(kind, params, config, trees=None, initial=False):
    if kind in ['logistic', 'spline_logistic']:
        return LogisticRegression(C=params['C'], max_iter=5000, random_state=config['seed'])
    if kind == 'random_forest':
        return RandomForestClassifier(n_estimators=config['forest_trees'], max_depth=params['depth'],
            min_samples_leaf=params['leaf'], max_features='sqrt', max_samples=.8, n_jobs=config['threads'],
            random_state=config['seed'], class_weight=None)
    if kind == 'xgboost':
        return XGBClassifier(objective='binary:logistic', tree_method='hist', eval_metric='logloss',
            n_estimators=trees if trees is not None else config['max_trees'], max_depth=params['depth'],
            reg_lambda=params['l2'], min_child_weight=5, learning_rate=.05, subsample=.8, colsample_bytree=.8,
            scale_pos_weight=1, random_state=config['seed'], n_jobs=config['threads'],
            early_stopping_rounds=config['patience'] if initial else None)
    return CatBoostClassifier(iterations=trees or config['max_trees'], depth=params['depth'], l2_leaf_reg=params['l2'],
        learning_rate=.05, loss_function='Logloss', eval_metric='Logloss', random_seed=config['seed'],
        thread_count=config['threads'], allow_writing_files=False, has_time=True, counter_calc_method='SkipTest',
        bootstrap_type='Bernoulli', subsample=.8, use_best_model=initial)


def fit_one(model, prep, frame, config, tune=None):
    kwargs = {}
    if prep.kind == 'catboost':
        kwargs.update(cat_features=prep.categorical, verbose=False)
        if tune is not None:
            kwargs.update(eval_set=(prep.transform(tune), tune.outage_label), early_stopping_rounds=config['patience'])
    elif prep.kind == 'xgboost':
        kwargs['verbose'] = False
        if tune is not None: kwargs['eval_set'] = [(prep.transform(tune), tune.outage_label)]
    model.fit(prep.transform(frame), frame.outage_label, **kwargs)
    if prep.kind in ['logistic', 'spline_logistic']: assert model.n_iter_.max() < model.max_iter


def fit_model(roles, config, spec, folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    best, best_key, records = None, None, []
    for trial, params in enumerate(candidates(spec['kind']), 1):
        prep = Transformer(spec['family'], spec['kind']).fit(roles['fit'])
        model = create(spec['kind'], params, config, initial=True)
        fit_one(model, prep, roles['fit'], config, roles['tune'])
        p = raw_predict(dict(model=model, transformer=prep), roles['tune'])
        score = metrics(roles['tune'].outage_label, p)
        trees = int(model.best_iteration+1) if spec['kind'] == 'xgboost' else int(model.tree_count_) if spec['kind'] == 'catboost' else None
        key = (score['log_loss'], -score['average_precision'], trial) if protocol == 'frozen' else (-score['average_precision'], score['log_loss'], trial)
        records.append(dict(trial=trial, **params, trees=trees, **score))
        if best_key is None or key < best_key:
            best_key, best = key, dict(parameters=params, trees=trees, tune_metrics=score)
            json_write(folder / 'initial_preprocessing.json', prep.summary())
    prep = Transformer(spec['family'], spec['kind']).fit(roles['refit'])
    model = create(spec['kind'], best['parameters'], config, trees=best['trees'])
    fit_one(model, prep, roles['refit'], config)
    if spec['kind'] == 'xgboost': assert model.get_booster().num_boosted_rounds() == best['trees']
    if spec['kind'] == 'catboost': assert model.tree_count_ == best['trees']
    csv_write(pd.DataFrame(records), folder / 'trials.csv')
    json_write(folder / 'selection.json', best)
    json_write(folder / 'preprocessing.json', prep.summary())
    return dict(spec=spec, model=model, transformer=prep, selection=best, protocol=protocol)


def raw_predict(bundle, frame):
    return np.asarray(bundle['model'].predict_proba(bundle['transformer'].transform(frame))[:, 1], dtype=float)


def warning_rules(y, p):
    result = dict(balanced_accuracy=freeze_threshold(y, p, 'balanced_accuracy'))
    for label, limit in [('fpr20', .2), ('fpr10', .1)]:
        result[label] = freeze_threshold(y, p, 'fpr20', limit)
        result[label]['policy'] = label
    result['f1'] = dict(threshold=f1_threshold(y, p), policy='f1')
    result['fixed050'] = dict(threshold=.5, policy='fixed050', fitted=False)
    return result


def attach(bundle, calibration, config, raw=None):
    p = raw_predict(bundle, calibration) if raw is None else np.asarray(raw, dtype=float)
    y = calibration.outage_label
    calibrator = (MonotoneCalibrator().fit(p, y, config['calibration_slope_penalty']) if bundle['protocol'] == 'frozen'
                  else InterceptAdjustment().fit(p, y))
    mode = 'monotone_sigmoid' if bundle['protocol'] == 'frozen' else 'intercept_adjusted'
    probs = dict(raw=p, **{mode: calibrator.predict(p)})
    bundle['calibrator'] = calibrator
    bundle['warnings'] = {name: warning_rules(y, values) for name, values in probs.items()}
    return bundle


def score(bundle, target, origin, config):
    raw = raw_predict(bundle, target)
    result = target[['row_id', 'date', 'recloser', 'outage_label']].copy()
    result['origin'] = origin
    records = []
    for mode, rules in bundle['warnings'].items():
        name = bundle['spec']['base_id'] + '__' + mode
        p = raw if mode == 'raw' else bundle['calibrator'].predict(raw)
        result[name + '_probability'] = p
        for policy in config['warning_policies']:
            w = (p >= rules[policy]['threshold']).astype(int)
            result[name + '_' + policy + '_warning'] = w
            records.append(dict(origin=origin, model_id=name, **bundle['spec'], mode=mode, policy=policy,
                                **metrics(target.outage_label, p, w)))
    if bundle['protocol'] == 'monthly':
        for policy in ['fpr20', 'fpr10', 'balanced_accuracy', 'f1']:
            a = result[bundle['spec']['base_id']+'__raw_'+policy+'_warning']
            b = result[bundle['spec']['base_id']+'__intercept_adjusted_'+policy+'_warning']
            np.testing.assert_array_equal(a, b)
    return result, records
