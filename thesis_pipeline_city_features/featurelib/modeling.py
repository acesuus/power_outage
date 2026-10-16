import numpy as np
import pandas as pd
from .common import csv_write, json_write
from .features import FAMILIES
from monthlib.modeling import MonthlyTransformer, candidates, create_model, fit_one, raw_predict, InterceptAdjustment
from citylib.modeling import freeze_threshold, metrics


class FeatureTransformer(MonthlyTransformer):
    def __init__(self, family, kind='xgboost'):
        self.family, self.kind = family, kind
        self.fields = list(FAMILIES[family])


def fit_month(cohorts, config, family, folder):
    folder.mkdir(parents=True, exist_ok=True)
    fit, tune = cohorts['fit'], cohorts['tune']
    records, best, best_key = [], None, None
    for trial, params in enumerate(candidates('xgboost', config), 1):
        prep = FeatureTransformer(family).fit(fit)
        model = create_model('xgboost', params, config, early=True)
        fit_one(model, prep, fit, tune)
        probability = model.predict_proba(prep.transform(tune))[:, 1].astype(float)
        score = metrics(tune.outage_label, probability)
        trees = int(model.best_iteration + 1)
        key = (-score['average_precision'], score['log_loss'], trial)
        records.append(dict(trial=trial, **params, trees=trees, **score))
        if best_key is None or key < best_key:
            best_key, best = key, dict(parameters=params, trees=trees, tune_metrics=score)
            json_write(folder / 'initial_preprocessing.json', prep.summary())
    refit = cohorts['refit']
    prep = FeatureTransformer(family).fit(refit)
    model = create_model('xgboost', best['parameters'], config, trees=best['trees'])
    fit_one(model, prep, refit)
    assert model.get_booster().num_boosted_rounds() == best['trees']
    assert prep.scaler is None
    csv_write(pd.DataFrame(records), folder / 'trials.csv')
    json_write(folder / 'inner_selection.json', best)
    json_write(folder / 'preprocessing.json', prep.summary())
    return dict(model=model, transformer=prep, selection=best, family=family)


def warning_rules(cal):
    adjust = InterceptAdjustment().fit(cal.raw_probability, cal.outage_label)
    rules = {}
    for mode, probability in [('raw', cal.raw_probability.to_numpy()),
                              ('intercept_adjusted', adjust.predict(cal.raw_probability))]:
        rules[mode] = {policy: freeze_threshold(cal.outage_label, probability, 'fpr20', limit)
                       for policy, limit in [('fpr20', .2), ('fpr10', .1)]}
        rules[mode]['fixed050'] = {'threshold': .5, 'policy': 'fixed050', 'fitted': False}
    return adjust, rules


def score(frame, raw, origin, family, adjust, rules):
    pred = frame[['row_id', 'date', 'outage_label']].copy()
    pred['origin'] = origin
    scores = []
    for mode, p in [('raw', raw), ('intercept_adjusted', adjust.predict(raw))]:
        name = family + '__' + mode
        pred[name + '_probability'] = p
        for policy, rule in rules[mode].items():
            warning = (p >= rule['threshold']).astype(int)
            pred[name + '_' + policy + '_warning'] = warning
            scores.append(dict(origin=origin, family=family, mode=mode, policy=policy,
                               **metrics(frame.outage_label, p, warning)))
    for policy in ['fpr20', 'fpr10']:
        assert np.array_equal(pred[family + '__raw_' + policy + '_warning'],
                              pred[family + '__intercept_adjusted_' + policy + '_warning'])
    return pred, scores


def rank(pred, monthly, policy='fpr20'):
    reference = monthly[(monthly.family == 'full_control') & (monthly['mode'] == 'raw') & (monthly.policy == policy)]
    rows = []
    for family in FAMILIES:
        g = monthly[(monthly.family == family) & (monthly['mode'] == 'raw') & (monthly.policy == policy)]
        pair = g.merge(reference, on='origin', suffixes=('', '_control'), validate='one_to_one')
        name = family + '__raw'
        score = metrics(pred.outage_label, pred[name + '_probability'], pred[name + '_' + policy + '_warning'])
        limit = .2 if policy == 'fpr20' else .1
        rows.append(dict(family=family, inputs=len(FAMILIES[family]), policy=policy,
                         mean_monthly_AP=float(g.average_precision.mean()),
                         mean_monthly_log_loss=float(g.log_loss.mean()),
                         AP_better_months=int((pair.average_precision > pair.average_precision_control + 1e-12).sum()),
                         loss_better_months=int((pair.log_loss < pair.log_loss_control - 1e-12).sum()),
                         development_budget_met=score['false_positive_rate'] <= limit + 1e-12, **score))
    ranked = pd.DataFrame(rows)
    # These stability conditions are disclosed experimental criteria, not universal thresholds.
    full = ranked[ranked.family == 'full_control'].iloc[0]
    ranked['stable_gain_over_full'] = ((ranked.AP_better_months >= 7) &
                                       (ranked.mean_monthly_log_loss <= full.mean_monthly_log_loss))
    ranked['eligible_for_replacement'] = ranked.development_budget_met & ranked.stable_gain_over_full
    return ranked.sort_values(['eligible_for_replacement', 'development_budget_met', 'recall',
                                'mean_monthly_AP', 'mean_monthly_log_loss', 'family'],
                               ascending=[False, False, False, False, True, True])


def choose(ranked):
    eligible = ranked[ranked.eligible_for_replacement]
    if len(eligible):
        return eligible.iloc[0].to_dict(), True
    return ranked[ranked.family == 'full_control'].iloc[0].to_dict(), False
