"""Four registered candidates per learner; chronological AP selection and separate calibration."""
from itertools import product
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lib.modeling import metrics,freeze_threshold,SigmoidCalibrator,xgb_model
from .common import csv_write,json_write
from .preprocessing import Transformer


def candidates(kind):
    if kind == 'spline_logistic':
        return [{'knots':k,'C':c} for k,c in product([4,6],[0.1,1.0])]
    return [{'depth':d,'weight':w} for d,w in product(([2,4] if kind == 'xgboost' else [3,5] if kind == 'catboost' else [6,10]),['none','training_ratio'])]


def create_model(kind,params,config,labels,trees=None,initial=False):
    ratio = float((labels == 0).sum()/(labels == 1).sum())
    if kind == 'catboost':
        return CatBoostClassifier(iterations=trees or config['catboost_max_trees'],depth=params['depth'],
            learning_rate=0.05,l2_leaf_reg=10,loss_function='Logloss',eval_metric='Logloss',
            class_weights=[1,ratio] if params['weight'] == 'training_ratio' else [1,1],
            random_seed=config['seed'],thread_count=config['threads'],allow_writing_files=False,
            has_time=True,counter_calc_method='SkipTest',bootstrap_type='Bernoulli',subsample=0.8,
            use_best_model=initial)
    if kind == 'xgboost':
        p = {'max_depth':params['depth'],'min_child_weight':20,'reg_lambda':10,'class_weight_mode':params['weight']}
        return xgb_model(config,p,trees=trees,early=initial,label=labels)
    if kind == 'random_forest':
        return RandomForestClassifier(n_estimators=config['random_forest_trees'],max_depth=params['depth'],
            min_samples_leaf=20,max_features='sqrt',max_samples=0.8,
            class_weight=None if params['weight'] == 'none' else 'balanced',
            n_jobs=config['threads'],random_state=config['seed'])
    return LogisticRegression(C=params['C'],max_iter=4000,random_state=config['seed'])


def fit_model(model,kind,prep,fit,tune=None):
    X = prep.transform(fit)
    if tune is not None and kind == 'catboost':
        model.fit(X,fit.outage_label,cat_features=prep.categorical,
            eval_set=(prep.transform(tune),tune.outage_label),early_stopping_rounds=50,verbose=False)
    elif tune is not None and kind == 'xgboost':
        model.fit(X,fit.outage_label,eval_set=[(prep.transform(tune),tune.outage_label)],verbose=False)
    elif kind == 'catboost':
        model.fit(X,fit.outage_label,cat_features=prep.categorical,verbose=False)
    else:
        model.fit(X,fit.outage_label)
    if kind == 'spline_logistic':
        assert model.n_iter_.max() < model.max_iter, 'Resolve logistic convergence before comparison.'


def search(cohorts,config,kind,family,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fit,tune = cohorts['fit'],cohorts['tune']
    records,best,rank = [],None,None
    for trial,params in enumerate(candidates(kind),1):
        prep = Transformer(config,family,kind,params.get('knots',4)).fit(fit)
        model = create_model(kind,params,config,fit.outage_label.to_numpy(),initial=True)
        fit_model(model,kind,prep,fit,tune)
        p = model.predict_proba(prep.transform(tune))[:,1]
        score = metrics(tune.outage_label,p)
        trees = int(model.tree_count_) if kind == 'catboost' else int(model.best_iteration+1) if kind == 'xgboost' else None
        records.append({'trial':trial,**params,'trees':trees,**score})
        key = (-score['average_precision'],score['log_loss'],trial)
        if rank is None or key < rank:
            rank = key
            best = {'parameters':params,'trees':trees,'raw_threshold':freeze_threshold(tune.outage_label,p),'tune_metrics':score}
            json_write(folder / 'selected_initial_preprocessing.json',prep.summary())
        csv_write(pd.DataFrame(records),folder / 'trials.csv')
        print(f'  {kind}/{family}: trial {trial}/4, tuning AP={score["average_precision"]:.4f}',flush=True)
    json_write(folder / 'inner_selection.json',best)
    return best


def predict_final(bundle,frame):
    raw = bundle['model'].predict_proba(bundle['transformer'].transform(frame))[:,1]
    probability = bundle['calibrator'].predict(raw)
    return raw,probability,(raw >= bundle['raw_threshold']).astype(int)


def refit_calibrate(cohorts,config,kind,family,selected,folder):
    folder.mkdir(parents=True,exist_ok=True)
    refit,cal = cohorts['refit'],cohorts['calibrate']
    prep = Transformer(config,family,kind,selected['parameters'].get('knots',4)).fit(refit)
    model = create_model(kind,selected['parameters'],config,refit.outage_label.to_numpy(),trees=selected['trees'])
    fit_model(model,kind,prep,refit)
    raw = model.predict_proba(prep.transform(cal))[:,1]
    calibrator = SigmoidCalibrator().fit(raw,cal.outage_label)
    assert calibrator.model.n_iter_.max() < calibrator.model.max_iter
    bundle = {'kind':kind,'family':family,'transformer':prep,'model':model,'calibrator':calibrator,
        'raw_threshold':selected['raw_threshold'],'mapped_threshold':float(calibrator.predict(np.array([selected['raw_threshold']]))[0]),
        'selection':selected}
    joblib.dump(bundle,folder / 'model.joblib')
    if kind == 'catboost':
        model.save_model(str(folder / 'native_catboost.cbm'))
    if kind == 'xgboost':
        model.save_model(folder / 'native_xgboost.json')
    predictions = cal[['row_id','date','psgc','outage_label']].copy()
    predictions['raw_probability'],predictions['probability'] = raw,calibrator.predict(raw)
    csv_write(predictions,folder / 'calibration_predictions.csv')
    json_write(folder / 'refit_preprocessing.json',prep.summary())
    json_write(folder / 'fit_calibration_report.json',{'kind':kind,'family':family,
        'refit_first_date':str(refit.date.min()),'refit_last_date':str(refit.date.max()),
        'refit_rows':len(refit),'refit_positives':int(refit.outage_label.sum()),
        'calibration_first_date':str(cal.date.min()),'calibration_last_date':str(cal.date.max()),
        'calibration_rows':len(cal),'calibration_positives':int(cal.outage_label.sum()),
        'calibrator':calibrator.summary(),'raw_threshold':bundle['raw_threshold'],
        'mapped_threshold':bundle['mapped_threshold'],'outer_labels_used':False})
    return bundle
