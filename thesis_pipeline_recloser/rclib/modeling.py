"""Small registered searches, fitting-only transforms and independent chronological calibration."""
from itertools import product
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lib.modeling import metrics, freeze_threshold, SigmoidCalibrator, frequency_fit, frequency_predict
from .common import csv_write, json_write
from .data import FAMILIES, STRESS


class Transformer:
    def __init__(self, config, family, kind):
        self.config, self.family, self.kind = config, family, kind
        self.fields = list(FAMILIES[family])

    def augment(self, frame):
        f = frame.copy()
        for raw, field in zip(['precipitation_mm','wind_gust_max_kmh'], STRESS):
            if field in self.fields:
                reference = f.psgc.map(self.p90[raw]).fillna(self.global_p90[raw])
                f[field] = f[raw]/reference
        return f

    def categories(self, frame):
        f = pd.DataFrame(index=frame.index)
        for field in self.categorical:
            f[field] = frame[field].fillna('__MISSING__').astype(str) if field == 'psgc' else frame[field].map(
                lambda v: '__MISSING__' if pd.isna(v) else str(int(v)))
        return f

    def fit(self, frame):
        self.p90, self.global_p90 = {}, {}
        if any(field in self.fields for field in STRESS):
            for raw in ['precipitation_mm','wind_gust_max_kmh']:
                self.p90[raw] = frame.groupby('psgc')[raw].quantile(self.config['weather_stress_quantile']).clip(lower=1).to_dict()
                self.global_p90[raw] = max(float(frame[raw].quantile(self.config['weather_stress_quantile'])),1)
        f = self.augment(frame)
        self.categorical = [n for n in ['psgc','weather_code'] if n in self.fields]
        numeric = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in numeric if f[n].isna().all()]
        self.numeric = [n for n in numeric if n not in self.dropped]
        self.medians = f[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if f[n].isna().any()]
        self.scaler = StandardScaler().fit(f[self.numeric].fillna(self.medians)) if self.kind == 'logistic' else None
        self.encoder = None if self.kind == 'catboost' else OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32).fit(self.categories(f))
        category_names = self.categorical if self.encoder is None else list(self.encoder.get_feature_names_out(self.categorical))
        self.names = self.numeric + [n+'_missing' for n in self.indicators] + category_names
        return self

    def transform(self, frame):
        f = self.augment(frame)
        values = f[self.numeric].fillna(self.medians)
        numeric = self.scaler.transform(values).astype(np.float32) if self.scaler else values.to_numpy(dtype=np.float32)
        if self.indicators:
            numeric = np.concatenate([numeric, f[self.indicators].isna().to_numpy(dtype=np.float32)],axis=1)
        assert np.isfinite(numeric).all()
        if self.kind == 'catboost':
            count = len(self.names)-len(self.categorical)
            return pd.concat([pd.DataFrame(numeric,columns=self.names[:count],index=f.index),self.categories(f)],axis=1)
        return np.concatenate([numeric,self.encoder.transform(self.categories(f))],axis=1)

    def summary(self):
        return {'family':self.family,'kind':self.kind,'fields':self.fields,'matrix_columns':len(self.names),
            'names':self.names,'medians':self.medians,'missingness_indicators':self.indicators,
            'dropped_all_missing':self.dropped,'p90_by_recloser':self.p90,'global_p90':self.global_p90,
            'categorical_fields':self.categorical,'identity_compatibility_note':'psgc stores the recloser name, not a geographic code'}


def candidates(kind):
    if kind == 'logistic': return [{'C':c} for c in [0.01,0.1,1,10]]
    return [{'depth':d,'weight':w} for d,w in product([2,3] if kind == 'xgboost' else [3,4],['none','training_ratio'])]


def create_model(kind, params, config, y, trees=None, initial=False):
    ratio = float((y == 0).sum()/(y == 1).sum())
    if kind == 'logistic':
        return LogisticRegression(C=params['C'],max_iter=4000,random_state=config['seed'])
    if kind == 'xgboost':
        return XGBClassifier(objective='binary:logistic',tree_method='hist',eval_metric='logloss',
            n_jobs=config['threads'],random_state=config['seed'],learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,n_estimators=trees or config['max_trees'],
            early_stopping_rounds=config['patience'] if initial else None,
            max_depth=params['depth'],min_child_weight=5,reg_lambda=10,
            scale_pos_weight=ratio if params['weight']=='training_ratio' else 1)
    return CatBoostClassifier(iterations=trees or config['max_trees'],depth=params['depth'],
        learning_rate=0.05,l2_leaf_reg=10,loss_function='Logloss',eval_metric='Logloss',
        class_weights=[1,ratio] if params['weight']=='training_ratio' else [1,1],
        random_seed=config['seed'],thread_count=config['threads'],allow_writing_files=False,
        has_time=True,counter_calc_method='SkipTest',bootstrap_type='Bernoulli',subsample=0.8,use_best_model=initial)


def fit_model(model,kind,prep,frame,config,tune=None):
    X = prep.transform(frame)
    if kind == 'catboost':
        kwargs = {'cat_features':prep.categorical,'verbose':False}
        if tune is not None:
            kwargs.update(eval_set=(prep.transform(tune),tune.outage_label),early_stopping_rounds=config['patience'])
        model.fit(X,frame.outage_label,**kwargs)
    elif kind == 'xgboost' and tune is not None:
        model.fit(X,frame.outage_label,eval_set=[(prep.transform(tune),tune.outage_label)],verbose=False)
    else:
        model.fit(X,frame.outage_label)
    if kind == 'logistic': assert model.n_iter_.max() < model.max_iter


def search(cohorts,config,kind,family,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fit,tune = cohorts['fit'],cohorts['tune']
    records,best,rank = [],None,None
    for trial,params in enumerate(candidates(kind),1):
        prep = Transformer(config,family,kind).fit(fit)
        model = create_model(kind,params,config,fit.outage_label.to_numpy(),initial=True)
        fit_model(model,kind,prep,fit,config,tune)
        p = model.predict_proba(prep.transform(tune))[:,1]
        score = metrics(tune.outage_label,p)
        trees = int(model.tree_count_) if kind=='catboost' else int(model.best_iteration+1) if kind=='xgboost' else None
        key = (-score['average_precision'],score['log_loss'],trial)
        records.append({'trial':trial,**params,'trees':trees,**score})
        if rank is None or key < rank:
            rank,best = key,{'parameters':params,'trees':trees,'raw_threshold':freeze_threshold(tune.outage_label,p),'tune_metrics':score}
            json_write(folder / 'selected_initial_preprocessing.json',prep.summary())
        csv_write(pd.DataFrame(records),folder / 'trials.csv')
    json_write(folder / 'inner_selection.json',best)
    return best


def fit_final(cohorts,config,kind,family,selected,folder):
    folder.mkdir(parents=True,exist_ok=True)
    f,cal = cohorts['refit'],cohorts['calibrate']
    prep = Transformer(config,family,kind).fit(f)
    model = create_model(kind,selected['parameters'],config,f.outage_label.to_numpy(),selected['trees'])
    fit_model(model,kind,prep,f,config)
    raw = model.predict_proba(prep.transform(cal))[:,1]
    calibrator = SigmoidCalibrator().fit(raw,cal.outage_label)
    assert calibrator.model.n_iter_.max() < calibrator.model.max_iter
    bundle = {'kind':kind,'family':family,'transformer':prep,'model':model,'calibrator':calibrator,
        'raw_threshold':selected['raw_threshold'],'selection':selected}
    joblib.dump(bundle,folder / 'model.joblib')
    json_write(folder / 'preprocessing.json',prep.summary())
    json_write(folder / 'fit_report.json',{'refit_rows':len(f),'refit_positives':int(f.outage_label.sum()),
        'refit_end':f.date.max(),'calibration_rows':len(cal),'calibration_positives':int(cal.outage_label.sum()),
        'calibration_start':cal.date.min(),'calibration_end':cal.date.max(),'calibrator':calibrator.summary(),
        'threshold_selected_on':'earlier tuning quarter, raw probabilities','2026_used':False})
    prediction = cal[['row_id','date','recloser','outage_label']].copy()
    prediction['raw_probability'],prediction['probability'] = raw,calibrator.predict(raw)
    csv_write(prediction,folder / 'calibration_predictions.csv')
    return bundle


def frequency_bundle(cohorts,folder):
    folder.mkdir(parents=True,exist_ok=True)
    initial = frequency_fit(cohorts['fit'],30)
    threshold = freeze_threshold(cohorts['tune'].outage_label,frequency_predict(initial,cohorts['tune']))
    bundle = {'kind':'frequency','model':frequency_fit(cohorts['refit'],30),'raw_threshold':threshold}
    joblib.dump(bundle,folder / 'model.joblib')
    json_write(folder / 'fit_report.json',{'model':bundle['model'],'threshold':threshold,'2026_used':False})
    return bundle


def predict(bundle,frame):
    if bundle['kind']=='frequency':
        p = frequency_predict(bundle['model'],frame)
        return p,p,(p>=bundle['raw_threshold']).astype(int)
    raw = bundle['model'].predict_proba(bundle['transformer'].transform(frame))[:,1]
    return raw,bundle['calibrator'].predict(raw),(raw>=bundle['raw_threshold']).astype(int)


def select(development):
    eligible = development[development.kind.ne('frequency')]
    means = eligible.groupby(['model_id','kind','family']).agg(
        mean_log_loss=('log_loss','mean'),mean_average_precision=('average_precision','mean'),
        mean_brier_score=('brier_score','mean'),mean_roc_auc=('roc_auc','mean'),folds=('fold','nunique')).reset_index()
    assert means.folds.eq(4).all()
    ranked = means.sort_values(['mean_log_loss','mean_average_precision','model_id'],ascending=[True,False,True])
    return ranked,ranked.iloc[0].to_dict()
