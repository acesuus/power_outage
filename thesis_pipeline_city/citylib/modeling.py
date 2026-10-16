"""Small chronological searches, optional monotone calibration, frozen warning rules."""
from itertools import product
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lib.modeling import metrics as original_metrics
from .common import csv_write, json_write
from .data import FAMILIES, FEATURES, FORBIDDEN

SMOOTH_FIELDS = ['rain_mean_mm','rain_max_mm','rain_hourly_peak_max_mm','gust_max_kmh',
    'temperature_mean_c','humidity_mean_pct','veg5_mean','veg5_spatial_p90']


def metrics(y,p,decision=None):
    result = original_metrics(y,p,decision=decision)
    result['specificity'] = result['tn']/max(result['tn']+result['fp'],1)
    result['false_positive_rate'] = result['fp']/max(result['tn']+result['fp'],1)
    result['balanced_accuracy'] = (result['recall']+result['specificity'])/2
    result['warnings'] = result['tp']+result['fp']
    return result


class Transformer:
    def __init__(self,family,kind):
        self.family,self.kind = family,kind
        self.fields = list(FAMILIES[family])
        assert set(self.fields).isdisjoint(FORBIDDEN) and set(self.fields)<=set(FEATURES)

    def categories(self,f):
        return pd.DataFrame({n:f[n].map(lambda x:'__MISSING__' if pd.isna(x) else str(int(x)))
            for n in self.categorical},index=f.index)

    def numeric_values(self,f):
        values = f[self.numeric].fillna(self.medians)
        pieces = [values[self.linear].to_numpy(dtype=float)]
        if self.spline is not None: pieces.append(self.spline.transform(values[self.smooth]))
        if self.indicators: pieces.append(f[self.indicators].isna().to_numpy(dtype=float))
        return np.concatenate(pieces,axis=1)

    def fit(self,f):
        self.categorical = [n for n in ['day_of_week','weather_code'] if n in self.fields]
        numeric = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in numeric if f[n].isna().all()]
        self.numeric = [n for n in numeric if n not in self.dropped]
        self.medians = f[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if f[n].isna().any()]
        self.smooth = [n for n in SMOOTH_FIELDS if n in self.numeric and f[n].nunique()>=3] if self.kind=='spline_logistic' else []
        self.linear = [n for n in self.numeric if n not in self.smooth]
        self.spline = SplineTransformer(n_knots=3,degree=2,knots='uniform',include_bias=False,extrapolation='constant').fit(
            f[self.smooth].fillna(self.medians)) if self.smooth else None
        spline_names = list(self.spline.get_feature_names_out(self.smooth)) if self.spline else []
        self.numeric_names = self.linear+spline_names+[n+'_missing' for n in self.indicators]
        matrix = self.numeric_values(f)
        self.scaler = StandardScaler().fit(matrix) if self.kind!='catboost' else None
        self.encoder = None if self.kind=='catboost' else OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(self.categories(f))
        self.names = self.numeric_names+(self.categorical if self.encoder is None else list(self.encoder.get_feature_names_out(self.categorical)))
        return self

    def transform(self,f):
        numeric = self.numeric_values(f)
        if self.scaler: numeric = self.scaler.transform(numeric)
        assert np.isfinite(numeric).all()
        if self.kind=='catboost':
            return pd.concat([pd.DataFrame(numeric,columns=self.numeric_names,index=f.index),self.categories(f)],axis=1)
        return np.concatenate([numeric,self.encoder.transform(self.categories(f))],axis=1)

    def summary(self):
        return {'fields':self.fields,'numeric_fields':self.numeric,'categorical_fields':self.categorical,
            'medians':self.medians,'missingness_indicators':self.indicators,'dropped_all_missing':self.dropped,
            'smooth_fields':self.smooth,'spline_knots': [s.t.tolist() for s in self.spline.bsplines_] if self.spline else [],
            'scaler_mean':self.scaler.mean_.tolist() if self.scaler else [],
            'encoded_names':self.names,'matrix_columns':len(self.names),
            'fitting_only':True,'kind':self.kind,'family':self.family}


class MonotoneCalibrator:
    def fit(self,p,y,penalty=0.001):
        x = logit(np.clip(np.asarray(p),1e-7,1-1e-7)); y = np.asarray(y,dtype=float)
        assert len(np.unique(y))==2
        def objective(theta):
            a,b = theta; z = a*x+b
            value = np.mean(np.logaddexp(0,z)-y*z)+penalty*(a-1)**2
            residual = expit(z)-y
            gradient = np.array([np.mean(residual*x)+2*penalty*(a-1),np.mean(residual)])
            return value,gradient
        fitted = minimize(objective,[1.,0.],jac=True,bounds=[(0,None),(None,None)],method='L-BFGS-B',
            options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
        assert fitted.success, fitted.message
        self.slope,self.intercept = map(float,fitted.x)
        self.penalty = penalty
        assert self.slope>=0
        return self

    def predict(self,p):
        return expit(self.slope*logit(np.clip(np.asarray(p),1e-7,1-1e-7))+self.intercept)

    def summary(self):
        return {'method':'nondecreasing logit-sigmoid','slope':self.slope,'intercept':self.intercept,
            'slope_penalty_towards_identity':self.penalty,'rank_reversal_allowed':False,
            'constant_if_zero_slope':self.slope==0}


def freeze_threshold(y,p,policy,limit=.2):
    y,p = np.asarray(y),np.asarray(p)
    assert set(np.unique(y))=={0,1}
    records = []
    for threshold in np.r_[1.000001,np.unique(p)]:
        w = p>=threshold
        tp = int(((y==1)&w).sum()); fp=int(((y==0)&w).sum())
        recall = tp/(y==1).sum(); fpr = fp/(y==0).sum()
        if policy=='fpr20' and fpr>limit+1e-12: continue
        objective = recall if policy=='fpr20' else (recall+1-fpr)/2
        records.append((float(objective),float(threshold),float(recall),float(fpr)))
    objective,threshold,recall,fpr = max(records)
    return {'threshold':threshold,'policy':policy,'validation_objective':objective,
        'validation_recall':recall,'validation_fpr':fpr,'fpr_limit':limit if policy=='fpr20' else None}


def candidates(kind):
    if kind!='catboost': return [{'C':c} for c in [.01,.1,1,10]]
    return [{'depth':d,'l2_leaf_reg':r} for d,r in product([2,3],[10,30])]


def create_model(kind,params,config,trees=None,initial=False):
    if kind!='catboost':
        return LogisticRegression(C=params['C'],max_iter=5000,random_state=config['seed'])
    return CatBoostClassifier(iterations=trees or config['max_trees'],depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],learning_rate=.05,loss_function='Logloss',eval_metric='Logloss',
        random_seed=config['seed'],thread_count=config['threads'],allow_writing_files=False,
        has_time=True,counter_calc_method='SkipTest',bootstrap_type='Bernoulli',subsample=.8,use_best_model=initial)


def fit_model(model,prep,frame,config,tune=None):
    if prep.kind=='catboost':
        kwargs = {'cat_features':prep.categorical,'verbose':False}
        if tune is not None: kwargs.update(eval_set=(prep.transform(tune),tune.outage_label),early_stopping_rounds=config['patience'])
        model.fit(prep.transform(frame),frame.outage_label,**kwargs)
    else:
        model.fit(prep.transform(frame),frame.outage_label)
        assert model.n_iter_.max()<model.max_iter


def search(cohorts,config,kind,family,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fit,tune = cohorts['fit'],cohorts['tune']; records=[]; best=None; best_key=None
    for trial,params in enumerate(candidates(kind),1):
        prep = Transformer(family,kind).fit(fit)
        model = create_model(kind,params,config,initial=True)
        fit_model(model,prep,fit,config,tune)
        p = model.predict_proba(prep.transform(tune))[:,1]
        score = metrics(tune.outage_label,p)
        trees = int(model.tree_count_) if kind=='catboost' else None
        key=(score['log_loss'],-score['average_precision'],trial)
        records.append({'trial':trial,**params,'trees':trees,**score})
        if best_key is None or key<best_key:
            best_key=key; best={'parameters':params,'trees':trees,'tune_metrics':score}
            json_write(folder / 'selected_initial_preprocessing.json',prep.summary())
    csv_write(pd.DataFrame(records),folder / 'trials.csv'); json_write(folder / 'inner_selection.json',best)
    return best


def fit_final(cohorts,config,kind,family,selected,folder):
    folder.mkdir(parents=True,exist_ok=True)
    f,cal = cohorts['refit'],cohorts['calibrate']
    prep = Transformer(family,kind).fit(f)
    model = create_model(kind,selected['parameters'],config,selected['trees'])
    fit_model(model,prep,f,config)
    raw = model.predict_proba(prep.transform(cal))[:,1]
    calibrator = MonotoneCalibrator().fit(raw,cal.outage_label,config['calibration_slope_penalty'])
    probabilities = {'raw':raw,'monotone_sigmoid':calibrator.predict(raw)}
    warnings = {mode:{policy:freeze_threshold(cal.outage_label,p,policy,config['false_positive_rate_limit'])
        for policy in config['warning_policies']} for mode,p in probabilities.items()}
    bundle={'kind':kind,'family':family,'transformer':prep,'model':model,'calibrator':calibrator,'warnings':warnings,'selection':selected}
    joblib.dump(bundle,folder / 'model.joblib')
    json_write(folder / 'preprocessing.json',prep.summary())
    json_write(folder / 'fit_report.json',{'refit_rows':len(f),'refit_positives':int(f.outage_label.sum()),
        'refit_end':f.date.max(),'calibration_rows':len(cal),'calibration_positives':int(cal.outage_label.sum()),
        'calibration_start':cal.date.min(),'calibration_end':cal.date.max(),'calibrator':calibrator.summary(),
        'warnings':warnings,'threshold_selected_on':'separate calibration period using refitted model predictions',
        'class_weighting':'none; natural city prevalence','2026_rows_used':False})
    pred=cal[['row_id','date','outage_label']].copy()
    for mode,p in probabilities.items(): pred[mode+'_probability']=p
    csv_write(pred,folder / 'calibration_predictions.csv')
    return bundle


def baseline_probabilities(bundle,f):
    if bundle['kind']=='frequency': return np.repeat(bundle['rate'],len(f))
    return pd.to_datetime(f.date).dt.month.map(bundle['month_rates']).to_numpy(dtype=float)


def baseline_bundle(cohorts,config,kind,folder):
    f=cohorts['refit']; rate=float(f.outage_label.mean())
    month=pd.to_datetime(f.date).dt.month
    g=f.assign(month=month).groupby('month').outage_label.agg(['sum','count'])
    rates={m:float((g.loc[m,'sum']+30*rate)/(g.loc[m,'count']+30)) if m in g.index else rate for m in range(1,13)}
    bundle={'kind':kind,'rate':rate,'month_rates':rates,'seasonal_pseudo_days':30}
    p=baseline_probabilities(bundle,cohorts['calibrate'])
    bundle['warnings']={'raw':{policy:freeze_threshold(cohorts['calibrate'].outage_label,p,policy,
        config['false_positive_rate_limit']) for policy in config['warning_policies']}}
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,folder / 'model.joblib');json_write(folder / 'fit_report.json',bundle)
    return bundle


def predict(bundle,f,mode='raw',policy='balanced_accuracy'):
    if bundle['kind'] in ['frequency','seasonal_frequency']:
        p=baseline_probabilities(bundle,f)
    else:
        raw=bundle['model'].predict_proba(bundle['transformer'].transform(f))[:,1]
        p=raw if mode=='raw' else bundle['calibrator'].predict(raw)
    threshold=bundle['warnings'][mode][policy]['threshold']
    return p,(p>=threshold).astype(int)


def select(development):
    learned=development[~development.kind.isin(['frequency','seasonal_frequency'])]
    means=learned.groupby(['model_id','base_id','kind','family','mode']).agg(mean_log_loss=('log_loss','mean'),
        mean_average_precision=('average_precision','mean'),mean_roc_auc=('roc_auc','mean'),
        mean_brier_score=('brier_score','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),folds=('fold','nunique')).reset_index()
    assert means.folds.eq(4).all()
    ranked=means.sort_values(['mean_log_loss','mean_average_precision','model_id'],ascending=[True,False,True])
    return ranked,ranked.iloc[0].to_dict()
