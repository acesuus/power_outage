from itertools import product
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .common import csv_write,json_write
from citylib.modeling import Transformer,freeze_threshold,metrics
from citylib.data import FAMILIES


class MonthlyTransformer(Transformer):
    def fit(self,frame):
        super().fit(frame)
        if self.kind=='xgboost':self.scaler=None
        return self


class InterceptAdjustment:
    """Fixed slope 1; only probability level changes, never the monthly score order."""
    def fit(self,p,y):
        x=logit(np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7));y=np.asarray(y,dtype=float)
        assert set(np.unique(y))=={0.,1.}
        self.offset=float(brentq(lambda b:float(expit(x+b).mean()-y.mean()),-40,40,xtol=1e-12))
        self.rows=len(y);self.positives=int(y.sum());return self

    def predict(self,p):
        return expit(logit(np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7))+self.offset)

    def summary(self):return {'method':'intercept-only logit adjustment','slope':1,'offset':self.offset,
        'rows':self.rows,'positives':self.positives,'fitted_on':'earlier monthly out-of-sample predictions'}


def candidates(kind,config):
    if kind=='logistic':return [{'C':c} for c in [.01,.1,1,10]]
    return [{'max_depth':d,'reg_lambda':r} for d,r in product(config['xgboost']['max_depth'],config['xgboost']['reg_lambda'])]


def create_model(kind,params,config,trees=None,early=False):
    if kind=='logistic':return LogisticRegression(C=params['C'],max_iter=5000,random_state=config['seed'])
    x=config['xgboost']
    return XGBClassifier(objective='binary:logistic',tree_method='hist',eval_metric='logloss',
        n_estimators=trees if trees is not None else config['max_trees'],max_depth=params['max_depth'],reg_lambda=params['reg_lambda'],
        min_child_weight=x['min_child_weight'],learning_rate=x['learning_rate'],subsample=x['subsample'],
        colsample_bytree=x['colsample_bytree'],scale_pos_weight=1,n_jobs=config['threads'],random_state=config['seed'],
        early_stopping_rounds=config['patience'] if early else None)


def fit_one(model,prep,frame,tune=None):
    kwargs={}
    if prep.kind=='xgboost':
        kwargs['verbose']=False
        if tune is not None:kwargs['eval_set']=[(prep.transform(tune),tune.outage_label)]
    model.fit(prep.transform(frame),frame.outage_label,**kwargs)
    if prep.kind=='logistic':assert model.n_iter_.max()<model.max_iter


def fit_month(cohorts,config,spec,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fit,tune=cohorts['fit'],cohorts['tune'];records=[];best=None;key=None
    for trial,params in enumerate(candidates(spec['kind'],config),1):
        prep=MonthlyTransformer(spec['family'],spec['kind']).fit(fit)
        model=create_model(spec['kind'],params,config,early=True);fit_one(model,prep,fit,tune)
        p=np.asarray(model.predict_proba(prep.transform(tune))[:,1],dtype=float)
        score=metrics(tune.outage_label,p)
        trees=int(model.best_iteration+1) if spec['kind']=='xgboost' else None
        rank=(-score['average_precision'],score['log_loss'],trial)
        records.append({'trial':trial,**params,'trees':trees,**score})
        if key is None or rank<key:
            key=rank;best={'parameters':params,'trees':trees,'tune_metrics':score}
            json_write(folder / 'initial_preprocessing.json',prep.summary())
    refit=cohorts['refit'];prep=MonthlyTransformer(spec['family'],spec['kind']).fit(refit)
    model=create_model(spec['kind'],best['parameters'],config,trees=best['trees']);fit_one(model,prep,refit)
    if spec['kind']=='xgboost':assert model.get_booster().num_boosted_rounds()==best['trees']
    bundle={'spec':spec,'transformer':prep,'model':model,'selection':best}
    csv_write(pd.DataFrame(records),folder / 'trials.csv');json_write(folder / 'inner_selection.json',best)
    json_write(folder / 'preprocessing.json',prep.summary())
    return bundle


def raw_predict(bundle,frame):
    return np.asarray(bundle['model'].predict_proba(bundle['transformer'].transform(frame))[:,1],dtype=float)


def attach_calibration(bundle,cal,config):
    assert len(cal)>=config['minimum_calibration_rows']
    adjust=InterceptAdjustment().fit(cal.raw_probability,cal.outage_label)
    probs={'raw':cal.raw_probability.to_numpy(dtype=float),'intercept_adjusted':adjust.predict(cal.raw_probability)}
    rules={mode:{policy:freeze_threshold(cal.outage_label,p,policy,config['false_positive_rate_limit'])
        for policy in config['warning_policies']} for mode,p in probs.items()}
    bundle['adjustment']=adjust;bundle['warnings']=rules
    return bundle


def predict(bundle,frame,mode='raw',policy='fpr20'):
    raw=raw_predict(bundle,frame)
    p=raw if mode=='raw' else bundle['adjustment'].predict(raw)
    return p,(p>=bundle['warnings'][mode][policy]['threshold']).astype(int)


def score_month(bundle,frame,origin,config):
    records=[];operating=[];pred=frame[['row_id','date','outage_label']].copy();pred['origin']=origin
    spec=bundle['spec']
    for mode in config['probability_modes']:
        name=spec['base_id']+'__'+mode
        p,w=predict(bundle,frame,mode,config['primary_warning_policy'])
        pred[name+'_probability']=p
        for policy in config['warning_policies']:
            _,decision=predict(bundle,frame,mode,policy)
            pred[name+'_'+policy+'_warning']=decision
            operating.append({'origin':origin,'model_id':name,'policy':policy,**metrics(frame.outage_label,p,decision)})
        records.append({'origin':origin,'model_id':name,**spec,'mode':mode,**metrics(frame.outage_label,p,w)})
    assert np.array_equal(pred[spec['base_id']+'__raw_fpr20_warning'],pred[spec['base_id']+'__intercept_adjusted_fpr20_warning'])
    return pd.DataFrame(records),pd.DataFrame(operating),pred


def rank_procedures(monthly,predictions,config):
    rows=[];y=predictions.outage_label
    for name,g in monthly.groupby('model_id'):
        r=g.iloc[0]
        score=metrics(y,predictions[name+'_probability'],predictions[name+'_'+config['primary_warning_policy']+'_warning'])
        rows.append({'model_id':name,'base_id':r.base_id,'kind':r.kind,'family':r.family,'window':r.window,'mode':r['mode'],
            'mean_monthly_AP':float(g.average_precision.mean()),'mean_monthly_log_loss':float(g.log_loss.mean()),
            'development_budget_met':score['false_positive_rate']<=config['false_positive_rate_limit']+1e-12,**score})
    ranked=pd.DataFrame(rows)
    return ranked.sort_values(['development_budget_met','recall','mean_monthly_AP','mean_monthly_log_loss','model_id'],
        ascending=[False,False,False,True,True])


def choose(ranked):
    feasible=ranked[ranked.development_budget_met]
    if len(feasible):return feasible.iloc[0].to_dict(),True
    r=ranked.sort_values(['false_positive_rate','mean_monthly_AP','mean_monthly_log_loss','model_id'],ascending=[True,False,True,True])
    return r.iloc[0].to_dict(),False
