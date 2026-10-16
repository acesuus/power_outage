from itertools import product
import numpy as np
import pandas as pd
from .common import city_run,csv_read


def load_inputs(config):
    base=city_run(config)
    f=csv_read(base / '02_features/city_feature_panel.csv')
    e=csv_read(base / '01_audit/city_outage_evidence.csv')
    earliest=e.assign(pub=pd.to_datetime(e.published_at_local,utc=True)).groupby('date').pub.min()
    f['first_support_utc']=pd.to_datetime(f.date.map(earliest),utc=True)
    assert f.row_id.equals(pd.Series(np.arange(len(f)),name='row_id'))
    return f


def origins(config,phase):
    start,end=(config['prequential_start'],config['development_end']) if phase=='development' else (config['test_start'],config['test_end'])
    return pd.date_range(start,pd.Timestamp(end).replace(day=1),freq='MS').strftime('%Y-%m-%d').tolist()


def specs(config):
    return [{'base_id':kind+'__'+family+'__'+window,'kind':kind,'family':family,'window':window}
        for kind,family,window in product(config['algorithms'],config['families'],config['training_windows'])]


def role_definition(origin,window,config):
    t=pd.Timestamp(origin);tune=t-pd.DateOffset(months=config['tuning_months'])
    start=pd.Timestamp(config['warmup_start'])
    if window!='all':start=max(start,t-pd.DateOffset(months=int(window[:-1])))
    mature=lambda d:(d-pd.Timedelta(days=config['reporting_buffer_days']+1)).strftime('%Y-%m-%d')
    return {'origin':origin,'window':window,'asof':(t.tz_localize('Asia/Manila')-pd.Timedelta(days=1)+pd.Timedelta(hours=20)).isoformat(),
        'fit':{'start':start.strftime('%Y-%m-%d'),'end':mature(tune)},
        'tune':{'start':tune.strftime('%Y-%m-%d'),'end':mature(t)},
        'refit':{'start':start.strftime('%Y-%m-%d'),'end':mature(t)},
        'calibration':{'start':(t-pd.DateOffset(months=config['calibration_months'])).strftime('%Y-%m-%d'),'end':mature(t)}}


def available(frame,asof):
    cutoff=pd.Timestamp(asof).tz_convert('UTC')
    return frame.outage_label.eq(0)|(frame.first_support_utc<=cutoff)


def cohorts(frame,definition):
    known=available(frame,definition['asof'])
    result={role:frame[frame.date.between(definition[role]['start'],definition[role]['end'])&known].copy()
        for role in ['fit','tune','refit','calibration']}
    for role,g in result.items():
        assert g.outage_label.nunique()==2,'Both classes required: '+role+' '+definition['origin']
        assert g.date.max()<definition['origin']
    return result


def target_month(frame,origin,config,phase):
    end=(pd.Timestamp(origin)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    if phase=='pool':end=min(end,frame.date.max())
    else:end=min(end,config['development_end'] if phase=='development' else config['test_end'])
    f=frame[frame.date.between(origin,end)].copy()
    # Development truth must be available before the first 2026 prediction.
    if phase=='development' and origin>=config['development_start']:
        f=f[available(f,config['selection_asof'])]
    if phase=='evaluation':f=f[available(f,config['test_truth_asof'])]
    return f


def earlier_predictions(pool,calibration):
    assert not pool.duplicated('row_id').any()
    merged=calibration[['row_id','date','outage_label']].merge(pool[['row_id','origin','raw_probability']],on='row_id',validate='one_to_one')
    assert len(merged)==len(calibration),'Missing genuinely earlier out-of-sample predictions.'
    assert (merged.origin<=merged.date).all()
    return merged
