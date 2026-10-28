"""Day-Ahead Outage Forecast: frozen scoring and smooth calibration."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE/'feature_schema.json').read_text(encoding='utf-8'))
FEATURES = SCHEMA['features']
META = SCHEMA['metadata']
CALENDAR = ['day_of_week','is_weekend','day_of_year_sin','day_of_year_cos','month_sin','month_cos']


def calendar_features(dates):
    d = pd.to_datetime(dates, format='%Y-%m-%d', errors='raise')
    angle = 2*np.pi*(d.dt.dayofyear-1)/np.where(d.dt.is_leap_year,366,365)
    return pd.DataFrame({'day_of_week':d.dt.dayofweek,'is_weekend':d.dt.dayofweek.ge(5).astype(int),
        'day_of_year_sin':np.sin(angle),'day_of_year_cos':np.cos(angle),
        'month_sin':np.sin(2*np.pi*(d.dt.month-1)/12),'month_cos':np.cos(2*np.pi*(d.dt.month-1)/12)})


def categorical_strings(frame):
    data = pd.DataFrame(frame).copy().astype(object)
    data.iloc[:, 0] = data.iloc[:, 0].fillna('<missing>').astype(str)
    data.iloc[:, 1] = data.iloc[:, 1].map(lambda x: '<missing>' if pd.isna(x) else format(float(x), '.12g'))
    return data.to_numpy(dtype=object)


def raw_predict(specification, frame):
    return np.clip(sum(c['weight'] * c['pipeline'].predict_proba(frame[FEATURES])[:, 1]
                       for c in specification['components']), 1e-15, 1-1e-15)


def calibrate(raw, parameters):
    raw = np.clip(np.asarray(raw, float), 1e-15, 1-1e-15)
    if parameters['method'] == 'identity':
        return raw
    if parameters['method'] == 'sigmoid':
        return expit(parameters['a'] * logit(raw) + parameters['intercept'])
    if parameters['method'] == 'sigmoid_raw':
        return expit(parameters['a'] * ((raw-parameters['mean'])/parameters['scale']) + parameters['intercept'])
    return expit(parameters['a'] * np.log(raw) - parameters['b'] * np.log1p(-raw) + parameters['intercept'])


class OutageForecast:
    def __init__(self, directory=HERE):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'model_manifest.json').read_text())
        path = self.directory/'model.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['model_sha256']:
            raise ValueError('Saved model differs from its manifest.')
        self.bundle = joblib.load(path)

    def predict(self, frame):
        expected = set(FEATURES) | set(META)
        if set(frame) != expected:
            raise ValueError('Supply only the 66 input features and required provenance fields. Missing='+
                str(sorted(expected-set(frame)))+'; unexpected='+str(sorted(set(frame)-expected)))
        data = frame.copy().reset_index(drop=True)
        if not len(data):
            raise ValueError('No input records supplied.')
        dates = pd.to_datetime(data.date, format='%Y-%m-%d', errors='raise')
        if not dates.dt.strftime('%Y-%m-%d').equals(data.date):
            raise ValueError('Use YYYY-MM-DD target dates.')
        cutoff = pd.to_datetime(data.prediction_cutoff_local, utc=True)
        expected_cutoff = dates.dt.tz_localize('Asia/Manila').dt.tz_convert('UTC')-pd.Timedelta(hours=4)
        if not cutoff.equals(expected_cutoff):
            raise ValueError('Prediction cutoff must be D-1 20:00 Asia/Manila.')
        for field in ['forecast_assumed_available_at_local','history_snapshot_as_of_local']:
            if any(pd.Timestamp(v).tzinfo is None for v in data[field]):
                raise ValueError(field+' requires timezone offsets.')
            values = pd.to_datetime(data[field], utc=True)
            if values.isna().any() or values.gt(cutoff).any():
                raise ValueError(field+' must be present and no later than cutoff.')
        history = pd.to_datetime(data.history_snapshot_as_of_local, utc=True)
        if (cutoff-history).gt(pd.Timedelta(hours=24)).any():
            raise ValueError('History snapshot is stale.')
        latest = pd.to_datetime(data.latest_history_publication_local, utc=True)
        if (latest.notna() & latest.gt(history)).any():
            raise ValueError('History includes notices published after snapshot.')
        now = pd.Timestamp.now(tz='Asia/Manila')
        if not data.history_snapshot_mode.isin(['retrospective_publication_filter','operator_attested_snapshot']).all():
            raise ValueError('Unknown history provenance.')
        if (dates.ge(now.normalize().tz_localize(None)) & data.history_snapshot_mode.ne('operator_attested_snapshot')).any():
            raise ValueError('Future predictions require an operator-attested snapshot.')
        if (data.history_snapshot_mode.eq('operator_attested_snapshot') & history.gt(now.tz_convert('UTC'))).any():
            raise ValueError('Snapshot cannot be in the future.')
        if not data.recloser.isin(self.manifest['supported_reclosers']).all():
            raise ValueError('Unrecognized recloser.')
        ids = data.date+'__'+data.recloser.str.replace(' ','_',regex=False)
        if not ids.equals(data.panel_id) or ids.duplicated().any():
            raise ValueError('Panel IDs must uniquely match date__RECLOSER.')
        numeric = [f for f in FEATURES if f!='recloser']
        data[numeric] = data[numeric].apply(pd.to_numeric,errors='raise')
        if np.isinf(data[numeric].to_numpy(float)).any():
            raise ValueError('Infinite inputs are not accepted.')
        if data[['forecast_precipitation_mm','forecast_temperature_mean_c','forecast_humidity_mean_pct',
                 'forecast_wind_speed_max_kmh']].isna().any().any():
            raise ValueError('Core forecast variables must be present.')
        calendar = calendar_features(data.date)
        if not np.allclose(calendar[CALENDAR],data[CALENDAR],atol=1e-10):
            raise ValueError('Calendar values do not match target date.')
        raw = raw_predict(self.bundle['specification'],data)
        probability = calibrate(raw,self.bundle['calibration'])
        high = self.bundle['high_threshold']
        medium = self.bundle['medium_threshold']
        out = data[['date','recloser','panel_id','prediction_cutoff_local']].copy()
        out['raw_score'] = raw
        out['outage_estimate_percent'] = 100*probability
        out['risk_level'] = np.where(probability>=high,'High',np.where(probability>=medium,'Medium','Low'))
        out['warning'] = (probability>=high).astype(int)
        out['medium_min_percent'] = 100*medium
        out['high_min_percent'] = 100*high
        out['forecast_horizon'] = 'next_local_calendar_day'
        out['model_name'] = 'Day-Ahead Outage Forecast'
        return out
