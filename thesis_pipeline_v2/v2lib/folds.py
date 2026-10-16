"""Calendar-date roles and maturity checks; no splitting individual barangay rows."""
import numpy as np
import pandas as pd


def boundary_date(date, buffer):
    return (pd.Timestamp(date) - pd.Timedelta(days=1 + buffer)).strftime('%Y-%m-%d')


def end_timestamp(date):
    return (pd.Timestamp(date).tz_localize('Asia/Manila') - pd.Timedelta(seconds=1)).isoformat()


def definition(origin, buffer=14):
    T = pd.Timestamp(origin)
    tune, calibration, end = T - pd.DateOffset(months=6), T - pd.DateOffset(months=3), T + pd.DateOffset(months=3)
    return {'origin': origin,
        'fit': {'start': None, 'end': boundary_date(tune, buffer), 'asof': end_timestamp(tune), 'window_anchor': tune.strftime('%Y-%m-%d')},
        'tune': {'start': tune.strftime('%Y-%m-%d'), 'end': boundary_date(calibration, buffer), 'asof': end_timestamp(calibration)},
        'refit': {'start': None, 'end': boundary_date(calibration, buffer), 'asof': end_timestamp(calibration), 'window_anchor': calibration.strftime('%Y-%m-%d')},
        'calibrate': {'start': calibration.strftime('%Y-%m-%d'), 'end': boundary_date(T, buffer), 'asof': end_timestamp(T)},
        'outer': {'start': origin, 'end': boundary_date(end, buffer), 'asof': end_timestamp(end)}}


def index_role(frame, evidence, role, window, config):
    start = role['start'] or config['warmup_start']
    if role['start'] is None and window != 'all':
        months = int(window[:-1])
        start = max(start, (pd.Timestamp(role['window_anchor']) - pd.DateOffset(months=months)).strftime('%Y-%m-%d'))
    mask = frame.date.between(start, role['end']) & frame.date.ge(config['warmup_start'])
    mask &= frame.unlocalized_negative_uncertainty.eq(0)
    earliest = evidence.assign(pub=pd.to_datetime(evidence.published_at_local, utc=True)).groupby(['psgc', 'date']).pub.min()
    known = earliest[earliest <= pd.Timestamp(role['asof']).tz_convert('UTC')].reset_index()[['psgc', 'date']]
    keys = pd.MultiIndex.from_frame(known)
    mature = frame.outage_label.eq(0) | pd.MultiIndex.from_frame(frame[['psgc', 'date']]).isin(keys)
    rows = frame[mask & mature]
    assert rows.outage_label.nunique() == 2, 'Both classes required in each role.'
    return rows.row_id.to_numpy(dtype=np.int64)


def get_cohorts(frame, archive):
    with np.load(archive) as stored:
        return {role: frame.iloc[stored[role]].copy() for role in stored.files}
