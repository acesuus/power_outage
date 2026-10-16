"""The 45 inputs and their source-aware, calendar-aware construction."""
import numpy as np
import pandas as pd

WEATHER_MAP = {
    'precipitation_mm': 'precipitation_sum (mm)',
    'temperature_max_c': 'temperature_2m_max (°C)',
    'temperature_min_c': 'temperature_2m_min (°C)',
    'temperature_mean_c': 'temperature_2m_mean (°C)',
    'humidity_mean_pct': 'relative_humidity_2m_mean (%)',
    'wind_speed_max_kmh': 'wind_speed_10m_max (km/h)',
    'wind_speed_mean_kmh': 'wind_speed_10m_mean (km/h)',
    'wind_gust_max_kmh': 'wind_gusts_10m_max (km/h)',
    'weather_code': 'weather_code (wmo code)',
    'apparent_temperature_max_c': 'apparent_temperature_max (°C)',
    'humidity_max_pct': 'relative_humidity_2m_max (%)',
    'humidity_min_pct': 'relative_humidity_2m_min (%)'
}
ORIGINAL_WEATHER = list(WEATHER_MAP)[:8] + ['wind_direction_sin', 'wind_direction_cos']
TARGET_WEATHER = ORIGINAL_WEATHER + ['weather_code', 'apparent_temperature_max_c',
                                      'humidity_max_pct', 'humidity_min_pct']
PRIOR = [
    'precipitation_lag1d_mm', 'wind_gust_max_lag1d_kmh',
    'temperature_mean_lag1d_c', 'humidity_mean_lag1d_pct'
] + [name for days in [3, 7] for name in [
    f'precipitation_sum_prev{days}d_mm', f'wind_gust_max_prev{days}d_kmh',
    f'temperature_mean_prev{days}d_c', f'humidity_mean_prev{days}d_pct']]
VEGETATION = ['veg_buffer_5m', 'veg_buffer_15m', 'veg_buffer_30m', 'ndvi_p90',
              'ndvi_mean_brgy', 'valid_frac_b30']
HISTORY = ['outage_days_prev3d', 'outage_days_prev7d', 'days_since_last_outage_capped7',
           'no_outage_prev7d']
EQUIPMENT = ['equipment_fault_days_prev7d', 'equipment_fault_days_prev14d']
CALENDAR = ['day_of_week', 'is_weekend', 'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
GROUPS = {'target_weather': TARGET_WEATHER, 'prior_weather': PRIOR, 'vegetation': VEGETATION,
          'outage_history': HISTORY, 'equipment_history': EQUIPMENT, 'calendar': CALENDAR,
          'identity': ['psgc']}
FEATURES = [name for group in GROUPS.values() for name in group]
assert len(FEATURES) == len(set(FEATURES)) == 45


def daily_weather(weather):
    """Shift before rolling; reindex each barangay to actual calendar days."""
    out = []
    for psgc, group in weather.groupby('psgc', sort=True):
        dates = pd.to_datetime(group.date)
        frame = group.set_index(dates).sort_index()
        if frame.index.has_duplicates:
            raise ValueError('Duplicate weather date: ' + psgc)
        frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq='D'))
        f = pd.DataFrame(index=frame.index)
        for model, raw in WEATHER_MAP.items():
            f[model] = pd.to_numeric(frame[raw], errors='raise')
        angle = pd.to_numeric(frame['wind_direction_10m_dominant (°)'], errors='raise')
        f['wind_direction_sin'] = np.sin(np.deg2rad(angle))
        f['wind_direction_cos'] = np.cos(np.deg2rad(angle))
        bases = [('precipitation_mm', 'precipitation', 'sum', 'mm'),
                 ('wind_gust_max_kmh', 'wind_gust_max', 'max', 'kmh'),
                 ('temperature_mean_c', 'temperature_mean', 'mean', 'c'),
                 ('humidity_mean_pct', 'humidity_mean', 'mean', 'pct')]
        for base, prefix, agg, unit in bases:
            f[f'{prefix}_lag1d_{unit}'] = f[base].shift(1)
            for days in [3, 7]:
                stem = prefix + ('_sum' if prefix == 'precipitation' else '')
                f[f'{stem}_prev{days}d_{unit}'] = getattr(f[base].shift(1).rolling(days, min_periods=days), agg)()
        f['psgc'], f['date'] = psgc, f.index.strftime('%Y-%m-%d')
        out.append(f.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def history_features(panel, evidence, flags, holdback_days=0):
    """A fault's availability comes from fault-bearing posts, not earlier non-fault updates."""
    e = evidence[['event_id', 'psgc', 'date', 'published_at_local']].merge(
        flags[['event_id', 'equipment_fault_stated']], on='event_id', how='left', validate='many_to_one')
    if e.equipment_fault_stated.isna().any():
        raise ValueError('Missing equipment annotation; do not treat unknown as zero.')
    e['incident'] = pd.to_datetime(e.date)
    e['publication'] = pd.to_datetime(e.published_at_local, utc=True).dt.tz_convert('Asia/Manila').dt.as_unit('ns')
    all_days = e.groupby(['psgc', 'incident']).publication.min().reset_index()
    fault_days = e[e.equipment_fault_stated.eq(1)].groupby(['psgc', 'incident']).publication.min().reset_index()
    result = []
    for psgc, g in panel.groupby('psgc', sort=True):
        a = all_days[all_days.psgc.eq(psgc)]
        f = fault_days[fault_days.psgc.eq(psgc)]
        adates = a.incident.to_numpy(dtype='datetime64[D]')
        fdates = f.incident.to_numpy(dtype='datetime64[D]')
        apub = a.publication.astype('int64').to_numpy()
        fpub = f.publication.astype('int64').to_numpy()
        for row in g.itertuples():
            date = np.datetime64(row.date, 'D')
            cutoff = (pd.Timestamp(row.prediction_cutoff_local) - pd.Timedelta(days=holdback_days)).value
            known = adates[(adates < date) & (apub <= cutoff)]
            known_fault = fdates[(fdates < date) & (fpub <= cutoff)]
            count = lambda arr, days: int(np.sum(arr >= date - np.timedelta64(days, 'D')))
            n3, n7 = count(known, 3), count(known, 7)
            recency = min(int((date - known.max()) / np.timedelta64(1, 'D')), 7) if len(known) else 7
            eq7, eq14 = count(known_fault, 7), count(known_fault, 14)
            if not (eq7 <= n7 and eq14 <= count(known, 14)):
                raise AssertionError('Equipment days exceed all outage days in the same window.')
            result.append({'psgc': psgc, 'date': row.date, HISTORY[0]: n3, HISTORY[1]: n7,
                           HISTORY[2]: recency, HISTORY[3]: int(n7 == 0),
                           EQUIPMENT[0]: eq7, EQUIPMENT[1]: eq14})
    return pd.DataFrame(result)


def build_features(panel, weather, vegetation, evidence, flags):
    f = panel.merge(daily_weather(weather), on=['psgc', 'date'], how='left', validate='one_to_one')
    f['vegetation_month'] = f.date.str[:7]
    f = f.merge(vegetation[['psgc', 'month'] + VEGETATION].rename(columns={'month': 'vegetation_month'}),
                on=['psgc', 'vegetation_month'], how='left', validate='many_to_one')
    f = f.merge(history_features(panel, evidence, flags), on=['psgc', 'date'],
                how='left', validate='one_to_one')
    d = pd.to_datetime(f.date)
    f['day_of_week'] = d.dt.dayofweek
    f['is_weekend'] = f.day_of_week.ge(5).astype(int)
    annual = 2 * np.pi * (d.dt.dayofyear - 1) / np.where(d.dt.is_leap_year, 366, 365)
    monthly = 2 * np.pi * (d.dt.month - 1) / 12
    f['day_of_year_sin'], f['day_of_year_cos'] = np.sin(annual), np.cos(annual)
    f['month_sin'], f['month_cos'] = np.sin(monthly), np.cos(monthly)
    if f[HISTORY + EQUIPMENT].isna().any().any():
        raise AssertionError('Missing required history.')
    if np.isinf(f[[x for x in FEATURES if x not in ['psgc', 'weather_code']]].to_numpy(dtype=float)).any():
        raise AssertionError('Non-finite model input.')
    return f.sort_values(['date', 'psgc']).reset_index(drop=True)
