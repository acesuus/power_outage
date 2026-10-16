"""62 registered candidates; weather reference ratios are fitted per cohort."""
import numpy as np
import pandas as pd
from lib.features import FEATURES as BASE_FEATURES, GROUPS as BASE_GROUPS

REPAIR = ['repair_notice_days_prev7d', 'repair_notice_days_prev14d']
CATEGORY_MAP = {
    'line_damage': ['cause_line_conductor_cable'],
    'jumper_fault': ['cause_jumper'],
    'transformer_fault': ['cause_transformer'],
    'pole_crossarm_fault': ['cause_pole', 'cause_crossarm']
}
CATEGORY = [f'{name}_days_prev{days}d' for name in CATEGORY_MAP for days in [7, 14]]
WET = ['wet_days_prev3d', 'wet_days_prev7d', 'consecutive_wet_days_capped7']
STRESS = ['precipitation_vs_fit_p90', 'gust_vs_fit_p90']
INTERACTIONS = ['gust_x_veg5m', 'precipitation_x_veg5m']
ADDED = REPAIR + CATEGORY + WET + STRESS + INTERACTIONS
FEATURES = list(BASE_FEATURES) + ADDED
GROUPS = {**BASE_GROUPS, 'repair_history': REPAIR, 'equipment_category_history': CATEGORY,
          'wet_weather_history': WET, 'weather_stress': STRESS, 'weather_vegetation_interactions': INTERACTIONS}
assert len(ADDED) == 17 and len(FEATURES) == len(set(FEATURES)) == 62


def additional_histories(panel, evidence, flags):
    flag_cols = ['event_id', 'repair_only', 'repair_only_categories'] + list({c for cols in CATEGORY_MAP.values() for c in cols})
    merged = evidence[['event_id', 'psgc', 'date', 'published_at_local']].merge(
        flags[flag_cols], on='event_id', how='left', validate='many_to_one')
    if merged.repair_only.isna().any():
        raise ValueError('Missing equipment annotations cannot be zero-filled.')
    merged['repair_notice'] = (merged.repair_only.eq(1) |
                                merged.repair_only_categories.fillna('').str.strip().ne('')).astype(int)
    indicators = {'repair_notice': REPAIR}
    for name, cols in CATEGORY_MAP.items():
        merged[name] = merged[cols].eq(1).any(axis=1).astype(int)
        indicators[name] = [f'{name}_days_prev{days}d' for days in [7, 14]]
    merged['incident'] = pd.to_datetime(merged.date)
    merged['publication'] = pd.to_datetime(merged.published_at_local, utc=True).dt.as_unit('ns')
    by_type = {}
    for indicator in indicators:
        days = merged[merged[indicator].eq(1)].groupby(['psgc', 'incident']).publication.min().reset_index()
        by_type[indicator] = {psgc: (g.incident.to_numpy(dtype='datetime64[D]'), g.publication.astype('int64').to_numpy())
                              for psgc, g in days.groupby('psgc')}
    results = []
    for row in panel[['psgc', 'date', 'prediction_cutoff_local']].itertuples(index=False):
        date, cutoff = np.datetime64(row.date, 'D'), pd.Timestamp(row.prediction_cutoff_local).value
        result = {'psgc': row.psgc, 'date': row.date}
        for indicator, names in indicators.items():
            incident, publication = by_type[indicator].get(row.psgc, (np.array([], dtype='datetime64[D]'), np.array([], dtype=np.int64)))
            known = incident[(incident < date) & (publication <= cutoff)]
            for span, field in zip([7, 14], names):
                result[field] = int((known >= date - np.timedelta64(span, 'D')).sum())
        results.append(result)
    return pd.DataFrame(results)


def wet_history(frame, threshold):
    out = []
    for psgc, g in frame.groupby('psgc', sort=True):
        rain = g.set_index(pd.to_datetime(g.date)).precipitation_mm.sort_index()
        rain = rain.reindex(pd.date_range(rain.index.min(), rain.index.max(), freq='D'))
        wet = rain.ge(threshold).astype(float).where(rain.notna())
        f = pd.DataFrame({'date': rain.index.strftime('%Y-%m-%d'), 'psgc': psgc})
        for span in [3, 7]:
            f[f'wet_days_prev{span}d'] = wet.shift(1).rolling(span, min_periods=span).sum().to_numpy()
        def streak(values):
            count = 0
            for value in values[::-1]:
                if value != 1:
                    break
                count += 1
            return count
        f['consecutive_wet_days_capped7'] = wet.shift(1).rolling(7, min_periods=7).apply(streak, raw=True).to_numpy()
        out.append(f)
    return pd.concat(out, ignore_index=True)


def build_static(base, evidence, flags, config):
    f = base.merge(additional_histories(base, evidence, flags), on=['psgc', 'date'], validate='one_to_one')
    f = f.merge(wet_history(base, config['wet_day_threshold_mm']), on=['psgc', 'date'], validate='one_to_one')
    f['gust_x_veg5m'] = f.wind_gust_max_kmh * f.veg_buffer_5m
    f['precipitation_x_veg5m'] = f.precipitation_mm * f.veg_buffer_5m
    # These are intentionally undefined in the global table: each fitting cohort owns its reference.
    for name in STRESS:
        f[name] = np.nan
    f = f.sort_values(['date', 'psgc']).reset_index(drop=True)
    f['row_id'] = np.arange(len(f), dtype=np.int64)
    assert not f.duplicated(['date', 'psgc']).any()
    assert f[CATEGORY + REPAIR].notna().all().all()
    return f
