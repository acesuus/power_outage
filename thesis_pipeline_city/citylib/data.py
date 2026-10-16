"""One calendar-day label, fixed spatial summaries, and publication-aware histories."""
import numpy as np
import pandas as pd
from .common import csv_read
from lib.features import history_features

# Predictor names deliberately carry city aggregation semantics.
GROUPS = {
    'calendar': ['day_of_week', 'day_of_year_sin', 'day_of_year_cos'],
    'outage_history': ['outage_days_prev3d', 'outage_days_prev7d', 'days_since_last_outage_capped7'],
    'equipment_history': ['equipment_fault_days_prev7d', 'equipment_fault_days_prev14d'],
    'target_weather': ['rain_mean_mm', 'rain_max_mm', 'rainy_barangay_fraction',
        'rain_hours_mean', 'rain_hourly_peak_max_mm', 'gust_mean_kmh', 'gust_max_kmh',
        'temperature_mean_c', 'humidity_mean_pct', 'weather_code'],
    'prior_weather': ['rain_mean_lag1d_mm', 'rain_mean_sum_prev3d_mm', 'rain_mean_sum_prev7d_mm',
        'rain_max_lag1d_mm', 'rain_max_sum_prev3d_mm', 'rain_max_sum_prev7d_mm',
        'gust_max_lag1d_kmh', 'gust_max_prev3d_kmh', 'gust_max_prev7d_kmh',
        'wet_days_prev3d', 'wet_days_prev7d'],
    'vegetation': ['veg5_mean', 'veg30_mean', 'veg5_spatial_p90', 'ndvi_barangay_mean'],
    'interactions': ['rain_max_x_veg5p90', 'gust_max_x_veg5p90', 'rain_present_x_equipment14d']
}
FEATURES = [n for fields in GROUPS.values() for n in fields]
HISTORY = GROUPS['calendar'] + GROUPS['outage_history'] + GROUPS['equipment_history']
FAMILIES = {'history_calendar': HISTORY,
    'history_weather': HISTORY + GROUPS['target_weather'] + GROUPS['prior_weather'],
    'full': FEATURES}
assert len(FEATURES) == len(set(FEATURES)) == 36
FORBIDDEN = {'outage_label','supporting_post_count','publication_fallback_post_count',
    'date_conflict_post_count','event_id','raw_text','published_at','barangay_clean','recloser_clean',
    'unlocalized_negative_uncertainty','vegetation_observed_fraction','row_id','date','psgc'}


def normalize_posts(posts):
    p = posts.fillna('').copy()
    for flag in ['is_planned','is_load_shedding','is_outage_record']:
        p[flag] = pd.to_numeric(p[flag], errors='raise')
    assert not p.is_planned.any() and not p.is_load_shedding.any()
    assert p.is_outage_record.eq(1).all() and not p.event_id.duplicated().any()
    pd.to_datetime(p.event_date, format='%Y-%m-%d', errors='raise')
    p['date'] = p.event_date
    p['published_at_local'] = pd.to_datetime(p.published_at, utc=True, errors='raise').dt.tz_convert('Asia/Manila').map(lambda t: t.isoformat())
    return p


def make_labels(posts):
    evidence = posts[['event_id','date','published_at_local','event_date_source','date_status','post_url']].copy()
    evidence['psgc'] = 'PPC'  # Compatibility identifier, not a PSGC code.
    evidence['publication_fallback'] = evidence.date_status.eq('inferred_from_publish_date').astype(int)
    evidence['date_conflict'] = evidence.event_id.eq('ca1e2964ca9e5568').astype(int)
    dates = pd.date_range(posts.date.min(), posts.date.max(), freq='D').strftime('%Y-%m-%d')
    panel = pd.DataFrame({'date':dates, 'psgc':'PPC'})
    counts = evidence.groupby('date').agg(supporting_post_count=('event_id','nunique'),
        publication_fallback_post_count=('publication_fallback','sum'), date_conflict_post_count=('date_conflict','sum')).reset_index()
    panel = panel.merge(counts, on='date', how='left', validate='one_to_one')
    for n in ['supporting_post_count','publication_fallback_post_count','date_conflict_post_count']:
        panel[n] = panel[n].fillna(0).astype(int)
    panel['outage_label'] = panel.supporting_post_count.gt(0).astype(int)
    panel['fallback_only_positive'] = (panel.outage_label.eq(1) & panel.supporting_post_count.eq(panel.publication_fallback_post_count)).astype(int)
    panel['unlocalized_negative_uncertainty'] = 0  # All retained posts localize to city scope.
    panel['prediction_cutoff_local'] = pd.to_datetime(panel.date).map(
        lambda d: (d.tz_localize('Asia/Manila')-pd.Timedelta(days=1)+pd.Timedelta(hours=20)).isoformat())
    panel['row_id'] = np.arange(len(panel), dtype=np.int64)
    return panel, evidence


def weather_registry(weather):
    registry = weather.groupby('psgc')[['barangay','requested_latitude','requested_longitude']].first().reset_index()
    assert registry.psgc.nunique() == len(registry)
    registry['equal_weight'] = 1/len(registry)
    return registry


def aggregate_weather(weather, rainfall, registry, config):
    assert not weather.duplicated(['psgc','date']).any()
    assert not rainfall.duplicated(['psgc','date']).any()
    w = weather[weather.psgc.isin(registry.psgc)].merge(
        rainfall[['psgc','date','precipitation_hours','precipitation_hourly_max_mm']],
        on=['psgc','date'], validate='one_to_one', how='left')
    fields = ['precipitation_sum (mm)','wind_gusts_10m_max (km/h)','temperature_2m_mean (°C)',
        'relative_humidity_2m_mean (%)','weather_code (wmo code)', 'precipitation_hours','precipitation_hourly_max_mm']
    assert w[fields].notna().all().all(), 'Complete weather coverage required.'
    assert w.groupby('date').psgc.nunique().eq(len(registry)).all()
    w['rainy'] = w['precipitation_sum (mm)'].ge(config['rain_threshold_mm']).astype(int)
    g = w.groupby('date')
    f = g.agg(rain_mean_mm=('precipitation_sum (mm)','mean'), rain_max_mm=('precipitation_sum (mm)','max'),
        rainy_barangay_fraction=('rainy','mean'), rain_hours_mean=('precipitation_hours','mean'),
        rain_hourly_peak_max_mm=('precipitation_hourly_max_mm','max'),
        gust_mean_kmh=('wind_gusts_10m_max (km/h)','mean'), gust_max_kmh=('wind_gusts_10m_max (km/h)','max'),
        temperature_mean_c=('temperature_2m_mean (°C)','mean'), humidity_mean_pct=('relative_humidity_2m_mean (%)','mean'))
    f['weather_code'] = g['weather_code (wmo code)'].agg(lambda s: s.mode().min())
    f['weather_barangay_count'] = g.psgc.nunique()
    return f.reset_index().sort_values('date').reset_index(drop=True)


def aggregate_vegetation(vegetation, registry):
    v = vegetation[vegetation.psgc.isin(registry.psgc)].copy()
    assert not v.duplicated(['psgc','month']).any()
    g = v.groupby('month')
    f = g.agg(veg5_mean=('veg_buffer_5m','mean'), veg30_mean=('veg_buffer_30m','mean'),
        veg5_spatial_p90=('veg_buffer_5m',lambda s:s.quantile(.9)), ndvi_barangay_mean=('ndvi_mean_brgy','mean'))
    f['vegetation_observed_fraction'] = g.veg_buffer_5m.count()/len(registry)
    return f.reset_index()


def prior_weather(daily, config):
    f = daily.set_index(pd.to_datetime(daily.date)).sort_index()
    f = f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='D'))
    for raw, stem, unit, operation in [('rain_mean_mm','rain_mean','mm','sum'),
        ('rain_max_mm','rain_max','mm','sum'),('gust_max_kmh','gust_max','kmh','max')]:
        f[f'{stem}_lag1d_{unit}'] = f[raw].shift(1)
        for span in [3,7]:
            prefix = stem+'_sum' if operation=='sum' else stem
            f[f'{prefix}_prev{span}d_{unit}'] = getattr(f[raw].shift(1).rolling(span,min_periods=span),operation)()
    wet = f.rain_max_mm.ge(config['rain_threshold_mm']).astype(float).where(f.rain_max_mm.notna())
    for span in [3,7]:
        f[f'wet_days_prev{span}d'] = wet.shift(1).rolling(span,min_periods=span).sum()
    f['date'] = f.index.strftime('%Y-%m-%d')
    return f.reset_index(drop=True)


def make_features(panel, daily, vegetation, evidence, flags, config):
    f = panel.merge(prior_weather(daily,config), on='date', how='left', validate='one_to_one')
    f['vegetation_month'] = f.date.str[:7]
    f = f.merge(vegetation.rename(columns={'month':'vegetation_month'}), on='vegetation_month', how='left', validate='many_to_one')
    histories = history_features(panel,evidence,flags)
    f = f.merge(histories, on=['psgc','date'], validate='one_to_one')
    d = pd.to_datetime(f.date)
    f['day_of_week'] = d.dt.dayofweek
    angle = 2*np.pi*(d.dt.dayofyear-1)/np.where(d.dt.is_leap_year,366,365)
    f['day_of_year_sin'],f['day_of_year_cos'] = np.sin(angle),np.cos(angle)
    f['rain_max_x_veg5p90'] = f.rain_max_mm*f.veg5_spatial_p90
    f['gust_max_x_veg5p90'] = f.gust_max_kmh*f.veg5_spatial_p90
    f['rain_present_x_equipment14d'] = f.rain_max_mm.ge(config['rain_threshold_mm']).astype(float).where(f.rain_max_mm.notna())*f.equipment_fault_days_prev14d
    assert f.row_id.equals(panel.row_id) and f.outage_label.equals(panel.outage_label)
    assert set(FEATURES).isdisjoint(FORBIDDEN)
    assert f[GROUPS['outage_history']+GROUPS['equipment_history']].notna().all().all()
    return f


def feature_description(name):
    exact = {
        'day_of_week': ('category','D','Monday=0 through Sunday=6; categorical'),
        'day_of_year_sin': ('unitless','D','Sine of annual calendar position; leap-year adjusted'),
        'day_of_year_cos': ('unitless','D','Cosine of annual calendar position; leap-year adjusted'),
        'rain_mean_mm': ('mm','D','Equal mean of daily precipitation across fixed weather barangays; never spatial sum'),
        'rain_max_mm': ('mm','D','Largest daily precipitation at any fixed weather barangay'),
        'rainy_barangay_fraction': ('fraction','D','Fraction of fixed weather barangays with precipitation >=0.1 mm'),
        'rain_hours_mean': ('hours','D','Equal mean of hourly precipitation-duration summaries'),
        'rain_hourly_peak_max_mm': ('mm in one hour','D','Largest daily hourly precipitation peak across fixed weather barangays'),
        'gust_mean_kmh': ('km/h','D','Equal mean of barangay daily maximum gusts'),
        'gust_max_kmh': ('km/h','D','Largest barangay daily maximum gust'),
        'temperature_mean_c': ('degrees C','D','Equal mean of barangay daily mean temperatures'),
        'humidity_mean_pct': ('percent','D','Equal mean of barangay daily mean relative humidity'),
        'weather_code': ('WMO category','D','Modal barangay daily WMO code, smallest numeric category breaks ties; categorical'),
        'veg5_mean': ('source fraction/index','D month','Available-barangay mean of supplied 5 m vegetation buffer proxy'),
        'veg30_mean': ('source fraction/index','D month','Available-barangay mean of supplied 30 m vegetation buffer proxy'),
        'veg5_spatial_p90': ('source fraction/index','D month','90th percentile across available barangay 5 m vegetation proxies; not pixel P90'),
        'ndvi_barangay_mean': ('NDVI','D month','Mean of available barangay-level NDVI means'),
        'days_since_last_outage_capped7': ('days','prior incidents','Days since last known earlier city outage date, capped at 7; 7 if none'),
        'rain_max_x_veg5p90': ('mm x source proxy','D and D month','City maximum rainfall multiplied by spatial P90 vegetation proxy'),
        'gust_max_x_veg5p90': ('km/h x source proxy','D and D month','City maximum gust multiplied by spatial P90 vegetation proxy'),
        'rain_present_x_equipment14d': ('days','D rain and prior incidents','Any-barangay rain >=0.1 mm times known equipment-fault city-days in prior 14 days')
    }
    if name in exact: return exact[name]
    if name.startswith('outage_days_prev'):
        span = int(name.split('prev')[1][:-1]); return ('days','prior incidents',f'Distinct city outage dates in [D-{span},D-1], published by D-1 20:00 PHT')
    if name.startswith('equipment_fault_days_prev'):
        span = int(name.split('prev')[1][:-1]); return ('days','prior incidents',f'Distinct equipment-fault city dates in [D-{span},D-1]; fault-bearing post published by D-1 20:00 PHT')
    if name.startswith('wet_days_prev'):
        span = int(name.split('prev')[1][:-1]); return ('days','historical weather',f'Count of days in [D-{span},D-1] with at least one sampled barangay precipitation >=0.1 mm')
    stem = 'rain_mean' if name.startswith('rain_mean') else 'rain_max' if name.startswith('rain_max') else 'gust_max'
    unit = 'km/h' if stem=='gust_max' else 'mm'
    base = {'rain_mean':'city mean daily rainfall','rain_max':'city spatial maximum daily rainfall','gust_max':'city spatial maximum daily gust'}[stem]
    if '_lag1d_' in name: return (unit,'D-1',f'Previous calendar-day {base}')
    span = int(name.split('prev')[1].split('d')[0])
    action = 'Maximum' if stem=='gust_max' else 'Sum'
    note = '; daily maxima may come from different barangays' if stem=='rain_max' else ''
    return (unit,'historical weather',f'{action} of {base} over [D-{span},D-1]{note}')


def dictionary():
    records = []
    for group, fields in GROUPS.items():
        for name in fields:
            unit,timing,description = feature_description(name)
            records.append({'variable':name,'group':group,'unit':unit,'timing':timing,'definition':description,
                'families':';'.join(k for k,v in FAMILIES.items() if name in v),
                'retrospective_only':int(group in ['target_weather','vegetation','interactions'])})
    return pd.DataFrame(records)
