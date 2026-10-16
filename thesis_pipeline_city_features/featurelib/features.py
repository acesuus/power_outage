import numpy as np
import pandas as pd
from .common import PROJECT, csv_read, read_json
from citylib.data import GROUPS, HISTORY, FEATURES, FORBIDDEN, feature_description
from monthlib.data import load_inputs

RAIN = ['rain_mean_mm', 'rain_max_mm', 'rainy_barangay_fraction', 'rain_hours_mean',
        'rain_hourly_peak_max_mm'] + [n for n in GROUPS['prior_weather'] if not n.startswith('gust')]
WIND = ['gust_mean_kmh', 'gust_max_kmh'] + [n for n in GROUPS['prior_weather'] if n.startswith('gust')]
THERMO = ['temperature_mean_c', 'humidity_mean_pct']
LOCAL = ['local_rain_veg5_mean', 'local_rain_veg5_max', 'local_gust_veg5_mean', 'local_gust_veg5_max']
DRY = ['rain_after_3dry_fraction', 'rain_after_7dry_fraction']
REPLACED = ['rain_max_x_veg5p90', 'gust_max_x_veg5p90']
LOCAL_FULL = [n for n in FEATURES if n not in REPLACED] + LOCAL
FAMILIES = {
    'history_control': HISTORY,
    'full_control': FEATURES,
    'rainfall': HISTORY + RAIN,
    'wind': HISTORY + WIND,
    'temperature_humidity': HISTORY + THERMO,
    'vegetation': HISTORY + GROUPS['vegetation'],
    'all_weather': HISTORY + GROUPS['target_weather'] + GROUPS['prior_weather'],
    'full_local': LOCAL_FULL,
    'full_dry': FEATURES + DRY,
    'full_local_dry': LOCAL_FULL + DRY,
}
CONTROLS = {'history_control': 'history_calendar', 'full_control': 'full'}
NEW_FEATURES = LOCAL + DRY
for fields in FAMILIES.values():
    assert len(fields) == len(set(fields))
    assert set(GROUPS['equipment_history']) <= set(fields)
    assert set(fields).isdisjoint(FORBIDDEN)


def local_daily(weather, vegetation, rain_threshold=.1):
    """Join exact barangay/month first; never multiply unrelated spatial extrema."""
    assert not weather.duplicated(['psgc', 'date']).any()
    assert not vegetation.duplicated(['psgc', 'month']).any()
    w = weather[['psgc', 'date', 'precipitation_sum (mm)', 'wind_gusts_10m_max (km/h)']].copy()
    w['month'] = w.date.str[:7]
    w = w.merge(vegetation[['psgc', 'month', 'veg_buffer_5m']], on=['psgc', 'month'],
                how='left', validate='many_to_one').sort_values(['psgc', 'date']).reset_index(drop=True)
    assert w[['precipitation_sum (mm)', 'wind_gusts_10m_max (km/h)']].notna().all().all()
    for _, g in w.groupby('psgc'):
        assert pd.to_datetime(g.date).diff().dropna().eq(pd.Timedelta(days=1)).all()
    w['rain_x_veg'] = w['precipitation_sum (mm)'] * w.veg_buffer_5m
    w['gust_x_veg'] = w['wind_gusts_10m_max (km/h)'] * w.veg_buffer_5m
    w['dry'] = w['precipitation_sum (mm)'].lt(rain_threshold).astype(float)
    rainy = w['precipitation_sum (mm)'].ge(rain_threshold)
    for span in [3, 7]:
        # Rolling histories are within a barangay and exclude the target day.
        previous = w.groupby('psgc').dry.transform(lambda s: s.shift(1).rolling(span, min_periods=span).sum())
        w[f'after{span}'] = (rainy & previous.eq(span)).astype(float).where(previous.notna())
    groups = w.groupby('date')
    daily = groups.agg(local_rain_veg5_mean=('rain_x_veg', 'mean'), local_rain_veg5_max=('rain_x_veg', 'max'),
                       local_gust_veg5_mean=('gust_x_veg', 'mean'), local_gust_veg5_max=('gust_x_veg', 'max'),
                       rain_after_3dry_fraction=('after3', 'mean'), rain_after_7dry_fraction=('after7', 'mean'),
                       local_vegetation_observed_count=('veg_buffer_5m', 'count'),
                       spatial_weather_count=('psgc', 'nunique')).reset_index()
    # All 58 sites must contribute to a dry-to-wet denominator; warmup stays missing.
    for span in [3, 7]:
        count = groups[f'after{span}'].count()
        counts = daily.date.map(count)
        daily.loc[counts.ne(daily.spatial_weather_count), f'rain_after_{span}dry_fraction'] = np.nan
    return daily


def build(config):
    original = load_inputs(config)
    city = PROJECT / config['city_run']
    sources = read_json(city / 'config.json')['sources']
    registry = csv_read(city / '02_features/fixed_weather_barangay_registry.csv')
    weather = csv_read(PROJECT / sources['weather'])
    weather = weather[weather.psgc.isin(registry.psgc)].copy()
    vegetation = csv_read(PROJECT / sources['vegetation'])
    vegetation = vegetation[vegetation.psgc.isin(registry.psgc)].copy()
    assert len(registry) == 58 and weather.groupby('date').psgc.nunique().eq(58).all()
    daily = local_daily(weather, vegetation, config['rain_threshold_mm'])
    result = original.merge(daily, on='date', how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result[DRY].dropna().ge(0).all().all() and result[DRY].dropna().le(1).all().all()
    assert np.isfinite(result[NEW_FEATURES].dropna().to_numpy()).all()
    return result, daily


def dictionary():
    rows = []
    definitions = {
        'local_rain_veg5_mean': ('mm x source proxy', 'Mean of same-barangay D rainfall x D-month 5 m vegetation; observed vegetation sites only'),
        'local_rain_veg5_max': ('mm x source proxy', 'Maximum of same-barangay D rainfall x D-month 5 m vegetation; observed sites only'),
        'local_gust_veg5_mean': ('km/h x source proxy', 'Mean of same-barangay D maximum gust x D-month 5 m vegetation; observed sites only'),
        'local_gust_veg5_max': ('km/h x source proxy', 'Maximum of same-barangay D maximum gust x D-month 5 m vegetation; observed sites only'),
        'rain_after_3dry_fraction': ('fraction of 58 sites', 'D rainfall >=0.1 mm following rainfall <0.1 mm on each of D-3 through D-1 at that same site'),
        'rain_after_7dry_fraction': ('fraction of 58 sites', 'D rainfall >=0.1 mm following rainfall <0.1 mm on each of D-7 through D-1 at that same site'),
    }
    for name in FEATURES + NEW_FEATURES:
        if name in definitions:
            unit, description = definitions[name]
            timing = 'D weather and D month' if name in LOCAL else 'D weather plus previous 3/7 days'
        else:
            unit, timing, description = feature_description(name)
        rows.append(dict(variable=name, unit=unit, timing=timing, definition=description,
                         families=';'.join(k for k, v in FAMILIES.items() if name in v)))
    return pd.DataFrame(rows)
