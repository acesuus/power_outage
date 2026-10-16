from itertools import product
import numpy as np
import pandas as pd
from .common import PROJECT, csv_read, read_json
from citylib.data import GROUPS, FEATURES as COMPACT, HISTORY, aggregate_weather, prior_weather
from featurelib.features import LOCAL, DRY, REPLACED, RAIN, WIND, THERMO, local_daily
from rclib.data import FEATURES as LEGACY
from monthlib.data import role_definition, origins
from v2lib.folds import get_cohorts

IDENTITY = ['recloser_id']
FAMILIES = {
    'history': IDENTITY + HISTORY,
    'full': IDENTITY + COMPACT,
    'rainfall': IDENTITY + HISTORY + RAIN,
    'wind': IDENTITY + HISTORY + WIND,
    'temperature_humidity': IDENTITY + HISTORY + THERMO,
    'vegetation': IDENTITY + HISTORY + GROUPS['vegetation'],
    'all_weather': IDENTITY + HISTORY + GROUPS['target_weather'] + GROUPS['prior_weather'],
    'full_local': IDENTITY + [n for n in COMPACT if n not in REPLACED] + LOCAL,
    'full_dry': IDENTITY + COMPACT + DRY,
    'full_local_dry': IDENTITY + [n for n in COMPACT if n not in REPLACED] + LOCAL + DRY,
    'legacy_full': LEGACY,
}
FORBIDDEN = {'outage_label', 'row_id', 'date', 'raw_text', 'supporting_post_count',
             'publication_fallback_post_count', 'date_conflict_post_count', 'event_id', 'post_url',
             'unlocalized_negative_uncertainty', 'unassigned_recloser_posts_on_date',
             'first_support_utc', 'vegetation_observed_fraction'}
for fields in FAMILIES.values():
    assert len(fields) == len(set(fields)) and set(fields).isdisjoint(FORBIDDEN)
    assert set(GROUPS['equipment_history']) <= set(fields)


def frozen_specs():
    broad = [dict(kind=kind, family=family, window='all',
                  base_id=kind+'__'+family+'__all') for kind, family in product(
                  ['logistic', 'spline_logistic', 'xgboost', 'catboost', 'random_forest'],
                  ['history', 'all_weather', 'full', 'legacy_full'])]
    return broad + [dict(kind='xgboost', family=family, window='all', base_id='xgboost__'+family+'__all')
                    for family in FAMILIES if family not in ['history', 'all_weather', 'full', 'legacy_full']]


def monthly_specs():
    broad = [dict(kind=kind, family=family, window=window, base_id=kind+'__'+family+'__'+window)
             for kind, family, window in product(['logistic', 'xgboost'], ['history', 'full'], ['12m', '24m', 'all'])]
    feature = [dict(kind='xgboost', family=family, window='12m', base_id='xgboost__'+family+'__12m')
               for family in FAMILIES if family not in ['history', 'full', 'legacy_full']]
    legacy = [dict(kind='xgboost', family='legacy_full', window=w, base_id='xgboost__legacy_full__'+w)
              for w in ['12m', '24m', 'all']]
    result = broad + feature + legacy
    assert len({r['base_id'] for r in result}) == len(result) == 23
    return result


def build(config):
    reference = PROJECT / config['reference_run']
    original = csv_read(reference / '02_features/recloser_feature_panel.csv')
    reference_config = read_json(reference / 'config.json')
    sources = reference_config['sources']
    weather, rain, vegetation = [csv_read(PROJECT / sources[n]) for n in ['weather', 'rainfall', 'vegetation']]
    mapping = csv_read(reference / '02_features/recloser_barangay_proxy_mapping.csv')
    extra = []
    spatial_audit = []
    for recloser, registry in mapping.groupby('recloser'):
        w = weather[weather.psgc.isin(registry.psgc)].copy()
        v = vegetation[vegetation.psgc.isin(registry.psgc)].copy()
        daily = aggregate_weather(w, rain, registry, config)
        fields = prior_weather(daily, config)
        vg = v.groupby('month').agg(veg5_mean=('veg_buffer_5m', 'mean'), veg30_mean=('veg_buffer_30m', 'mean'),
                veg5_spatial_p90=('veg_buffer_5m', lambda s: s.quantile(.9)), ndvi_barangay_mean=('ndvi_mean_brgy', 'mean'))
        fields['month'] = fields.date.str[:7]
        fields = fields.merge(vg, on='month', how='left', validate='many_to_one')
        local = local_daily(w, v, config['rain_threshold_mm'])
        fields = fields.merge(local, on='date', validate='one_to_one')
        fields['rain_max_x_veg5p90'] = fields.rain_max_mm * fields.veg5_spatial_p90
        fields['gust_max_x_veg5p90'] = fields.gust_max_kmh * fields.veg5_spatial_p90
        fields['psgc'] = recloser
        local['recloser'] = recloser
        spatial_audit.append(local)
        extra.append(fields)
    new = pd.concat(extra, ignore_index=True)
    # Keep the established recloser wet-day definition (mapped mean rain >=0.1 mm).
    # The city helper's any-site wet-day definition is different and is not substituted.
    new = new.drop(columns=['wet_days_prev3d', 'wet_days_prev7d'])
    keys = ['psgc', 'date']
    shared = [n for n in new if n in original and n not in keys]
    joined = original[keys + shared].merge(new[keys + shared], on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    for name in shared:
        if pd.api.types.is_numeric_dtype(joined[name+'_old']):
            np.testing.assert_allclose(joined[name+'_old'], joined[name+'_new'], rtol=1e-12, atol=1e-12, equal_nan=True)
    new = new.drop(columns=shared + ['month'])
    result = original.merge(new, on=keys, how='left', validate='one_to_one')
    result['recloser_id'] = result.psgc
    result['rain_present_x_equipment14d'] = result.rain_max_mm.ge(.1).astype(float) * result.equipment_fault_days_prev14d
    evidence = csv_read(reference / '01_audit/recloser_outage_evidence.csv')
    earliest = evidence.assign(pub=pd.to_datetime(evidence.published_at_local, utc=True)).groupby(['psgc', 'date']).pub.min()
    index = pd.MultiIndex.from_frame(result[keys])
    result['first_support_utc'] = pd.to_datetime(earliest.reindex(index).to_numpy(), utc=True)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert len(result) == 13473 and int(result.outage_label.sum()) == 418
    assert set(n for v in FAMILIES.values() for n in v) <= set(result.columns)
    return result, pd.concat(spatial_audit, ignore_index=True)


def load(run):
    frame = csv_read(run / '02_features/recloser_feature_panel.csv')
    frame['first_support_utc'] = pd.to_datetime(frame.first_support_utc, utc=True)
    assert np.array_equal(frame.row_id.to_numpy(), np.arange(len(frame)))
    return frame


def known(frame, asof):
    return frame.outage_label.eq(0) | frame.first_support_utc.le(pd.Timestamp(asof).tz_convert('UTC'))


def eligible(frame, asof):
    return known(frame, asof) & frame.unlocalized_negative_uncertainty.eq(0)


def monthly_roles(frame, origin, window, config):
    definition = role_definition(origin, window, config)
    valid = eligible(frame, definition['asof'])
    roles = {role: frame[frame.date.between(definition[role]['start'], definition[role]['end']) & valid].copy()
             for role in ['fit', 'tune', 'refit', 'calibration']}
    for role in ['fit', 'tune', 'refit', 'calibration']:
        assert roles[role].outage_label.nunique() == 2, origin+' '+window+' '+role
        assert roles[role].date.max() < origin
    assert set(roles['fit'].row_id).isdisjoint(roles['tune'].row_id)
    assert (pd.Timestamp(origin)-pd.Timestamp(roles['refit'].date.max())).days >= 15
    return roles, definition


def target_month(frame, origin, config, phase, full_pool=False, sensitivity=False):
    end = (pd.Timestamp(origin)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    if phase == 'evaluation': end = min(end, config['test_end'])
    elif not full_pool: end = min(end, config['development_end'])
    take = frame.date.between(origin, end)
    if not full_pool:
        asof = config['test_truth_asof'] if phase == 'evaluation' else config['selection_asof']
        take &= known(frame, asof)
        if not sensitivity: take &= frame.unlocalized_negative_uncertainty.eq(0)
    return frame[take].copy()


def frozen_roles(frame, origin, config):
    return get_cohorts(frame, PROJECT / config['reference_run'] / '03_cohorts' / (origin+'.npz'))


def calibration_rows(pool, calibration, origin, minimum=45):
    assert pool.row_id.is_unique
    result = calibration[['row_id', 'date', 'recloser', 'outage_label']].merge(
        pool[['row_id', 'origin', 'raw_probability']], on='row_id', validate='one_to_one')
    assert len(result) == len(calibration) and len(result) >= minimum
    assert result.origin.max() < origin and result.date.max() < origin
    return result
