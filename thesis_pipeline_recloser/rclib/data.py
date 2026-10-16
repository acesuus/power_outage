"""Named-area labels, publication-aware histories and frozen spatial exposure proxies."""
import re
import numpy as np
import pandas as pd
from .common import csv_read, source
from lib.features import (TARGET_WEATHER, PRIOR, VEGETATION, CALENDAR, WEATHER_MAP,
                         history_features)
from v2lib.features import (FEATURES as BASE62, GROUPS as GROUP62, STRESS,
                           additional_histories, wet_history)
from v3lib.features import (FEATURES as BASE79, GROUPS as GROUP79, rain_pattern,
                           specific_fault_history, other_area_history)

SPATIAL_WEATHER = ['precipitation_area_max_mm', 'precipitation_area_range_mm',
    'gust_area_max_kmh', 'gust_area_range_kmh', 'precipitation_hours_area_max',
    'precipitation_hourly_max_area_max_mm']
SPATIAL_VEGETATION = [f'veg_buffer_{width}m_area_max' for width in [5, 15, 30]]
SHARED = [f'other_recloser_outage_days_prev{span}d' for span in [3, 7]]
FEATURES = [n.replace('other_barangay_', 'other_recloser_') for n in BASE79] + SPATIAL_WEATHER + SPATIAL_VEGETATION
GROUPS = {g: [n.replace('other_barangay_', 'other_recloser_') for n in fields]
          for g, fields in GROUP79.items()}
GROUPS.update(spatial_weather=SPATIAL_WEATHER, spatial_vegetation=SPATIAL_VEGETATION)
HISTORY_GROUPS = ['identity', 'calendar', 'outage_history', 'equipment_history',
    'repair_history', 'equipment_category_history', 'insulation_protection_history', 'shared_area_history']
FAMILIES = {'history_calendar': [n for g in HISTORY_GROUPS for n in GROUPS[g]],
    'history_weather': [n for n in FEATURES if n not in VEGETATION + SPATIAL_VEGETATION + GROUPS['weather_vegetation_interactions']],
    'full': FEATURES}
assert len(FEATURES) == len(set(FEATURES)) == 88

# An affected Kalikasan notice describes replacing failed Lacsamana equipment.
# That equipment mention alone does not establish an interruption in Lacsamana's area.
EXCLUDED_ASSIGNMENTS = {('46daf9104d5dfdf2', 'LACSAMANA'):
    'Failed equipment mentioned; explicitly affected service area is Kalikasan, not Lacsamana.'}


def tokens(value):
    return {n.strip() for n in str(value).split(';') if n.strip()}


def normalize_posts(posts):
    p = posts.fillna('').copy()
    for flag in ['is_planned', 'is_load_shedding', 'is_outage_record']:
        p[flag] = pd.to_numeric(p[flag])
    assert not p.is_planned.any() and not p.is_load_shedding.any()
    assert p.is_outage_record.eq(1).all()
    assert not p.event_id.duplicated().any()
    p['date'] = p.event_date
    p['published_at_local'] = pd.to_datetime(p.published_at, utc=True).dt.tz_convert('Asia/Manila').map(lambda t: t.isoformat())
    return p


def make_labels(posts, config):
    records, review = [], []
    for row in posts.itertuples(index=False):
        for name in sorted(tokens(row.recloser_clean) & set(config['reclosers'])):
            key = (row.event_id, name)
            if key in EXCLUDED_ASSIGNMENTS:
                review.append({'event_id': row.event_id, 'date': row.date, 'recloser': name,
                    'reason': EXCLUDED_ASSIGNMENTS[key]})
                continue
            pattern = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I)
            context = 'affected_area' if pattern.search(row.affected_area_raw) else 'title' if pattern.search(row.title) else 'body_restoration_isolation_or_affected_list'
            hits = [re.sub(r'\s+', ' ', row.raw_text[max(0, m.start()-80):m.end()+150])
                    for m in pattern.finditer(row.raw_text)]
            assert hits, 'Assigned recloser lacks raw-text evidence.'
            records.append({'event_id': row.event_id, 'date': row.date, 'psgc': name,
                'recloser': name, 'published_at_local': row.published_at_local,
                'assignment_context': context, 'event_date_source': row.event_date_source,
                'publication_fallback': int(row.date_status == 'inferred_from_publish_date'),
                'date_conflict': int(row.event_id == 'ca1e2964ca9e5568'),
                'post_url': row.post_url, 'evidence': ' || '.join(hits)})
    evidence = pd.DataFrame(records)
    dates = pd.date_range(posts.date.min(), posts.date.max(), freq='D').strftime('%Y-%m-%d')
    panel = pd.MultiIndex.from_product([dates, sorted(config['reclosers'])], names=['date','psgc']).to_frame(index=False)
    counts = evidence.groupby(['date','psgc']).agg(supporting_post_count=('event_id','nunique'),
        publication_fallback_post_count=('publication_fallback','sum'), date_conflict_post_count=('date_conflict','sum')).reset_index()
    panel = panel.merge(counts, on=['date','psgc'], how='left', validate='one_to_one')
    fields = ['supporting_post_count','publication_fallback_post_count','date_conflict_post_count']
    panel[fields] = panel[fields].fillna(0).astype(int)
    panel['outage_label'] = panel.supporting_post_count.gt(0).astype(int)
    no_names = posts[posts.recloser_clean.map(lambda x: not tokens(x))].groupby('date').size()
    panel['unassigned_recloser_posts_on_date'] = panel.date.map(no_names).fillna(0).astype(int)
    panel['unlocalized_negative_uncertainty'] = (panel.outage_label.eq(0) & panel.unassigned_recloser_posts_on_date.gt(0)).astype(int)
    for row in review:
        take = panel.date.eq(row['date']) & panel.psgc.eq(row['recloser']) & panel.outage_label.eq(0)
        panel.loc[take, 'unlocalized_negative_uncertainty'] = 1
    panel['prediction_cutoff_local'] = pd.to_datetime(panel.date).map(
        lambda d: (d.tz_localize('Asia/Manila')-pd.Timedelta(days=1)+pd.Timedelta(hours=20)).isoformat())
    panel['recloser'] = panel.psgc
    panel['row_id'] = np.arange(len(panel), dtype=np.int64)
    return panel, evidence, pd.DataFrame(review)


def learn_proxy_mapping(posts, barangay_mapping, config):
    end, cutoff = config['mapping_incident_end'], pd.Timestamp(config['mapping_publication_asof'])
    eligible = posts[(posts.date <= end) & (pd.to_datetime(posts.published_at_local, utc=True) <= cutoff.tz_convert('UTC'))]
    names = barangay_mapping.set_index('outage_location_name').psgc.to_dict()
    lookup = barangay_mapping.drop_duplicates('psgc').set_index('psgc').barangay.to_dict()
    records, provenance = [], []
    recent_start = (pd.Timestamp(end)-pd.Timedelta(days=config['mapping_recent_days']-1)).strftime('%Y-%m-%d')
    for name in config['reclosers']:
        sub = eligible[eligible.recloser_clean.map(lambda x: tokens(x) == {name}) &
            eligible.affected_area_raw.str.contains(r'(?<!\w)'+re.escape(name)+r'\s+recloser\b', case=False, regex=True)]
        recent = sub[sub.date >= recent_start]
        chosen = recent if len(recent) else sub
        linked = []
        for row in chosen.itertuples(index=False):
            for b in tokens(row.barangay_clean):
                if b in names:
                    linked.append({'recloser':name, 'psgc':names[b], 'event_id':row.event_id,
                        'date':row.date, 'published_at_local':row.published_at_local,
                        'barangay_source':row.barangay_source, 'affected_area':row.affected_area_raw})
        assert linked, 'No pre-development spatial proxy for ' + name
        support = pd.DataFrame(linked).groupby('psgc').agg(
            supporting_incident_dates=('date','nunique'), first_date=('date','min'), last_date=('date','max')).reset_index()
        count = len(support)
        for row in support.itertuples(index=False):
            records.append({'recloser':name, 'psgc':row.psgc, 'barangay':lookup[row.psgc],
                'equal_weight':1/count, 'supporting_incident_dates':row.supporting_incident_dates,
                'first_support_date':row.first_date, 'last_support_date':row.last_date,
                'older_history_fallback':int(not len(recent)), 'proxy_only':1})
        provenance += linked
    return pd.DataFrame(records), pd.DataFrame(provenance)


def aggregate_daily(weather, rainfall, mapping):
    w = weather[['psgc','date'] + list(WEATHER_MAP.values()) + ['wind_direction_10m_dominant (°)']].copy()
    for name, raw in WEATHER_MAP.items():
        w[name] = pd.to_numeric(w[raw], errors='raise')
    angle = np.deg2rad(pd.to_numeric(w['wind_direction_10m_dominant (°)']))
    w['wind_direction_sin'], w['wind_direction_cos'] = np.sin(angle), np.cos(angle)
    w = w.merge(rainfall[['psgc','date','precipitation_hours','precipitation_hourly_max_mm']],
        on=['psgc','date'], validate='one_to_one')
    rows = []
    numeric = [n for n in TARGET_WEATHER if n != 'weather_code'] + ['precipitation_hours','precipitation_hourly_max_mm']
    for recloser, links in mapping.groupby('recloser'):
        sub = w[w.psgc.isin(links.psgc)].copy()
        assert sub.groupby('date').size().eq(len(links)).all()
        grouped = sub.groupby('date')
        # Equal barangay weights; require complete mapped coverage for weather.
        f = grouped[numeric].mean().where(grouped[numeric].count().eq(len(links)))
        f['weather_code'] = grouped.weather_code.agg(lambda s: s.mode().min() if s.notna().all() else np.nan)
        for raw, maximum, spread in [('precipitation_mm','precipitation_area_max_mm','precipitation_area_range_mm'),
                                     ('wind_gust_max_kmh','gust_area_max_kmh','gust_area_range_kmh')]:
            f[maximum] = grouped[raw].max().where(grouped[raw].count().eq(len(links)))
            f[spread] = (grouped[raw].max()-grouped[raw].min()).where(grouped[raw].count().eq(len(links)))
        f['precipitation_hours_area_max'] = grouped.precipitation_hours.max()
        f['precipitation_hourly_max_area_max_mm'] = grouped.precipitation_hourly_max_mm.max()
        f['psgc'] = recloser
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def aggregate_vegetation(vegetation, mapping):
    rows = []
    for recloser, links in mapping.groupby('recloser'):
        sub = vegetation[vegetation.psgc.isin(links.psgc)].copy()
        grouped = sub.groupby('month')
        f = grouped[VEGETATION].mean()
        # Keep entirely unobserved months missing; available-barangay means are documented.
        for width in [5,15,30]:
            field = f'veg_buffer_{width}m'
            f[field+'_area_max'] = grouped[field].max()
        f['vegetation_observed_barangay_fraction'] = grouped.veg_buffer_5m.count()/len(links)
        f['psgc'] = recloser
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def make_features(panel, daily, vegetation, evidence, flags, config):
    daily = daily.sort_values(['psgc','date']).copy()
    blocks = []
    bases = [('precipitation_mm','precipitation','sum','mm'),('wind_gust_max_kmh','wind_gust_max','max','kmh'),
        ('temperature_mean_c','temperature_mean','mean','c'),('humidity_mean_pct','humidity_mean','mean','pct')]
    for name, g in daily.groupby('psgc'):
        g = g.set_index(pd.to_datetime(g.date)).reindex(pd.date_range(g.date.min(),g.date.max(),freq='D'))
        for raw, stem, agg, unit in bases:
            g[f'{stem}_lag1d_{unit}'] = g[raw].shift(1)
            for span in [3,7]:
                prefix = stem+'_sum' if stem == 'precipitation' else stem
                g[f'{prefix}_prev{span}d_{unit}'] = getattr(g[raw].shift(1).rolling(span,min_periods=span),agg)()
        g['date'],g['psgc'] = g.index.strftime('%Y-%m-%d'),name
        blocks.append(g.reset_index(drop=True))
    f = panel.merge(pd.concat(blocks,ignore_index=True),on=['psgc','date'],how='left',validate='one_to_one')
    f['vegetation_month'] = f.date.str[:7]
    f = f.merge(vegetation.rename(columns={'month':'vegetation_month'}),on=['psgc','vegetation_month'],how='left',validate='many_to_one')
    for addition in [history_features(panel,evidence,flags),additional_histories(panel,evidence,flags),
        wet_history(f,config['wet_day_threshold_mm']),rain_pattern(f,config['rain_presence_threshold_mm']),
        specific_fault_history(panel,evidence,flags),other_area_history(panel,evidence).rename(
            columns={f'other_barangay_outage_days_prev{s}d':f'other_recloser_outage_days_prev{s}d' for s in [3,7]})]:
        f = f.merge(addition,on=['psgc','date'],validate='one_to_one')
    d = pd.to_datetime(f.date)
    f['day_of_week'], f['is_weekend'] = d.dt.dayofweek,d.dt.dayofweek.ge(5).astype(int)
    annual = 2*np.pi*(d.dt.dayofyear-1)/np.where(d.dt.is_leap_year,366,365)
    month = 2*np.pi*(d.dt.month-1)/12
    f['day_of_year_sin'],f['day_of_year_cos'] = np.sin(annual),np.cos(annual)
    f['month_sin'],f['month_cos'] = np.sin(month),np.cos(month)
    f['gust_x_veg5m'],f['precipitation_x_veg5m'] = f.wind_gust_max_kmh*f.veg_buffer_5m,f.precipitation_mm*f.veg_buffer_5m
    for prefix,spans in [('outage',[3,7]),('equipment_fault',[7,14])]:
        for span in spans:
            field = f'{prefix}_days_prev{span}d'
            f['rain_present_x_'+field] = f.rain_present*f[field]
    f['precipitation_x_equipment_fault_days_prev14d'] = f.precipitation_mm*f.equipment_fault_days_prev14d
    for n in STRESS: f[n] = np.nan
    f = f.sort_values(['date','psgc']).reset_index(drop=True)
    assert f.row_id.equals(panel.row_id) and f.outage_label.equals(panel.outage_label)
    assert set(FEATURES) <= set(f.columns)
    return f
