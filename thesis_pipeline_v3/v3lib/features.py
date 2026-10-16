"""Rainfall exposure and source-aware vulnerability candidates; 79 total inputs."""
import numpy as np
import pandas as pd
from v2lib.features import FEATURES as BASE62, GROUPS as BASE_GROUPS

EXPOSURE = ['precipitation_hours','precipitation_hourly_max_mm']
PATTERN = ['rain_present','consecutive_dry_days_capped7','rain_onset_after3drydays','precipitation_change_lag1d_mm']
VULNERABILITY = ['rain_present_x_outage_days_prev3d','rain_present_x_outage_days_prev7d',
    'rain_present_x_equipment_fault_days_prev7d','rain_present_x_equipment_fault_days_prev14d',
    'precipitation_x_equipment_fault_days_prev14d']
FAULTS = [f'{kind}_fault_days_prev{span}d' for kind in ['insulator','fuse_cutout'] for span in [7,14]]
NETWORK = ['other_barangay_outage_days_prev3d','other_barangay_outage_days_prev7d']
ADDED = EXPOSURE + PATTERN + VULNERABILITY + FAULTS + NETWORK
FEATURES = list(BASE62) + ADDED
GROUPS = {**BASE_GROUPS,'rainfall_exposure':EXPOSURE,'rainfall_pattern':PATTERN,
    'rain_vulnerability_interactions':VULNERABILITY,'insulation_protection_history':FAULTS,'shared_area_history':NETWORK}
assert len(ADDED) == 17 and len(FEATURES) == len(set(FEATURES)) == 79
history_groups = ['identity','calendar','outage_history','equipment_history','repair_history',
    'equipment_category_history','insulation_protection_history','shared_area_history']
HISTORY = [n for group in history_groups for n in GROUPS[group]]
NON_VEGETATION = [n for n in FEATURES if n not in GROUPS['vegetation'] + GROUPS['weather_vegetation_interactions']]
FAMILIES = {'base62':list(BASE62),'rain79':FEATURES,'history_calendar':HISTORY,'history_weather':NON_VEGETATION}


def rain_pattern(frame, threshold):
    result = []
    for psgc, g in frame.groupby('psgc',sort=True):
        rain = g.set_index(pd.to_datetime(g.date)).precipitation_mm.sort_index()
        rain = rain.reindex(pd.date_range(rain.index.min(),rain.index.max(),freq='D'))
        wet = rain.gt(threshold).astype(float).where(rain.notna())
        dry = 1-wet
        def streak(values):
            count = 0
            for value in values[::-1]:
                if value != 1:
                    break
                count += 1
            return count
        preceding3 = dry.shift(1).rolling(3,min_periods=3).sum()
        onset = (wet.eq(1) & preceding3.eq(3)).astype(float).where(wet.notna() & preceding3.notna())
        result.append(pd.DataFrame({'psgc':psgc,'date':rain.index.strftime('%Y-%m-%d'),
            'rain_present':wet.to_numpy(),
            'consecutive_dry_days_capped7':dry.shift(1).rolling(7,min_periods=7).apply(streak,raw=True).to_numpy(),
            'rain_onset_after3drydays':onset.to_numpy(),
            'precipitation_change_lag1d_mm':(rain-rain.shift(1)).to_numpy()}))
    return pd.concat(result,ignore_index=True)


def specific_fault_history(panel, evidence, flags):
    e = evidence[['event_id','psgc','date','published_at_local']].merge(
        flags[['event_id','cause_insulator','cause_fuse_cutout']],on='event_id',how='left',validate='many_to_one')
    assert e[['cause_insulator','cause_fuse_cutout']].notna().all().all()
    e['incident'] = pd.to_datetime(e.date)
    e['publication'] = pd.to_datetime(e.published_at_local,utc=True).dt.as_unit('ns')
    by_type = {}
    for kind in ['insulator','fuse_cutout']:
        days = e[e['cause_'+kind].eq(1)].groupby(['psgc','incident']).publication.min().reset_index()
        by_type[kind] = {p:(g.incident.to_numpy(dtype='datetime64[D]'),g.publication.astype('int64').to_numpy())
                         for p,g in days.groupby('psgc')}
    result = []
    for row in panel[['psgc','date','prediction_cutoff_local']].itertuples(index=False):
        D, cutoff = np.datetime64(row.date,'D'),pd.Timestamp(row.prediction_cutoff_local).value
        r = {'psgc':row.psgc,'date':row.date}
        for kind in by_type:
            incident,pub = by_type[kind].get(row.psgc,(np.array([],dtype='datetime64[D]'),np.array([],dtype=np.int64)))
            prior = incident[(incident < D) & (pub <= cutoff)]
            for span in [7,14]:
                r[f'{kind}_fault_days_prev{span}d'] = int((prior >= D-np.timedelta64(span,'D')).sum())
        result.append(r)
    return pd.DataFrame(result)


def other_area_history(panel,evidence):
    e = evidence.assign(publication=pd.to_datetime(evidence.published_at_local,utc=True).dt.as_unit('ns'))
    e = e[e.psgc.isin(panel.psgc.unique())].groupby(['psgc','date']).publication.min().reset_index()
    incident = pd.to_datetime(e.date).to_numpy(dtype='datetime64[D]')
    publication = e.publication.astype('int64').to_numpy()
    areas = e.psgc.to_numpy()
    result = []
    for d,g in panel.groupby('date',sort=True):
        cutoff = g.prediction_cutoff_local.unique()
        assert len(cutoff) == 1
        D = np.datetime64(d,'D')
        available = (incident < D) & (publication <= pd.Timestamp(cutoff[0]).value)
        counts = {}
        for span in [3,7]:
            relevant = areas[available & (incident >= D-np.timedelta64(span,'D'))]
            counts[span] = (len(relevant),pd.Series(relevant).value_counts().to_dict())
        for p in g.psgc:
            r = {'date':d,'psgc':p}
            for span in [3,7]:
                total,own = counts[span]
                r[f'other_barangay_outage_days_prev{span}d'] = total-own.get(p,0)
            result.append(r)
    return pd.DataFrame(result)


def build_features(base,rainfall,evidence,flags,config):
    f = base.merge(rainfall[['psgc','date']+EXPOSURE],on=['psgc','date'],how='left',validate='one_to_one')
    assert f[EXPOSURE].notna().all().all(), 'Rainfall supplement must cover the complete frozen panel.'
    for addition in [rain_pattern(base,config['rain_presence_threshold_mm']),
                     specific_fault_history(base,evidence,flags),other_area_history(base,evidence)]:
        f = f.merge(addition,on=['psgc','date'],how='left',validate='one_to_one')
    for prefix in ['outage','equipment_fault']:
        for span in ([3,7] if prefix == 'outage' else [7,14]):
            field = f'{prefix}_days_prev{span}d'
            f['rain_present_x_'+field] = f.rain_present * f[field]
    f['precipitation_x_equipment_fault_days_prev14d'] = f.precipitation_mm * f.equipment_fault_days_prev14d
    f = f.sort_values(['date','psgc']).reset_index(drop=True)
    assert f.row_id.equals(base.row_id) and f.outage_label.equals(base.outage_label)
    return f
