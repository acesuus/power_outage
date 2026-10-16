"""Read-only comparisons of barangay daily weather and the original PPC series."""
from pathlib import Path
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
new_path = ROOT / 'weather_data/barangay_daily_weather.csv'
old_path = ROOT / 'daily_weather_data.csv'
hashes_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (new_path, old_path)}
old = pd.read_csv(old_path, skiprows=3, encoding='utf-8-sig').rename(columns={'time': 'date'})
new = pd.read_csv(new_path, encoding='utf-8-sig')
features = list(new.columns[12:])
assert len(features) == 14 and all(c in old.columns for c in features)
assert not old.date.duplicated().any()
assert not new.duplicated(['psgc_current', 'date']).any()
dates = sorted(set(old.date) & set(new.date))
baseline = old.set_index('date').loc[dates, features].to_numpy(float)
names = sorted(new.barangay.unique())
arrays = []
for name in names:
    block = new[new.barangay == name].set_index('date')
    assert set(dates).issubset(block.index)
    arrays.append(block.loc[dates, features].to_numpy(float))
values = np.stack(arrays)
assert np.isfinite(values).all() and np.isfinite(baseline).all()
exact = values == baseline[None, :, :]
groups = defaultdict(list)
for name, data in zip(names, values):
    groups[data.tobytes()].append(name)

metrics = {}
for j, col in enumerate(features):
    x, y = values[:, :, j], baseline[:, j]
    diff = np.abs(x - y[None, :])
    circular = col.startswith('wind_direction')
    categorical = col.startswith('weather_code')
    if circular:
        diff = np.minimum(diff, 360 - diff)
    mae = diff.mean(axis=1)
    ranges = np.ptp(x, axis=0)
    entry = {
        'exact_match_percent_vs_ppc': float(100 * exact[:, :, j].mean()),
        'distinct_barangay_series': len({a.tobytes() for a in x}),
        'days_all_barangays_identical_percent': float(100 * (ranges == 0).mean()),
        'barangays_entire_feature_identical_to_ppc': [names[k] for k in range(len(names)) if exact[k, :, j].all()],
    }
    if not categorical:
        entry.update({
            'mean_absolute_difference_vs_ppc': float(diff.mean()),
            'mae_barangay_min': float(mae.min()),
            'mae_barangay_max': float(mae.max()),
            'max_mae_barangays': [names[k] for k in range(len(names)) if mae[k] == mae.max()],
            'absolute_difference_p95_vs_ppc': float(np.quantile(diff, .95)),
            'absolute_difference_max_vs_ppc': float(diff.max()),
        })
    if not categorical and not circular:
        corr = np.array([np.corrcoef(a, y)[0, 1] for a in x])
        entry.update({
            'correlation_mean_vs_ppc': float(corr.mean()),
            'correlation_min_vs_ppc': float(corr.min()),
            'correlation_max_vs_ppc': float(corr.max()),
            'mean_daily_spatial_range': float(ranges.mean()),
            'p95_daily_spatial_range': float(np.quantile(ranges, .95)),
            'max_daily_spatial_range': float(ranges.max()),
        })
    metrics[col] = entry

per_barangay = []
for i, name in enumerate(names):
    row = {'barangay': name, 'days_all_14_match_ppc': int(exact[i].all(axis=1).sum()),
           'all_feature_values_match_percent': float(exact[i].mean() * 100)}
    for field in ('temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)',
                  'precipitation_sum (mm)', 'wind_gusts_10m_max (km/h)'):
        j = features.index(field)
        row[field + '_mae'] = float(np.abs(values[i, :, j] - baseline[:, j]).mean())
    per_barangay.append(row)

examples = {}
for field in ('precipitation_sum (mm)', 'wind_gusts_10m_max (km/h)', 'temperature_2m_mean (°C)'):
    j = features.index(field)
    x = values[:, :, j]
    k = int(np.argmax(np.ptp(x, axis=0)))
    examples[field] = {'date': dates[k], 'ppc': float(baseline[k, j]),
                       'lowest_barangay': names[int(np.argmin(x[:, k]))], 'lowest': float(x[:, k].min()),
                       'highest_barangay': names[int(np.argmax(x[:, k]))], 'highest': float(x[:, k].max())}

yearly = {}
for year in sorted({d[:4] for d in dates}):
    mask = np.array([d.startswith(year) for d in dates])
    yearly[year] = {'days': int(mask.sum()), 'exact_values_percent': float(exact[:, mask].mean() * 100)}
    for field in ('precipitation_sum (mm)', 'temperature_2m_mean (°C)', 'wind_gusts_10m_max (km/h)'):
        j = features.index(field)
        yearly[year][field + '_mae'] = float(np.abs(values[:, mask, j] - baseline[mask, j]).mean())

result = {
    'period': [dates[0], dates[-1]], 'shared_days': len(dates), 'barangays': len(names),
    'compared_barangay_days': len(dates) * len(names),
    'distinct_full_series': len(groups),
    'identical_groups': [g for g in groups.values() if len(g) > 1],
    'all_14_exact_match_to_ppc_entire_period': [names[k] for k in range(len(names)) if exact[k].all()],
    'all_values_exact_match_percent_vs_ppc': float(exact.mean() * 100),
    'metrics': metrics, 'per_barangay': per_barangay,
    'largest_daily_spatial_difference_examples': examples, 'yearly': yearly,
    'precipitation_equals_rain_new': bool((new['precipitation_sum (mm)'] == new['rain_sum (mm)']).all()),
    'precipitation_equals_rain_original': bool((old['precipitation_sum (mm)'] == old['rain_sum (mm)']).all()),
    'source_sha256': hashes_before,
}
assert hashes_before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (new_path, old_path)}
print(json.dumps(result, ensure_ascii=False, indent=2))
