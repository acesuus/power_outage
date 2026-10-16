"""Retrieve rain duration and hourly precipitation; export only daily summaries."""
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
API = 'https://archive-api.open-meteo.com/v1/archive'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def dates(start, end):
    return [(start + timedelta(days=n)).isoformat() for n in range((end-start).days+1)]


def summarize(response, expected):
    """Validate local daily and hourly alignment before retaining a response."""
    assert response.get('timezone') == 'Asia/Manila'
    assert response.get('utc_offset_seconds') == 28800
    daily, hourly = response['daily'], response['hourly']
    assert daily['time'] == expected
    assert response['daily_units']['precipitation_hours'] == 'h'
    assert response['daily_units']['precipitation_sum'] == 'mm'
    assert response['hourly_units']['precipitation'] == 'mm'
    assert len(hourly['time']) == len(expected) * 24
    result = []
    for i, d in enumerate(expected):
        stamps, amounts = hourly['time'][i*24:(i+1)*24], hourly['precipitation'][i*24:(i+1)*24]
        assert stamps == [d + f'T{h:02}:00' for h in range(24)], 'Local hour alignment failed.'
        hours, total = daily['precipitation_hours'][i], daily['precipitation_sum'][i]
        assert hours is not None and total is not None and all(v is not None for v in amounts)
        assert 0 <= hours <= 24 and total >= 0 and all(v >= 0 and math.isfinite(v) for v in amounts)
        summed = sum(amounts)
        # Rounded hourly values can differ slightly from provider daily aggregation.
        assert abs(total-summed) <= 2.5, f'Hourly/day precipitation discrepancy: {d}'
        result.append({'date': d, 'precipitation_hours': hours,
            'precipitation_hourly_max_mm': max(amounts), 'retrieved_precipitation_sum_mm': total,
            'hourly_sum_mm': round(summed, 4), 'hourly_daily_difference_mm': round(summed-total, 4)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/rainfall')
    parser.add_argument('--batch-size', type=int, choices=range(1, 11), default=10)
    parser.add_argument('--pause-seconds', type=float, default=2.0)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(ROOT.resolve())
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    coord_path, reference_path = PROJECT / 'weather_data/barangay_coordinates.csv', PROJECT / 'weather_data/barangay_daily_weather.csv'
    coords, old = read_csv(coord_path), read_csv(reference_path)
    start, end = date.fromisoformat(min(r['date'] for r in old)), date.fromisoformat(max(r['date'] for r in old))
    model = {r['weather_model_selection'] for r in old}
    assert model == {'best_match'}
    reference = {(r['psgc'],r['date']): float(r['precipitation_sum (mm)']) for r in old}
    assert len(reference) == len(old)
    records, manifest = [], []
    jobs = [(max(start,date(year,1,1)),min(end,date(year,12,31)),coords[i:i+args.batch_size])
            for year in range(start.year,end.year+1) for i in range(0,len(coords),args.batch_size)]
    for number, (begin, finish, batch) in enumerate(jobs, 1):
        query = {'latitude': ','.join(r['latitude'] for r in batch), 'longitude': ','.join(r['longitude'] for r in batch),
            'start_date': begin.isoformat(), 'end_date': finish.isoformat(),
            'daily': 'precipitation_hours,precipitation_sum', 'hourly': 'precipitation',
            'timezone': 'Asia/Manila', 'precipitation_unit': 'mm', 'cell_selection': 'land'}
        url = API + '?' + urllib.parse.urlencode(query)
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = cache / (key+'.json')
        if cached.exists():
            envelope = json.loads(cached.read_text(encoding='utf-8'))
            assert envelope['request_url'] == url
        else:
            print(f'Fetching rainfall batch {number}/{len(jobs)}: {begin.year}, {len(batch)} locations', flush=True)
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={'User-Agent':'PuertoPrincesaThesisDailyRainfall/3.0'})
                    with urllib.request.urlopen(request, timeout=120) as connection:
                        raw = connection.read()
                    response = json.loads(raw)
                    break
                except urllib.error.HTTPError as error:
                    message = error.read().decode('utf-8', errors='replace')
                    save_json(out / 'download_error.json', {'utc':datetime.now(timezone.utc).isoformat(),
                        'status':error.code,'message':message,'validated_cached_batches':number-1})
                    if error.code == 429:
                        raise RuntimeError('Provider quota reached; validated responses are cached for a later resume. '+message) from error
                    if error.code not in [500,502,503,504] or attempt == 2:
                        raise RuntimeError(message) from error
                    time.sleep(10)
            envelope = {'request_url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                        'response_sha256':hashlib.sha256(raw).hexdigest(),'response':response}
        responses = envelope['response'] if isinstance(envelope['response'],list) else [envelope['response']]
        assert len(responses) == len(batch)
        expected = dates(begin,finish)
        for index, (coordinate, response) in enumerate(zip(batch,responses)):
            assert response.get('location_id',index) == index
            for row in summarize(response,expected):
                row.update(psgc=coordinate['psgc'],psgc_current=coordinate['psgc_current'],barangay=coordinate['barangay'],
                    requested_latitude=coordinate['latitude'],requested_longitude=coordinate['longitude'],
                    api_latitude=response['latitude'],api_longitude=response['longitude'])
                row['frozen_precipitation_sum_mm'] = reference[(row['psgc'],row['date'])]
                row['retrieved_minus_frozen_mm'] = round(row['retrieved_precipitation_sum_mm']-row['frozen_precipitation_sum_mm'],4)
                records.append(row)
        save_json(cached,envelope)
        manifest.append({k:envelope[k] for k in ['request_url','retrieved_at_utc','response_sha256']})
        print(f'Validated rainfall batch {number}/{len(jobs)}', flush=True)
        time.sleep(args.pause_seconds)
    records.sort(key=lambda r:(r['date'],r['psgc']))
    assert len(records) == len(reference) and len({(r['psgc'],r['date']) for r in records}) == len(reference)
    output = out / 'barangay_rainfall_daily.csv'
    temporary = output.with_suffix('.part')
    with temporary.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output)
    different = [r for r in records if abs(r['retrieved_minus_frozen_mm']) > 0.1]
    save_json(out / 'download_manifest.json', {'complete':True,'version':'3.0','rows':len(records),'barangays':len(coords),
        'start_date':start.isoformat(),'end_date':end.isoformat(),'timezone':'Asia/Manila','model_selection':'best_match',
        'output_sha256':digest(output),'inputs_sha256':{str(p.relative_to(PROJECT)):digest(p) for p in [coord_path,reference_path]},
        'daily_features':['precipitation_hours','precipitation_hourly_max_mm'],
        'hourly_source':'precipitation; reduced to one maximum per local day; no hourly predictor rows',
        'frozen_rainfall_preserved':True,'rows_with_daily_revision_over_0_1mm':len(different),
        'maximum_absolute_daily_revision_mm':max(abs(r['retrieved_minus_frozen_mm']) for r in records),
        'maximum_absolute_hourly_daily_difference_mm':max(abs(r['hourly_daily_difference_mm']) for r in records),
        'requests':manifest})
    print(f'Complete daily rainfall supplement: {len(records)} rows, {len(different)} daily revisions >0.1 mm',flush=True)


if __name__ == '__main__':
    main()
