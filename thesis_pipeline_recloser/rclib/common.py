from pathlib import Path
import argparse
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
for dependency in ['thesis_pipeline', 'thesis_pipeline_v2', 'thesis_pipeline_v3']:
    sys.path.insert(0, str(PROJECT / dependency))
from lib.common import csv_read, csv_write, json_write, read_json, sha, now, versions


def source(run, key):
    return PROJECT / read_json(run / 'config.json')['sources'][key]


def code_hashes():
    files = list(ROOT.glob('*.py')) + list((ROOT / 'rclib').glob('*.py'))
    for folder in ['thesis_pipeline/lib', 'thesis_pipeline_v2/v2lib', 'thesis_pipeline_v3/v3lib']:
        files += list((PROJECT / folder).glob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)}


def check_sources(run):
    for name, expected in read_json(run / '01_audit/source_hashes.json').items():
        assert sha(PROJECT / name) == expected, 'Input changed: ' + name


def stage_entry(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to((ROOT / 'runs').resolve())
    status = run / 'status' / f'{number:02}.json'
    assert not status.exists() or read_json(status)['status'] != 'completed'
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check_sources(run)
    start, clock = now(), time.perf_counter()
    base = {'stage': number, 'title': title, 'started_utc': start, 'code_sha256': code_hashes()}
    json_write(status, {**base, 'status': 'running'})
    try:
        detail = action(run, read_json(run / 'config.json')) or {}
        json_write(status, {**base, 'status': 'completed', 'finished_utc': now(),
            'elapsed_seconds': round(time.perf_counter()-clock, 3), 'detail': detail})
        print(f'Recloser stage {number:02} completed: {title}', flush=True)
    except Exception as error:
        json_write(status, {**base, 'status': 'failed', 'error': str(error),
            'traceback': traceback.format_exc(), 'failed_utc': now()})
        raise
