from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'thesis_pipeline_city_monthly'))
from monthlib.common import csv_read, csv_write, json_write, read_json, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def own_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(
        list(ROOT.glob('*.py')) + list((ROOT / 'featurelib').glob('*.py')) +
        [ROOT / 'README.md', ROOT / 'METHODOLOGY.md'])}


def check(run):
    assert own_hashes() == read_json(run / '01_audit/code_hashes.json'), 'Code changed; start a new run.'
    assert sha(run / 'config.json') == read_json(run / '01_audit/config_hash.json')['sha256']
    for path, expected in read_json(run / '01_audit/input_hashes.json').items():
        assert sha(PROJECT / path) == expected, path


def verify_parent(parent):
    manifest = read_json(parent / '06_report/artifact_manifest.json')
    for relative, record in manifest.items():
        assert sha(parent / relative) == record['sha256'], relative
    return manifest


def stage(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to(ROOT / 'runs')
    status = run / 'status' / f'{number:02}.json'
    assert not status.exists() or read_json(status)['status'] != 'completed'
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check(run)
    start, clock = now(), time.perf_counter()
    json_write(status, {'status': 'running', 'started_utc': start, 'title': title})
    try:
        detail = action(run, read_json(run / 'config.json'))
        json_write(status, {'status': 'completed', 'started_utc': start, 'finished_utc': now(),
                           'elapsed_seconds': round(time.perf_counter()-clock, 3), 'detail': detail})
        print(f'Feature stage {number:02} completed: {title}', flush=True)
    except Exception as error:
        json_write(status, {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
