"""Stage provenance and version-2-only outputs."""
from pathlib import Path
import argparse
import shutil
import time
import traceback

from lib.common import csv_read, csv_write, json_write, read_json, sha, now, versions

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
V1 = PROJECT / 'thesis_pipeline'


def source(run, key):
    return PROJECT / read_json(run / 'config.json')['sources'][key]


def code_hashes():
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'v2lib').glob('*.py'))
    files += sorted((V1 / 'lib').glob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def check_sources(run):
    for path, expected in read_json(run / '01_audit/source_hashes.json').items():
        assert sha(PROJECT / path) == expected, 'Input changed since V2 audit: ' + path


def stage_entry(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to((ROOT / 'runs').resolve())
    status = run / 'status' / f'{number:02}.json'
    if status.exists() and read_json(status)['status'] == 'completed':
        raise RuntimeError('Stage completed; use coordinator resume or a new run.')
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check_sources(run)
    config, started, clock = read_json(run / 'config.json'), now(), time.perf_counter()
    code = code_hashes()
    json_write(status, {'stage': number, 'title': title, 'status': 'running', 'started_utc': started, 'code_sha256': code})
    try:
        report = action(run, config) or {}
        json_write(status, {'stage': number, 'title': title, 'status': 'completed', 'started_utc': started,
                            'finished_utc': now(), 'elapsed_seconds': round(time.perf_counter()-clock, 3),
                            'detail': report, 'code_sha256': code})
        print(f'V2 stage {number:02} completed: {title}', flush=True)
    except Exception as exc:
        json_write(status, {'stage': number, 'title': title, 'status': 'failed', 'started_utc': started,
                            'failed_utc': now(), 'error': str(exc), 'traceback': traceback.format_exc(), 'code_sha256': code})
        raise
