from pathlib import Path
import argparse
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'thesis_pipeline'))
sys.path.insert(0, str(PROJECT / 'thesis_pipeline_v2'))
from lib.common import csv_read, csv_write, json_write, read_json, sha, now, versions


def source(run, name):
    return PROJECT / read_json(run / 'config.json')['sources'][name]


def code_hashes():
    files = list(ROOT.glob('*.py')) + list((ROOT / 'citylib').glob('*.py'))
    files += list((PROJECT / 'thesis_pipeline/lib').glob('*.py'))
    files += [PROJECT / 'thesis_pipeline_v2/v2lib/folds.py']
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)}


def check_inputs(run):
    for path, expected in read_json(run / '01_audit/source_hashes.json').items():
        assert sha(PROJECT / path) == expected, 'Source changed: ' + path
    assert code_hashes() == read_json(run / '01_audit/code_hashes.json'), 'Code changed: use a new run.'


def stage_entry(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to((ROOT / 'runs').resolve())
    status = run / 'status' / f'{number:02}.json'
    assert not status.exists() or read_json(status)['status'] != 'completed'
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check_inputs(run)
    start, clock = now(), time.perf_counter()
    base = {'stage': number, 'title': title, 'started_utc': start}
    json_write(status, {**base, 'status': 'running'})
    try:
        detail = action(run, read_json(run / 'config.json'))
        json_write(status, {**base, 'status': 'completed', 'finished_utc': now(),
            'elapsed_seconds': round(time.perf_counter()-clock, 3), 'detail': detail})
        print(f'City stage {number:02} completed: {title}', flush=True)
    except Exception as error:
        json_write(status, {**base, 'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
