from pathlib import Path
from datetime import datetime, timezone
import argparse
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl_cache'))
for folder in ['thesis_pipeline_city_features', 'thesis_pipeline_recloser', 'thesis_pipeline_city']:
    sys.path.insert(0, str(PROJECT / folder))
from featurelib.common import csv_read, csv_write, read_json, json_write, sha
from rclib.common import versions


def now(): return datetime.now(timezone.utc).isoformat()


def code_hashes():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'suitelib').glob('*.py'))
    paths += [ROOT / 'README.md', ROOT / 'METHODOLOGY.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def check(run):
    assert code_hashes() == read_json(run / '01_audit/code_hashes.json'), 'Use a new run after code changes.'
    assert sha(run / 'config.json') == read_json(run / '01_audit/config_hash.json')['sha256']
    for relative, expected in read_json(run / '01_audit/input_hashes.json').items():
        assert sha(PROJECT / relative) == expected, relative


def stage(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to(ROOT / 'runs')
    marker = run / 'status' / f'{number:02}.json'
    assert not marker.exists() or read_json(marker)['status'] != 'completed'
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check(run)
    started, clock = now(), time.perf_counter()
    json_write(marker, dict(status='running', title=title, started_utc=started))
    try:
        detail = action(run, read_json(run / 'config.json'))
        json_write(marker, dict(status='completed', title=title, started_utc=started, finished_utc=now(),
                               elapsed_seconds=round(time.perf_counter()-clock, 3), detail=detail))
        print(f'Recloser suite stage {number:02} completed: {title}', flush=True)
    except Exception as error:
        json_write(marker, dict(status='failed', error=str(error), traceback=traceback.format_exc()))
        raise
