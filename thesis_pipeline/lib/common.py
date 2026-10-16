"""Run provenance, source checks and explicit stage dependencies."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
PROJECT = PIPELINE.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                    encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    return pd.read_csv(path, encoding='utf-8-sig', dtype={'psgc': str, 'event_id': str})


def csv_write(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def source(run, name):
    return PROJECT / read_json(run / 'config.json')['sources'][name]


def assert_sources(run):
    recorded = read_json(run / '01_audit/source_hashes.json')
    for relative, expected in recorded.items():
        if sha(PROJECT / relative) != expected:
            raise RuntimeError('Input changed since audit: ' + relative + '. Start a new run.')


def stage_entry(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to((PIPELINE / 'runs').resolve()):
        raise ValueError('Run outputs must stay inside thesis_pipeline/runs.')
    config = read_json(run / 'config.json')
    status = run / 'status' / f'{number:02}.json'
    if status.exists() and read_json(status)['status'] == 'completed':
        raise RuntimeError('Stage already completed. Resume via run_pipeline.py; do not repeat final testing.')
    if number > 1:
        previous = read_json(run / 'status' / f'{number - 1:02}.json')
        if previous['status'] != 'completed':
            raise RuntimeError('Previous stage has not completed.')
        assert_sources(run)
    started, clock = now(), time.perf_counter()
    code_files = sorted(PIPELINE.glob('*.py')) + sorted((PIPELINE / 'lib').glob('*.py'))
    executed_code = {str(p.relative_to(PIPELINE)): sha(p) for p in code_files}
    json_write(status, {'stage': number, 'title': title, 'status': 'running', 'started_utc': started,
                        'code_sha256': executed_code})
    try:
        detail = action(run, config) or {}
        json_write(status, {'stage': number, 'title': title, 'status': 'completed',
                           'started_utc': started, 'finished_utc': now(),
                           'elapsed_seconds': round(time.perf_counter() - clock, 3), 'detail': detail,
                           'code_sha256': executed_code})
        print(f'Stage {number:02} completed: {title}', flush=True)
    except Exception as exc:
        json_write(status, {'stage': number, 'title': title, 'status': 'failed',
                           'started_utc': started, 'failed_utc': now(), 'error': str(exc),
                           'traceback': traceback.format_exc(), 'code_sha256': executed_code})
        raise


def versions():
    import importlib.metadata as md
    names = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'xgboost', 'matplotlib', 'joblib']
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': {name: md.version(name) for name in names}}
