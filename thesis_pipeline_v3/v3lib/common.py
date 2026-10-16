from pathlib import Path
import argparse
import time
import traceback
from lib.common import csv_read, csv_write, json_write, read_json, sha, now, versions

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def source(run, key):
    return PROJECT / read_json(run / 'config.json')['sources'][key]


def code_hashes():
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'v3lib').glob('*.py'))
    files += sorted((PROJECT / 'thesis_pipeline/lib').glob('*.py'))
    files += sorted((PROJECT / 'thesis_pipeline_v2/v2lib').glob('*.py'))
    return {str(p.relative_to(PROJECT)): sha(p) for p in files}


def check_sources(run):
    for path, expected in read_json(run / '01_audit/source_hashes.json').items():
        assert sha(PROJECT / path) == expected, 'Source changed: ' + path


def stage_entry(number, title, action):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    assert run.is_relative_to((ROOT / 'runs').resolve())
    status = run / 'status' / f'{number:02}.json'
    if status.exists() and read_json(status)['status'] == 'completed':
        raise RuntimeError('Stage completed; use the coordinator or a new run.')
    if number > 1:
        assert read_json(run / 'status' / f'{number-1:02}.json')['status'] == 'completed'
        check_sources(run)
    config, start, clock, code = read_json(run / 'config.json'), now(), time.perf_counter(), code_hashes()
    base = {'stage':number,'title':title,'started_utc':start,'code_sha256':code}
    json_write(status,{**base,'status':'running'})
    try:
        result = action(run,config) or {}
        json_write(status,{**base,'status':'completed','finished_utc':now(),
            'elapsed_seconds':round(time.perf_counter()-clock,3),'detail':result})
        print(f'V3 stage {number:02} completed: {title}',flush=True)
    except Exception as error:
        json_write(status,{**base,'status':'failed','failed_utc':now(),'error':str(error),'traceback':traceback.format_exc()})
        raise
