from pathlib import Path
import argparse
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT / 'thesis_pipeline_city'))
from citylib.common import csv_read,csv_write,json_write,read_json,sha,now,check_inputs


def city_run(config):return PROJECT / config['city_run']
def xgb_run(config):return PROJECT / config['xgboost_run']


def code_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(list(ROOT.glob('*.py'))+list((ROOT / 'monthlib').glob('*.py')))}


def check(run):
    config=read_json(run / 'config.json');check_inputs(city_run(config))
    if (run / '01_audit/code_hashes.json').exists():
        assert code_hashes()==read_json(run / '01_audit/code_hashes.json'),'Monthly code changed: use a new run.'
        assert sha(run / 'config.json')==read_json(run / '01_audit/config_hash.json')['sha256']


def stage_entry(number,title,action):
    parser=argparse.ArgumentParser(description=title);parser.add_argument('--run-dir',type=Path,required=True)
    run=parser.parse_args().run_dir.resolve();assert run.is_relative_to(ROOT / 'runs')
    path=run / 'status' / f'{number:02}.json'
    assert not path.exists() or read_json(path)['status']!='completed'
    if number>1:assert read_json(run / 'status' / f'{number-1:02}.json')['status']=='completed'
    check(run);started=now();clock=time.perf_counter()
    json_write(path,{'stage':number,'title':title,'status':'running','started_utc':started})
    try:
        detail=action(run,read_json(run / 'config.json'))
        json_write(path,{'stage':number,'title':title,'status':'completed','started_utc':started,
            'finished_utc':now(),'elapsed_seconds':round(time.perf_counter()-clock,3),'detail':detail})
        print(f'Monthly stage {number:02} completed: {title}',flush=True)
    except Exception as error:
        json_write(path,{'stage':number,'title':title,'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise
