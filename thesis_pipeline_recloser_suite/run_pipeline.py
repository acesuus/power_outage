import argparse
import shutil
import subprocess
import sys
from suitelib.common import ROOT, read_json

STEPS = ['01_audit_sources.py', '02_features_and_cohorts.py', '03_frozen_development.py',
         '04_select_and_fit_frozen.py', '05_monthly_development.py', '06_select_and_evaluate_monthly.py',
         '07_evaluate_frozen_and_compare.py', '08_verify_and_report.py']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    assert args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id)
    run = ROOT / 'runs' / args.run_id
    if run.exists():
        assert args.resume, 'Existing run: pass --resume or choose a new ID.'
        assert not (run / '08_report/COMPLETION.json').exists(), 'Completed runs are frozen.'
    else:
        run.mkdir(parents=True)
        shutil.copy2(ROOT / 'config.json', run / 'config.json')
    for number, filename in enumerate(STEPS[:args.until], 1):
        marker = run / 'status' / f'{number:02}.json'
        if marker.exists() and read_json(marker)['status'] == 'completed': continue
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / filename), '--run-dir', str(run)], check=True)
    (ROOT / 'LATEST_RUN.txt').write_text(str(run), encoding='utf-8')


if __name__ == '__main__': main()
