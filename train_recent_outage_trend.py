"""Refit the featured frozen trend candidate using the original earlier cohorts."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'thesis_pipeline_recloser_suite'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path,help='New directory; existing outputs are never overwritten')
    args=parser.parse_args()
    out=args.output.resolve()
    assert not out.exists(),'Choose a new output directory.'
    import joblib
    from suitelib.common import read_json,json_write,csv_write
    from suitelib.data import load,frozen_roles,FAMILIES
    from suitelib.modeling import fit_model,attach,score,raw_predict
    from suitelib.stages import complete
    source=ROOT/'thesis_pipeline_recloser_suite/runs/recloser_suite_first'
    config=read_json(source/'config.json')
    frame=load(source).copy()
    frame['outage_rate_recent3d_minus_previous4d']=frame.outage_days_prev3d/3-(frame.outage_days_prev7d-frame.outage_days_prev3d)/4
    FAMILIES['targeted_outage_trend']=read_json(ROOT/'models/recent_outage_trend/features.json')['fields']
    spec=dict(kind='xgboost',family='targeted_outage_trend',window='all',base_id='xgboost__targeted_outage_trend__all')
    out.mkdir(parents=True)
    json_write(out/'config.json',config)
    for origin in config['development_origins']+[config['test_start']]:
        folder=out/('final' if origin==config['test_start'] else 'development/'+origin)
        roles=frozen_roles(frame,origin,config)
        bundle=fit_model(roles,config,spec,folder,'frozen')
        attach(bundle,roles['calibrate'],config)
        pred,records=score(bundle,roles['outer'],origin,config)
        joblib.dump(bundle,folder/'model.joblib')
        csv_write(pred,folder/'predictions.csv')
        json_write(folder/'scores.json',records)
        csv_write(roles['calibrate'][['row_id','date','recloser','outage_label']].assign(
            raw_probability=raw_predict(bundle,roles['calibrate'])),folder/'calibration.csv')
        json_write(folder/'calibration_and_warnings.json',dict(calibrator=bundle['calibrator'].summary(),warnings=bundle['warnings']))
        complete(folder)
    print('Completed frozen trend refit: '+str(out))

if __name__=='__main__':main()
