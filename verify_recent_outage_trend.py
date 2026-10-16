"""Verify the featured trend model and its four chronological development fits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'thesis_pipeline_recloser_suite'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plot',action='store_true',help='Regenerate the calibration reliability figure')
    args=parser.parse_args()
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
    from suitelib.common import read_json, csv_read
    from suitelib.data import load, frozen_roles
    from suitelib.modeling import score
    folder=ROOT/'models/recent_outage_trend'
    provenance=read_json(folder/'provenance.json')
    for relative,item in provenance['copied_files'].items():
        with (ROOT/relative).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256'],relative
    original=ROOT/'thesis_pipeline_recloser_suite/runs/recloser_suite_first'
    config=read_json(original/'config.json')
    frame=load(original).copy()
    frame['outage_rate_recent3d_minus_previous4d']=frame.outage_days_prev3d/3-(frame.outage_days_prev7d-frame.outage_days_prev3d)/4
    fields=read_json(folder/'features.json')['fields']
    worst=0.0
    for origin in config['development_origins']+[config['test_start']]:
        final=origin==config['test_start']
        current=folder if final else folder/'development'/origin
        bundle=joblib.load(current/'model.joblib')
        assert bundle['transformer'].fields==fields and len(fields)==10
        roles=frozen_roles(frame,origin,config)
        saved=pd.read_csv(current/'predictions.csv',float_precision='round_trip')
        pred,_=score(bundle,roles['outer'],origin,config)
        np.testing.assert_array_equal(pred.row_id,saved.row_id)
        for name in pred:
            if name.endswith('_warning'):
                np.testing.assert_array_equal(pred[name],saved[name])
            elif name.endswith('_probability'):
                error=float(np.max(abs(pred[name].to_numpy()-saved[name].to_numpy())))
                assert error<1e-12
                worst=max(worst,error)
        if final:
            prefix=bundle['spec']['base_id']+'__raw'
            y=pred.outage_label.to_numpy()
            p=pred[prefix+'_probability'].to_numpy()
            w=pred[prefix+'_fpr20_warning'].to_numpy()
            tn,fp,fn,tp=map(int,confusion_matrix(y,w,labels=[0,1]).ravel())
            assert (len(y),int(y.sum()),tn,fp,fn,tp)==(2098,57,1146,895,19,38)
            result=dict(status='passed',model_bundles_reloaded=5,maximum_probability_reload_error=worst,
                rows=len(y),positives=int(y.sum()),auc_roc=float(roc_auc_score(y,p)),
                f1_score=2*tp/(2*tp+fp+fn),precision=tp/(tp+fp),recall=tp/(tp+fn),
                brier_score=float(brier_score_loss(y,p)),tp=tp,fp=fp,fn=fn,tn=tn,
                raw_fpr20_cutoff=bundle['warnings']['raw']['fpr20']['threshold'])
    metrics=pd.read_csv(ROOT/'reports/recent_outage_trend/metrics.csv',float_precision='round_trip')
    final_bundle=joblib.load(folder/'model.joblib')
    pred,records=score(final_bundle,frozen_roles(frame,config['test_start'],config)['outer'],'2026_frozen',config)
    for r in records:
        old=metrics[(metrics.model_id==r['model_id'])&(metrics.policy==r['policy'])].iloc[0]
        for name in ['roc_auc','f1','precision','recall','brier_score']:
            assert np.isclose(old[name],r[name],rtol=0,atol=1e-12)
    result['metric_rows_verified']=len(records)
    if args.plot:
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl_cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        bins=pd.read_csv(ROOT/'reports/recent_outage_trend/reliability_bins.csv',float_precision='round_trip')
        fig,ax=plt.subplots(figsize=(7,5.5),constrained_layout=True)
        limit=min(1,max(bins.mean_probability.max(),bins.observed_rate.max(),.05)*1.18)
        ax.plot([0,limit],[0,limit],'--',color='#777777',linewidth=1,label='Perfect calibration')
        for mode,label,color in [('raw','Raw','#2166ac'),('monotone_sigmoid','Earlier-fitted sigmoid','#b35806')]:
            group=bins[bins['mode']==mode]
            assert group.rows.sum()==2098 and group.positives.sum()==57
            ax.plot(group.mean_probability,group.observed_rate,'o-',label=label,color=color,markersize=4)
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Mean predicted outage probability',
               ylabel='Observed outage rate',title='Frozen XGBoost with recent-outage trend\n2,098 rows, 57 positives; up to 10 quantile bins per mode')
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.legend(loc='upper left',fontsize=9)
        fig.savefig(ROOT/'reports/recent_outage_trend/calibration_reliability.png',dpi=180)
        plt.close(fig)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
