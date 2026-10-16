"""Focused extraction of already completed monthly XGBoost/history experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/recloser_suite_first'
MONTHLY=RUN/'06_monthly_evaluation'
OUT=ROOT/'analysis/recloser_suite_first/monthly_xgboost_history'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load(path): return pd.read_csv(path,float_precision='round_trip')
def keep(frame): return frame.kind.eq('xgboost') & frame.family.eq('history')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scores=load(MONTHLY/'overall_metrics.csv')
    focused=scores[keep(scores)].copy()
    assert len(focused)==30
    monthly=load(MONTHLY/'monthly_metrics.csv')
    monthly=monthly[keep(monthly)].copy()
    assert len(monthly)==270
    thresholds=load(MONTHLY/'monthly_thresholds.csv')
    names={f'xgboost__history__{w}' for w in ['12m','24m','all']}
    thresholds=thresholds[thresholds.base_id.isin(names)].copy()
    development=load(MONTHLY/'ranked_development.csv')
    development=development[keep(development)].copy()
    per_unit=load(MONTHLY/'per_recloser_metrics.csv')
    per_unit=per_unit[per_unit.model_id.isin(focused.model_id)].copy()
    fitting=load(MONTHLY/'fitting_summary.csv')
    fitting=fitting[keep(fitting)].copy()
    frozen=load(RUN/'07_frozen_evaluation/overall_metrics.csv')
    frozen=frozen[keep(frozen)].copy()
    sources=[MONTHLY/n for n in ['overall_metrics.csv','monthly_metrics.csv','monthly_thresholds.csv',
        'ranked_development.csv','per_recloser_metrics.csv','fitting_summary.csv','predictions.csv']]
    sources+=[RUN/'07_frozen_evaluation/overall_metrics.csv',RUN/'02_features/families.json']
    models=sorted((MONTHLY/'models').glob('2026-*/xgboost__history__*/model.joblib'))
    assert len(models)==27
    sources+=models
    manifest=json.loads((RUN/'08_report/artifact_manifest.json').read_text())
    before={str(p.relative_to(RUN)):sha(p) for p in sources}
    for relative,digest in before.items(): assert manifest[relative]['sha256']==digest,relative
    pred=load(MONTHLY/'predictions.csv')
    assert len(pred)==2098 and pred.outage_label.sum()==57
    for r in focused.itertuples():
        p=pred[r.model_id+'_probability'];w=pred[r.model_id+'_'+r.policy+'_warning']
        np.testing.assert_array_equal(confusion_matrix(pred.outage_label,w,labels=[0,1]).ravel(),[r.tn,r.fp,r.fn,r.tp])
        assert np.isclose(average_precision_score(pred.outage_label,p),r.average_precision)
        assert np.isclose(roc_auc_score(pred.outage_label,p),r.roc_auc)
        blocks=monthly[(monthly.model_id==r.model_id)&(monthly.policy==r.policy)]
        np.testing.assert_array_equal(blocks[['tn','fp','fn','tp']].sum(),[r.tn,r.fp,r.fn,r.tp])
    for r in monthly.itertuples():
        block=pred[pred.origin==r.origin]
        np.testing.assert_array_equal(confusion_matrix(block.outage_label,
            block[r.model_id+'_'+r.policy+'_warning'],labels=[0,1]).ravel(),[r.tn,r.fp,r.fn,r.tp])
    for w in ['12m','24m','all']:
        for policy in ['fpr20','fpr10','f1','balanced_accuracy']:
            np.testing.assert_array_equal(pred[f'xgboost__history__{w}__raw_{policy}_warning'],
                                          pred[f'xgboost__history__{w}__intercept_adjusted_{policy}_warning'])
    for name,frame in [('all_window_and_warning_results.csv',focused),('monthly_results.csv',monthly),
            ('monthly_cutoffs.csv',thresholds),('earlier_development_results.csv',development),
            ('per_recloser_results.csv',per_unit),('fitting_dates.csv',fitting),('frozen_history_reference.csv',frozen)]:
        frame.to_csv(OUT/name,index=False)
    fields=json.loads((RUN/'02_features/families.json').read_text())['history']
    assert len(fields)==9
    text=['# Monthly retraining: XGBoost with history inputs','',
          'These experiments were already completed within the broad recloser suite. This focused review '
          'extracts their saved results and independently recounts them. No model was refitted and no '
          'threshold or window was selected from the examined 2026 results.','',
          '## Same evaluation population','',
          'Nine reclosers, 2,098 eligible recloser-days, 57 recorded positive recloser-days on 41 dates, '
          'January 1–September 21, 2026. Each warning concerns one recloser on one day.','',
          '## Primary comparison: raw scores and earlier-learned FPR20 rules','',
          '| Training procedure | Detected /57 | False warnings | Missed | Recall | Precision | Actual FPR | AP |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    reference=frozen[(frozen['mode']=='raw')&(frozen.policy=='fpr20')].iloc[0]
    selected=focused[(focused['mode']=='raw')&(focused.policy=='fpr20')].set_index('window').loc[['12m','24m','all']]
    records=[('Frozen, expanding history',reference)]+[
        ('Monthly, '+{'12m':'12-month window','24m':'24-month window','all':'expanding history'}[w],r)
        for w,r in selected.iterrows()]
    for label,r in records:
        text.append(f'| {label} | {int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {r.recall:.1%} | '
                    f'{r.precision:.1%} | {r.false_positive_rate:.1%} | {r.average_precision:.4f} |')
    text+=['','Monthly retraining reduces warnings under these earlier FPR20 rules, but also misses many '
           'more positives. These are observed operating points, not a comparison at matched future warning '
           'burden. FPR20 constrains earlier calibration, not the future false-positive rate.','',
           '## Different warning rules for the 12-month model','',
           '| Rule learned earlier | Detected /57 | False warnings | Recall | Precision |',
           '|---|---:|---:|---:|---:|']
    part=focused[(focused.window=='12m')&(focused['mode']=='raw')].set_index('policy')
    for policy in ['fpr20','fpr10','balanced_accuracy','f1','fixed050']:
        r=part.loc[policy];precision=f'{r.precision:.1%}' if r.tp+r.fp else 'undefined'
        text.append(f'| {policy} | {int(r.tp)} | {int(r.fp)} | {r.recall:.1%} | {precision} |')
    text+=['','The 12-month model can detect 42/57 under the earlier balanced-accuracy rule, but that '
           'produces 1,255 false warnings. It therefore does not demonstrate keeping the frozen model’s '
           'high detection count while reducing false warnings. All three monthly windows issue no warnings '
           'at a raw fixed 0.50 cutoff.','',
           '## Training windows and inputs','',
           '- **12 months:** refit each month on the recent eligible 12-month history.',
           '- **24 months:** refit each month on the recent eligible 24-month history.',
           '- **Expanding:** refit each month on all eligible history from September 30, 2022.',
           '- Each monthly fit uses earlier outcomes available before the target month and a 14-day '
           'incident-maturity buffer. Earlier completed 2026 months may enter later monthly fits after '
           'availability/maturity checks. Current/future target-month labels never enter fitting or calibration.',
           '- Cutoffs are learned anew each month from genuinely earlier out-of-sample prediction pools. '
           'They are not a single 0.30 cutoff and need not equal the frozen cutoff.',
           '- The history family has nine raw inputs, listed below. It uses no weather or vegetation. '
           'Training-window length and the daily history-feature lookback are separate choices.','']
    text += ['- `'+field+'`' for field in fields]
    text+=['','The outage lookbacks remain 3/7 days and equipment lookbacks 7/14 days. A 12-month '
           'training window does not change those lookbacks into 12-month feature accumulations.','',
           '## Monthly FPR20 cutoffs, raw score scale','',
           '| Target month | 12-month history | 24-month history | Expanding history |',
           '|---|---:|---:|---:|']
    cuts=thresholds[(thresholds['mode']=='raw')&(thresholds.policy=='fpr20')]
    for origin,group in cuts.groupby('origin'):
        values=group.set_index('base_id').threshold
        labels=[]
        for w in ['12m','24m','all']:
            value=values[f'xgboost__history__{w}']
            labels.append(f'{value:.6f}' if value<=1 else 'No-warning rule')
        text.append('| '+origin[:7]+' | '+' | '.join(labels)+' |')
    text+=['','## Interpretation and thesis reporting','',
           'The current evidence does not show that monthly XGBoost/history solves the false-warning '
           'problem. Under FPR20 the monthly 24-month window detects the most of the three examined '
           'monthly windows (17), while 12 months has higher AP (0.0362). Those are different criteria; '
           'no 2026 winner replaces the registered main procedure.','',
           'Within the three raw XGBoost/history windows, earlier 2025 FPR20 development recall was '
           '18/89 for 24 months versus 17/89 for 12 months and expanding history. The suite-wide '
           'monthly selection remains logistic/full/expanding with intercept adjustment; this focused '
           'comparison does not replace it.','',
           'All 2026 results remain exploratory because this period was examined during earlier '
           'experiments. No-record rows are not verified uninterrupted power; primary eligibility uses '
           'a retrospective reporting mask. The later source audit did not change these 57 evaluation '
           'positives, and these original classifiers were not retrained on the audited source copy.','',
           '## Navigation','',
           '- [Every window, score scale and warning rule](all_window_and_warning_results.csv)',
           '- [Month-by-month results](monthly_results.csv)',
           '- [Actual monthly cutoffs](monthly_cutoffs.csv)',
           '- [Earlier development results](earlier_development_results.csv)',
           '- [Per-recloser results](per_recloser_results.csv)',
           '- [Fitting end dates](fitting_dates.csv)',
           '- [Verification](verification.json)',
           '- [Complete training methodology](../../../METHODOLOGY.md)',
           '- [January 2026 12-month model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__12m/model.joblib)',
           '- [January 2026 24-month model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__24m/model.joblib)',
           '- [January 2026 expanding model](../../../runs/recloser_suite_first/06_monthly_evaluation/models/2026-01-01/xgboost__history__all/model.joblib)','',
           'Later month-specific models are stored under the corresponding target-month folder in '
           '`runs/recloser_suite_first/06_monthly_evaluation/models/`. There is a model per month and '
           'training window, not one fixed monthly model for the whole evaluation period.','']
    (OUT/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    assert before=={str(p.relative_to(RUN)):sha(p) for p in sources}
    verification=dict(source_files_and_evaluation_model_hashes_verified=len(sources),
        source_evaluation_model_bundles=27,aggregate_results_independently_recomputed=30,
        monthly_confusion_matrices_independently_recomputed=270,monthly_counts_reconcile_with_overall=True,
        raw_and_intercept_adjusted_learned_warning_decisions_identical=True,
        earlier_completed_suite_verification='analysis/recloser_suite_first/numerical_review.json',
        original_artifacts_unchanged=True,new_model_fits=0,new_threshold_searches=0,
        selected_a_new_main_model=False,independent_test=False)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    import re
    refs=re.findall(r'\]\(([^)]+)\)',(OUT/'RESULTS.md').read_text(encoding='utf-8'))
    assert all((OUT/r).exists() for r in refs)
    print(selected[['tp','fp','precision','recall','false_positive_rate','average_precision']].to_string())
    print(json.dumps(verification,indent=2))
    print(OUT/'RESULTS.md')


if __name__=='__main__': main()
