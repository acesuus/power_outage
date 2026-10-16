"""Report requested thesis metrics from unchanged archived predictions."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             brier_score_loss, confusion_matrix)

RUN = ROOT / 'thesis_pipeline_recloser_suite/runs/recloser_suite_first'
OUT = ROOT / 'reports/candidate_metrics'
SPECS = [('Frozen expanding', 'frozen', 'all')]
POLICIES = ['fpr20','fpr10','balanced_accuracy','f1','fixed050']

def read(p):
    return pd.read_csv(p, float_precision='round_trip')

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def reliability(y, p, approach, mode):
    # Equal-frequency bins, with tied probabilities kept together. All-equal
    # scores become one bin, rather than producing an empty qcut result.
    bins = pd.qcut(pd.Series(p), q=10, labels=False, duplicates='drop')
    if bins.isna().all():
        bins = pd.Series(np.zeros(len(p), dtype=int))
    f = pd.DataFrame(dict(bin=bins.astype(int), observed=y, probability=p))
    table = f.groupby('bin', sort=True).agg(rows=('observed','size'),
          positives=('observed','sum'), mean_probability=('probability','mean'),
          observed_rate=('observed','mean'), minimum_probability=('probability','min'),
          maximum_probability=('probability','max')).reset_index()
    table.insert(0, 'mode', mode)
    table.insert(0, 'approach', approach)
    table['absolute_gap'] = (table.mean_probability-table.observed_rate).abs()
    assert table.rows.sum() == len(y) and table.positives.sum() == y.sum()
    ece = float(np.average(table.absolute_gap, weights=table.rows))
    return table, ece

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {key: read(RUN / folder / 'predictions.csv') for key, folder in
            [('frozen','07_frozen_evaluation')]}
    scores = {key: read(RUN / folder / 'overall_metrics.csv') for key, folder in
            [('frozen','07_frozen_evaluation')]}
    assert len(data['frozen']) == 2098 and data['frozen'].outage_label.sum() == 57
    source_paths = [RUN / folder / filename for folder in ['07_frozen_evaluation']
                    for filename in ['predictions.csv','overall_metrics.csv']]
    before = {str(p.relative_to(ROOT).as_posix()): digest(p) for p in source_paths}
    metrics, calibration, bin_tables = [], [], []
    for label, protocol, window in SPECS:
        pred = data[protocol]
        y = pred.outage_label.to_numpy(dtype=int)
        base = 'xgboost__history__'+window
        prior = np.zeros(len(pred))
        path = RUN / '04_frozen_selection_fit/models' / base / 'calibration.csv'
        prior[:] = read(path).outage_label.mean()
        source_paths.append(path)
        prior_brier = brier_score_loss(y, prior)
        for mode in ['raw', 'monotone_sigmoid']:
            model_id = base+'__'+mode
            p = pred[model_id+'_probability'].to_numpy()
            assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
            bins, ece = reliability(y, p, label, mode)
            bin_tables.append(bins)
            bs = float(brier_score_loss(y, p))
            auc = float(roc_auc_score(y, p))
            calibration.append(dict(approach=label, mode=mode, rows=len(y), positives=int(y.sum()),
                observed_rate=float(y.mean()), mean_probability=float(p.mean()),
                mean_probability_minus_observed=float(p.mean()-y.mean()),
                brier_score=bs, calibration_ece_10_quantile=ece, nonempty_bins=len(bins),
                earlier_calibration_rate_reference_brier=float(prior_brier),
                brier_skill_vs_earlier_calibration_rate=float(1-bs/prior_brier)))
            for policy in POLICIES:
                w = pred[model_id+'_'+policy+'_warning'].to_numpy(dtype=int)
                tn, fp, fn, tp = map(int, confusion_matrix(y,w,labels=[0,1]).ravel())
                row = dict(approach=label, protocol=protocol, window=window, mode=mode, policy=policy,
                    model_id=model_id, rows=len(y), positives=int(y.sum()), auc_roc=auc,
                    f1_score=float(f1_score(y,w,zero_division=0)),
                    precision=float(precision_score(y,w,zero_division=0)),
                    recall=float(recall_score(y,w,zero_division=0)), brier_score=bs,
                    tn=tn, fp=fp, fn=fn, tp=tp, warnings=int(w.sum()),
                    precision_defined=bool(w.sum()), calibration_ece_10_quantile=ece)
                saved = scores[protocol]
                old = saved[saved.model_id.eq(model_id) & saved.policy.eq(policy)]
                assert len(old)==1
                old = old.iloc[0]
                for new, field in [('auc_roc','roc_auc'),('f1_score','f1'),('recall','recall'),('brier_score','brier_score')]:
                    assert np.isclose(row[new],old[field],rtol=0,atol=1e-12), (model_id,policy,field)
                if row['precision_defined']:
                    assert np.isclose(row['precision'],old.precision,rtol=0,atol=1e-12)
                assert (tn,fp,fn,tp)==tuple(int(old[k]) for k in ['tn','fp','fn','tp'])
                metrics.append(row)
    metrics = pd.DataFrame(metrics)
    calibration = pd.DataFrame(calibration)
    bins = pd.concat(bin_tables, ignore_index=True)
    primary = metrics[metrics['mode'].eq('raw') & metrics.policy.eq('fpr20')].copy()
    assert primary[['tp','fp']].to_numpy().tolist() == [[44,1140]]
    metrics.to_csv(OUT/'all_warning_policies.csv',index=False)
    primary.to_csv(OUT/'primary_fpr20_results.csv',index=False)
    calibration.to_csv(OUT/'calibration_summary.csv',index=False)
    bins.to_csv(OUT/'reliability_bins.csv',index=False)

    # Preserve the low-probability range relevant to rare outages.
    fig, axis = plt.subplots(figsize=(7,5.5), constrained_layout=True)
    colors = ['#2166ac','#b35806']
    for ax, (label,protocol,window) in zip([axis],SPECS):
        c = calibration[calibration.approach.eq(label)]
        b = bins[bins.approach.eq(label)]
        maximum = max(float(b.mean_probability.max()),float(b.observed_rate.max()),.05)
        limit = min(1.0,maximum*1.18)
        ax.plot([0,limit],[0,limit],color='#777777',linestyle='--',linewidth=1,label='Perfect calibration')
        for color, mode in zip(colors,c['mode']):
            g = b[b['mode'].eq(mode)]
            display = 'Raw' if mode=='raw' else 'Earlier-fitted calibration'
            ax.plot(g.mean_probability,g.observed_rate,'o-',color=color,label=display,markersize=4,linewidth=1.2)
        ax.set(title=label, xlabel='Mean predicted outage probability', ylabel='Observed outage rate',
               xlim=(0,limit),ylim=(0,limit))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Frozen XGBoost/history probability calibration\n2,098 rows, 57 positives; up to 10 quantile bins per mode',fontsize=12)
    fig.savefig(OUT/'calibration_reliability.png',dpi=180)
    plt.close(fig)

    lines = ['# Frozen XGBoost/history evaluation','',
        'Same **2,098 recloser-days**, **57 positive rows** on 41 dates, January 1 to September 21, 2026. '
        'The table uses raw probabilities and the original earlier-learned **FPR20** warnings. '
        'No model, cutoff or prediction was changed.','',
        '| Training approach | AUC-ROC | F1-score | Precision | Recall | Brier score |',
        '|---|---:|---:|---:|---:|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.approach} | {r.auc_roc:.4f} | {r.f1_score:.4f} | {r.precision:.2%} | {r.recall:.2%} | {r.brier_score:.6f} |')
    lines += ['', 'AUC-ROC measures ranking across cutoffs (0.5 is chance-level ranking). '
        'F1 is the harmonic mean of precision and recall at the saved warning rule. '
        'Brier score is the mean squared error of predicted outage probabilities (lower is better). '
        'AUC-ROC and Brier use continuous probabilities, not thresholded warnings.','',
        '| Training approach | Detected | False warnings | Missed | True negatives |',
        '|---|---:|---:|---:|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.approach} | {r.tp}/57 | {r.fp:,} | {r.fn} | {r.tn:,} |')
    lines += ['', 'All five warning policies and both probability '
        'modes are in [all_warning_policies.csv](all_warning_policies.csv). When no warnings are issued, '
        'the CSV uses the zero-division convention precision=0 and flags `precision_defined=False`.','',
        '## Calibration analysis','',
        'Observed outage frequency is **57/2,098 = 2.7169%**. Calibration asks whether predicted '
        'probabilities match observed frequencies; it is distinct from detecting outages at a cutoff. '
        'The previously fitted frozen sigmoid calibration is compared with raw '
        'scores below. They are not refitted on these evaluation outcomes.','',
        '| Training approach | Probability mode | Mean probability | Observed rate | Brier score | ECE (10 quantile bins) |',
        '|---|---|---:|---:|---:|---:|']
    for r in calibration.itertuples():
        lines.append(f'| {r.approach} | {r.mode} | {r.mean_probability:.2%} | {r.observed_rate:.2%} | {r.brier_score:.6f} | {r.calibration_ece_10_quantile:.6f} |')
    lines += ['', '![Reliability diagrams](calibration_reliability.png)','',
        'Points below the diagonal indicate predicted risk exceeding the observed rate in that bin. '
        'Points above it indicate predicted risk below the observed rate. Each score mode has up to '
        '10 equal-frequency bins, keeping tied scores together. Bin counts, positives, probability '
        'ranges and observed rates are in [reliability_bins.csv](reliability_bins.csv). '
        'ECE is the bin-size-weighted absolute difference between mean probability and observed rate. '
        'It depends on binning and is descriptive. With only 57 positives and correlated rows across '
        'dates/reclosers, bin frequencies can be noisy.','',
        'A small Brier score alone can be misleading when outages are rare. The always-zero '
        'probability reference scores **0.027169** here. A more informative reference predicts the '
        'outage rate from the frozen model\'s earlier calibration data. No evaluation labels set those reference '
        'probabilities. The following skill scores compare against that reference (positive is better):','',
        '| Training approach | Probability mode | Earlier-rate reference Brier | Brier skill |',
        '|---|---|---:|---:|']
    for r in calibration.itertuples():
        lines.append(f'| {r.approach} | {r.mode} | {r.earlier_calibration_rate_reference_brier:.6f} | {r.brier_skill_vs_earlier_calibration_rate:.2%} |')
    frozen_raw = calibration[(calibration.approach=='Frozen expanding') & (calibration['mode']=='raw')].iloc[0]
    frozen_cal = calibration[(calibration.approach=='Frozen expanding') & (calibration['mode']=='monotone_sigmoid')].iloc[0]
    lines += ['', '### Interpretation','',
        f'The frozen raw model averages {frozen_raw.mean_probability:.2%} predicted risk against '
        f'{frozen_raw.observed_rate:.2%} observed outages. Its earlier-fitted calibration reduces '
        f'that average to {frozen_cal.mean_probability:.2%}, and reduces Brier score from '
        f'{frozen_raw.brier_score:.6f} to {frozen_cal.brier_score:.6f}, but it still overpredicts on average.','',
        'Matching the overall outage frequency does not establish reliable probabilities '
        'for individual risk groups. Brier skill relative to the earlier-rate '
        'reference is near zero or negative, so the small absolute Brier scores do not demonstrate '
        'a substantial probability-prediction improvement over that reference.','',
        'The frozen monotone calibration preserves its overall ordering.','',
        'These reports use the existing retrospective evaluation population and do not select a new '
        'main model from the examined 2026 results. Low precision and the false-warning counts remain '
        'material to interpretation even where recall is high.','',
        '## Reproduce','', '`python report_candidate_metrics.py`','',
        'The script independently recomputes all 10 frozen mode/policy metric rows and compares '
        'them with archived scores. Input hashes and verification are in '
        '[verification.json](verification.json). Original data, models and archived reports remain unchanged.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    after = {name: digest(ROOT/name) for name in before}
    assert before == after
    manifest = {p.relative_to(ROOT).as_posix():digest(p) for p in source_paths}
    verification = dict(status='passed', metric_rows_recomputed=len(metrics),
        calibration_rows=len(calibration), all_metrics_match_archived_scores=True,
        primary_counts_match_requested_comparison=True, original_inputs_unchanged=True,
        reliability_bin_counts_and_positive_totals_verified=True, source_hashes=manifest)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(primary[['approach','auc_roc','f1_score','precision','recall','brier_score']].to_string(index=False))
    print(calibration.to_string(index=False))

if __name__ == '__main__':
    main()
