"""V2 figures and paired calendar-block exploratory uncertainty."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mpl_cache'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from lib.reporting import plt, save_figure, top_k_scores


def paired_bootstrap(frame, predictions, config):
    calendar = pd.date_range(frame.date.min(), frame.date.max(), freq='D').strftime('%Y-%m-%d').to_numpy()
    indices = {d: np.flatnonzero(frame.date.to_numpy() == d) for d in calendar}
    width, rng = config['bootstrap_block_days'], np.random.default_rng(config['seed'])
    records = []
    for replicate in range(config['bootstrap_replicates']):
        starts = rng.integers(0, len(calendar)-width+1, size=int(np.ceil(len(calendar)/width)))
        days = np.concatenate([calendar[s:s+width] for s in starts])[:len(calendar)]
        take = np.concatenate([indices[d] for d in days])
        y = frame.outage_label.to_numpy()[take]
        scores = {}
        for name, p in predictions.items():
            probability = np.clip(p[take], 1e-7, 1-1e-7)
            scores[name] = {'average_precision': float(average_precision_score(y, probability)),
                'brier_score': float(brier_score_loss(y, probability)),
                'log_loss': float(log_loss(y, probability, labels=[0, 1])),
                'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan}
            records.append({'replicate': replicate, 'model': name, **scores[name]})
        for comparison in ['xgboost_v1_original', 'xgboost_base45_matched']:
            delta = {k: scores['xgboost_v2'][k]-scores[comparison][k] for k in scores['xgboost_v2']}
            records.append({'replicate': replicate, 'model': 'delta_v2_minus_' + comparison, **delta})
    draws, rows = pd.DataFrame(records), []
    for model, g in draws.groupby('model'):
        for metric in ['average_precision', 'brier_score', 'log_loss', 'roc_auc']:
            values = g[metric].dropna()
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({'model': model, 'metric': metric, 'lower_95': lower,
                         'upper_95': upper, 'valid_replicates': len(values),
                         'method': 'paired moving seven-calendar-day percentile bootstrap; exploratory'})
    return pd.DataFrame(rows)


def comparison_figures(frame, probabilities, results, folder):
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    curves, bins = [], []
    for name, p in probabilities.items():
        precision, recall, threshold = precision_recall_curve(frame.outage_label, p)
        ax.step(recall, precision, where='post', label=name)
        curves.append(pd.DataFrame({'model': name, 'precision': precision, 'recall': recall,
                                    'threshold': np.r_[threshold, np.nan]}))
    ax.axhline(frame.outage_label.mean(), linestyle='--', color='gray', label='Positive fraction')
    ax.set(xlabel='Recall', ylabel='Precision', title='Already-seen 2026 cohort: exploratory precision–recall', xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    save_figure(fig, folder / 'exploratory_precision_recall.png')
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in probabilities.items():
        observed, predicted = calibration_curve(frame.outage_label, p, n_bins=10, strategy='quantile')
        ax.plot(predicted, observed, marker='o', label=name)
        bins.append(pd.DataFrame({'model': name, 'mean_predicted': predicted, 'observed_fraction': observed}))
    ax.plot([0, 0.3], [0, 0.3], linestyle='--', color='gray')
    ax.set(xlabel='Mean predicted probability', ylabel='Recorded outage fraction',
           title='Exploratory 2026: probability reliability', xlim=(0, 0.3), ylim=(0, 0.3))
    ax.legend(fontsize=8)
    save_figure(fig, folder / 'exploratory_reliability.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = list(results)
    for ax, metric, title in zip(axes, ['average_precision', 'brier_score', 'log_loss'],
                                 ['AP (higher is better)', 'Brier score (lower is better)', 'Log loss (lower is better)']):
        ax.bar(labels, [results[n]['calibrated'][metric] for n in labels], color='#28576c')
        ax.tick_params(axis='x', labelrotation=65, labelsize=8)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    save_figure(fig, folder / 'exploratory_model_comparison.png')
    return pd.concat(curves, ignore_index=True), pd.concat(bins, ignore_index=True)
