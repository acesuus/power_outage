"""Grouped evaluation, calendar-block uncertainty and standalone research figures."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from .modeling import metrics


def block_bootstrap(frame, probabilities, decisions, config):
    """Sample overlapping seven-calendar-day blocks, retaining all eligible areas."""
    calendar = pd.date_range(frame.date.min(), frame.date.max(), freq='D').strftime('%Y-%m-%d').to_numpy()
    indices = {d: np.flatnonzero(frame.date.to_numpy() == d) for d in calendar}
    rng, width = np.random.default_rng(config['seed']), config['bootstrap_block_days']
    samples = {name: [] for name in probabilities}
    for replicate in range(config['bootstrap_replicates']):
        starts = rng.integers(0, len(calendar) - width + 1, size=int(np.ceil(len(calendar) / width)))
        days = np.concatenate([calendar[s:s + width] for s in starts])[:len(calendar)]
        take = np.concatenate([indices[d] for d in days])
        for name, p in probabilities.items():
            result = metrics(frame.outage_label.to_numpy()[take], p[take], decision=decisions[name][take])
            samples[name].append(result)
    out = []
    for name, draws in samples.items():
        for metric in ['average_precision', 'brier_score', 'log_loss', 'roc_auc', 'precision', 'recall', 'f1']:
            values = [r[metric] for r in draws if r[metric] is not None]
            lower, upper = np.quantile(values, [0.025, 0.975]) if values else [None, None]
            out.append({'model': name, 'metric': metric, 'lower_95': lower, 'upper_95': upper,
                        'valid_replicates': len(values), 'method': 'moving seven-calendar-day block percentile bootstrap'})
    return pd.DataFrame(out)


def top_k_scores(frame, probabilities, k, area_count):
    rows = []
    for name, p in probabilities.items():
        scored = frame[['date', 'psgc', 'outage_label']].assign(probability=p)
        for date, day in scored.groupby('date', sort=True):
            if day.psgc.nunique() != area_count:
                continue  # Partial labels must not make top-k artificially easy.
            ranked = day.sort_values(['probability', 'psgc'], ascending=[False, True]).head(k)
            hits, positives = int(ranked.outage_label.sum()), int(day.outage_label.sum())
            rows.append({'model': name, 'date': date, 'k': k, 'hits': hits,
                         'positives': positives, 'precision_at_k': hits / k,
                         'recall_at_k': hits / positives if positives else np.nan})
    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def evaluation_figures(frame, probabilities, results, folder):
    folder.mkdir(exist_ok=True)
    curves, bins = [], []
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in probabilities.items():
        precision, recall, thresholds = precision_recall_curve(frame.outage_label, p)
        ax.step(recall, precision, where='post', label=name)
        curves.append(pd.DataFrame({'model': name, 'precision': precision, 'recall': recall,
                                    'threshold': np.r_[thresholds, np.nan]}))
    ax.axhline(frame.outage_label.mean(), color='gray', linestyle='--', label='Test prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Final test: precision–recall curves', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure(fig, folder / 'precision_recall.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probabilities.items():
        observed, predicted = calibration_curve(frame.outage_label, p, n_bins=10, strategy='quantile')
        ax.plot(predicted, observed, marker='o', label=name)
        bins.append(pd.DataFrame({'model': name, 'mean_predicted_probability': predicted,
                                  'observed_positive_fraction': observed}))
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set(xlabel='Mean predicted probability', ylabel='Observed outage fraction',
           title='Final test: probability reliability', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure(fig, folder / 'reliability.png')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric, title in zip(axes, ['average_precision', 'brier_score', 'log_loss'],
                                 ['Average precision (higher is better)', 'Brier score (lower is better)', 'Log loss (lower is better)']):
        ax.bar(list(results), [results[n]['calibrated'][metric] for n in results], color=['#8193a4', '#4c8eae', '#28576c'])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(bottom=0)
    save_figure(fig, folder / 'model_comparison.png')
    r = results['xgboost']['calibrated']
    matrix = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:,}', ha='center', va='center', color='white' if value > matrix.max() / 2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No recorded outage', 'Recorded outage'],
           yticklabels=['No recorded outage', 'Recorded outage'], xlabel='Frozen-threshold prediction',
           ylabel='Recorded target', title='XGBoost: final-test confusion matrix')
    save_figure(fig, folder / 'xgboost_confusion_matrix.png')
    return pd.concat(curves, ignore_index=True), pd.concat(bins, ignore_index=True)
