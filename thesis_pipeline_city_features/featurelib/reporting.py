import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import csv_read, read_json
from .features import FAMILIES


def figures(run):
    out = run / '06_report/figures'
    out.mkdir(parents=True, exist_ok=True)
    ranked = csv_read(run / '04_selection/ranked_features.csv').sort_values('mean_monthly_AP')
    evaluation = csv_read(run / '05_exploratory_2026/overall_metrics.csv')
    evaluation = evaluation[(evaluation['mode'] == 'raw') & (evaluation.policy == 'fpr20')].set_index('family').loc[ranked.family]
    selected = read_json(run / '04_selection/selection.json')['main']['family']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), layout='constrained')
    positions = np.arange(len(ranked))
    colors = ['#d17b22' if name == selected else '#397a9d' for name in ranked.family]
    axes[0].barh(positions, ranked.mean_monthly_AP, color=colors)
    axes[0].set_yticks(positions, ranked.family)
    axes[0].set_xlabel('Mean monthly average precision')
    axes[0].set_title('2025 development: feature comparison')
    axes[0].set_xlim(0, max(.55, ranked.mean_monthly_AP.max() + .04))
    axes[1].barh(positions-.18, evaluation.tp, height=.34, label='Outage dates detected', color='#307a64')
    axes[1].barh(positions+.18, evaluation.fp, height=.34, label='False warnings', color='#bf6257')
    axes[1].set_yticks(positions, ranked.family)
    axes[1].set_xlabel('Number of dates (264 total; 92 recorded outage dates)')
    axes[1].set_title('2026 exploratory: earlier-selected FPR20 rules')
    axes[1].legend(loc='lower right', fontsize=8)
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Monthly 12-month XGBoost: feature groups\nOrange = 2025-selected procedure; FPR20 constrains earlier calibration, not future FPR', fontsize=12)
    fig.savefig(out / 'feature_comparison.png', dpi=160)
    fig.savefig(out / 'feature_comparison.pdf')
    plt.close(fig)


def write_report(run, config):
    out = run / '06_report'
    choice = read_json(run / '04_selection/selection.json')
    ranked = csv_read(run / '04_selection/ranked_features.csv')
    overall = csv_read(run / '05_exploratory_2026/overall_metrics.csv')
    name = choice['main']['family']
    selected = overall[(overall.family == name) & (overall['mode'] == 'raw') & (overall.policy == 'fpr20')].iloc[0]
    full = overall[(overall.family == 'full_control') & (overall['mode'] == 'raw') & (overall.policy == 'fpr20')].iloc[0]
    text = ['# City-day XGBoost feature experiment', '',
            f"The earlier-development rule {'selected a replacement' if choice['replacement_qualified'] else 'found no qualifying replacement and retained the original full control'}: **{name}**. "
            f"Its later exploratory result is **{int(selected.tp)}/92 outage dates detected**, **{int(selected.fp)} false warnings**, "
            f"recall {selected.recall:.1%}, precision {selected.precision:.1%}, FPR {selected.false_positive_rate:.1%}.", '',
            f"Against the original monthly full model (69 detections, 99 false warnings), the selected procedure changes "
            f"detections by {int(selected.tp-full.tp):+d} and false warnings by {int(selected.fp-full.fp):+d}. "
            'A lower warning count alone does not establish better prediction. Compare ranking/probability quality and missed outages as well.', '',
            '## Question and unchanged target', '',
            'Can changes to feature representation improve city-day outage prediction? One row remains one calendar date, '
            'with at least one qualifying public outage record as positive. The same 1,369 posts, corrected incident dates, '
            'publication fallbacks, equipment histories and 58-barangay weather registry are retained. '
            'Zero means no qualifying public record, not independently verified uninterrupted service.', '',
            '## Feature groups', '', '| Family | Inputs | Definition |', '|---|---:|---|']
    labels = dict(history_control='Original calendar/outage/equipment histories; preserved predictions',
                  full_control='Original 36-feature full model; preserved predictions',
                  rainfall='History + current/prior rainfall, duration, hourly peak, wet-day counts',
                  wind='History + current/prior gusts', temperature_humidity='History + mean temperature/humidity',
                  vegetation='History + four monthly vegetation summaries',
                  all_weather='History + all original weather; no vegetation/interactions',
                  full_local='Replace two cross-city rain/gust x vegetation products with four same-barangay products',
                  full_dry='Original full + two within-barangay dry-to-wet fractions',
                  full_local_dry='Both spatial interaction replacements and dry-to-wet additions')
    for family, fields in FAMILIES.items():
        text.append(f'| {family} | {len(fields)} | {labels[family]} |')
    text += ['', 'Every family retains the mandatory 7/14-day equipment history. Outage history remains 3/7 days '
             'and recency capped at seven. Weather histories never exceed seven days. Newly engineered dry-to-wet '
             'fractions are the share of sites rainy on D after every preceding 3 or 7 days at that same site was dry '
             '(rain <0.1 mm). Same-barangay products use observed same-month vegetation; missing values remain '
             'missing, and no absent vegetation is turned into zero. These are geographic proxies, not verified asset exposure.', '',
             '## Earlier 2025 development', '',
             '| Family | Mean monthly AP | Mean monthly loss | AP better months /12 | Loss better months /12 | Recall | FPR | Qualifies |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for _, r in ranked.iterrows():
        text.append(f'| {r.family} | {r.mean_monthly_AP:.4f} | {r.mean_monthly_log_loss:.4f} | '
                    f'{int(r.AP_better_months)} | {int(r.loss_better_months)} | {r.recall:.1%} | '
                    f'{r.false_positive_rate:.1%} | {bool(r.eligible_for_replacement)} |')
    text += ['', choice['rule'], '',
             'The 7-of-12 stability and FPR20 criteria are disclosed experimental choices, not industry standards. '
             'The ranking alternative and stricter FPR10 alternative remain saved in selection.json. '
             'No 2026 outcome enters the replacement decision; the candidate definitions were motivated by already-examined results.', '',
             '## Same-date exploratory 2026 comparison', '',
             '| Family | AP | AUC | Log loss | Detected /92 | False warnings | Missed | Precision | Recall | FPR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    show = overall[((overall['mode'] == 'raw') & (overall.policy == 'fpr20')) |
                   (overall.family == 'static_xgboost_full')]
    for _, r in show.iterrows():
        text.append(f'| {r.family} | {r.average_precision:.4f} | {r.roc_auc:.4f} | {r.log_loss:.4f} | '
                    f'{int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {r.precision:.1%} | {r.recall:.1%} | {r.false_positive_rate:.1%} |')
    text += ['', 'All candidates, raw/adjusted probabilities, FPR20/FPR10/fixed-0.50 rules, monthly counts, '
             'and seven-day-block bootstrap intervals for selected/control/reference procedures are saved. '
             'These allowance rules use earlier calibration predictions; their future realized FPR can exceed the allowance. '
             'Fixed 0.50 is a reference, not evidence of calibrated probabilities. No best-looking 2026 row replaces the earlier choice.', '',
             '## Chronological protocol and reproducibility', '',
             '1. Audit and hash original sources, code dependencies and preserved earlier artifacts.',
             '2. Add six documented features, preserve all original columns, and reuse exact original monthly cohort memberships.',
             '3. Generate October–December 2024 warmup predictions and January–December 2025 monthly predictions. '
             'Score January 1–December 17 only. Use the original four XGBoost trials per new family/month, '
             'select tuning AP then loss, and refit with the selected number of trees.',
             '4. Save the feature selection using only earlier 2025 development results.',
             '5. Update monthly January–September 2026; evaluate the same January 1–September 21 dates '
             '(264 days; 92 recorded positives). Earlier 2026 outcomes may enter later updates after publication/maturity checks.',
             '6. Reload all 192 new monthly model bundles, verify predictions and fitting-only medians/round counts, '
             'recompute confusion matrices, verify original artifact preservation, and produce this report.', '',
             'The rolling window is 12 months with a trailing 14-day reporting buffer. Initial fitting precedes tuning '
             'with a 14-day gap; tuning uses the prior three months and refit includes tuning. Calibration uses '
             'genuinely earlier monthly predictions. Preprocessing is fitting-only. No target-month label is fitted. '
             'The two controls reuse their exact original monthly scores, thresholds and model settings. '
             'Only new feature families are trained; 192 refits and 768 inner trials are saved. '
             'Month checkpoints support resuming without repeating completed fits.', '',
             '## Limits', '',
             'All 2026 results are exploratory because those dates were examined before this experiment. '
             'Historical D weather and own-month vegetation do not establish operational day-ahead availability. '
             'City aggregation predicts somewhere in PPC, not an affected barangay/recloser. '
             'Unreported events, reporting changes, uncertain network exposure and absent asset-condition measurements remain unresolved. '
             'No public records or zero labels were changed to improve a score. '
             'A better-performing feature group would indicate predictive usefulness in these periods, not a causal mechanism. '
             'Bootstrap intervals are conditional on this experiment and do not cover all feature-selection uncertainty.', '',
             'References: [threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html), '
             '[feature selection](https://scikit-learn.org/stable/modules/feature_selection.html), '
             '[importance limitations](https://scikit-learn.org/stable/modules/permutation_importance.html).', '']
    (out / 'THESIS_FEATURE_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
