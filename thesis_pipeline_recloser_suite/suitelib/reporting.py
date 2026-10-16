import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import csv_read, read_json
from .data import FAMILIES


def selected_rows(run):
    result = []
    for protocol, folder, choice_folder in [('Frozen', '07_frozen_evaluation', '04_frozen_selection_fit'),
                                           ('Monthly', '06_monthly_evaluation', '06_monthly_evaluation')]:
        scores = csv_read(run / folder / 'overall_metrics.csv')
        choice = read_json(run / choice_folder / 'selection.json')
        keys = ['main', 'best_full', 'best_raw'] if protocol == 'Frozen' else ['main', 'AP_alternative', 'feature_choice']
        names = list(dict.fromkeys(choice[k]['model_id'] for k in keys))
        names += [n for n in ['xgboost__full__all__raw' if protocol == 'Frozen' else 'xgboost__full__12m__raw'] if n not in names]
        for name in names:
            row = scores[(scores.model_id == name) & (scores.policy == 'fpr20')].iloc[0].to_dict()
            result.append((protocol, row))
    return result


def figures(run):
    out = run / '08_report/figures'; out.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(run)
    labels = [protocol+' / '+r['model_id'] for protocol, r in rows]
    pos = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(rows)*.65)), layout='constrained')
    axes[0].barh(pos-.18, [r['tp'] for _, r in rows], height=.34, label='Detected positive unit-days', color='#307a64')
    axes[0].barh(pos+.18, [r['fp'] for _, r in rows], height=.34, label='False warnings', color='#bf6257')
    axes[0].set_yticks(pos, labels, fontsize=8)
    axes[0].set_xlabel('Unit-days (57 recorded positives in 2,098 eligible rows)')
    axes[0].legend(fontsize=8)
    axes[1].barh(pos, [r['average_precision'] for _, r in rows], color='#397a9d')
    axes[1].set_yticks(pos, [])
    axes[1].axvline(57/2098, ls='--', color='#c47a20', label='Positive fraction 2.72%')
    axes[1].set_xlabel('Exploratory 2026 average precision')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Recloser-day experiment suite: earlier-selected procedures and XGBoost references\nFPR20 constrains earlier calibration; all 2026 results remain exploratory', fontsize=12)
    fig.savefig(out / 'selected_procedures.png', dpi=150)
    fig.savefig(out / 'selected_procedures.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, folder, window, title in [(axes[0], '07_frozen_evaluation', 'all', 'Frozen, expanding history'),
                                     (axes[1], '06_monthly_evaluation', '12m', 'Monthly, 12-month history')]:
        scores = csv_read(run / folder / 'overall_metrics.csv')
        part = scores[(scores.kind == 'xgboost') & (scores['mode'] == 'raw') &
                      (scores.window == window) & (scores.policy == 'fpr20')].set_index('family').loc[list(FAMILIES)]
        positions = np.arange(len(part))
        ax.barh(positions-.18, part.tp, height=.34, color='#307a64', label='Detected')
        ax.barh(positions+.18, part.fp, height=.34, color='#bf6257', label='False warnings')
        ax.set_yticks(positions, part.index)
        ax.set_xlabel('Unit-days'); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle('All recloser XGBoost feature groups — same eligible 2026 rows; earlier-selected FPR20 rules', fontsize=11)
    fig.savefig(out / 'feature_combinations.png', dpi=150)
    plt.close(fig)


def rows_table(rows):
    result = ['| Procedure | Detected /57 | False warnings | Missed | Precision | Recall | FPR | AP | AUC | Loss |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in rows:
        result.append(f"| {name} | {int(r['tp'])} | {int(r['fp'])} | {int(r['fn'])} | {r['precision']:.1%} | "
                      f"{r['recall']:.1%} | {r['false_positive_rate']:.1%} | {r['average_precision']:.4f} | "
                      f"{r['roc_auc']:.4f} | {r['log_loss']:.4f} |")
    return result


def write_report(run, config):
    frozen = read_json(run / '04_frozen_selection_fit/selection.json')
    monthly = read_json(run / '06_monthly_evaluation/selection.json')
    selected = selected_rows(run)
    text = ['# Recloser-day experiment suite', '',
            'The city-day experiment families were repeated for nine named recloser service areas using unchanged '
            'recloser labels, frozen geographic proxies and the exact earlier evaluation population. '
            'There are 2,098 eligible evaluated unit-days, 57 positive recloser-days on 41 dates, and nine reclosers. '
            'A no-record row is not verified uninterrupted power.', '',
            f"Earlier frozen probability-loss selection: **{frozen['main']['model_id']}**. "
            f"Earlier monthly warning-oriented selection: **{monthly['main']['model_id']}**. "
            f"Monthly pooled development FPR allowance met: **{monthly['development_budget_met']}**. "
            'These choices are recorded before their 2026 evaluation, but 2026 was already examined before designing this suite.', '',
            '## Main comparisons', '']
    text += rows_table([(p+' / '+r['model_id'], r) for p, r in selected])
    text += ['', 'All rows above use FPR20 thresholds learned on earlier data. Future FPR can exceed 20%. '
             'Frozen and monthly procedures are selected under different earlier criteria, as in the city experiments; '
             'do not attribute differences solely to updating. Their evaluation rows and labels match exactly.', '',
             '## Scope of experiments', '',
             '- Frozen algorithm baseline: logistic regression, small spline logistic regression, XGBoost, CatBoost '
             'and random forest on history, all-weather, compact-full and original 88-feature families.',
             '- Frozen XGBoost feature combinations: rainfall, wind, temperature/humidity, vegetation, all-weather, '
             'local weather-vegetation products, dry-to-wet and both additions.',
             '- Monthly logistic/XGBoost: history/full x 12-month/24-month/expanding histories. '
             'Monthly XGBoost additionally tests every feature combination at 12 months and the legacy 88-input '
             'family across all three windows. This repeats experiment families, not every possible Cartesian product.',
             '- Frozen raw/nonnegative-sigmoid and monthly raw/intercept-only probability comparisons.',
             '- FPR20, FPR10, balanced-accuracy, F1 and fixed-0.50 warning rules for every probability procedure.',
             '- All-no-record-zero sensitivity with unchanged fitted models, plus per-recloser and source-support audits.', '',
             'There are 27 frozen and 23 monthly base procedures, 135 frozen and 552 monthly model bundles '
             '(687 total), with 2,748 four-trial inner fits. No new source download or package installation occurs.', '',
             '## Feature definitions', '', '| Family | Inputs |', '|---|---:|']
    for name, fields in FAMILIES.items(): text.append(f'| {name} | {len(fields)} |')
    text += ['', 'Compact history includes recloser identity, categorical weekday, annual sine/cosine, earlier '
             'outage counts 3/7, recency capped at seven, and mandatory equipment counts 7/14. '
             'Full adds mapped weather summaries/histories, monthly vegetation and interactions. '
             'Local variants replace cross-area rain/gust x vegetation products with means/maxima of products '
             'formed within the same barangay before recloser aggregation. Dry-to-wet fractions use 3/7 preceding '
             'dry days within each mapped barangay. Weather histories never exceed seven days. '
             'Prior wet-day counts retain the original recloser mapped-mean-rain definition (>=0.1 mm); '
             'the city any-site definition is not silently substituted.', '',
             'The legacy 88-feature family retains the earlier feature catalogue, including fitting-only per-unit '
             'P90 stress, fault/repair subtypes, shared-other-recloser histories, and original remote-sensing coverage '
             'proxy. Its new models use the new common unweighted algorithm/calibration policy. '
             'Original fitted legacy models and their F1 rules are separately preserved as references; these are '
             'not the same procedure as refitting the 88 inputs under modern suite settings.', '',
             '## Spatial and label limits', '',
             'The nine names are Buncag, Chariot, Factor, Kalikasan, Lacsamana, Mendoza, PENRO, PILTEL and San Manuel. '
             'The existing pre-development mapping has 33 recloser/barangay links across 24 distinct barangays. '
             'Weights remain equal within each recloser proxy. Shared weather grid values are retained; no distinct '
             'weather, asset ages, electrical connectivity or service-area polygons are invented. Same-month vegetation '
             'is a geographic proxy, not measured branch contact or verified equipment exposure.', '',
             'The label is a recorded interruption affecting part of a named service area, not necessarily failure '
             'of the physical recloser. Existing assignment exclusions and publication-date fallbacks remain unchanged. '
             'Primary fitting/scoring excludes otherwise-zero rows on dates with notices lacking an extracted recloser, '
             'while preserving positives. That mask uses target-day reporting evidence and is retrospective/conditional, '
             'not an operational prediction-time filter. Sensitivity includes all 2,376 unit-days and the same 57 positives.', '',
             '## Chronological workflow', '',
             'Frozen models reuse four original 2025 development origins and five-role cohorts. Final refit/calibration '
             'use only pre-2026 data. The model, preprocessor, calibrator and thresholds stay fixed throughout 2026. '
             'Frozen selection minimizes mean earlier outer-quarter log loss, then higher AP and ID. '
             'A fitted sigmoid slope zero yields a constant procedure and must not be presented as useful detection.', '',
             'Monthly origins run October 2024–September 2026, with three initial calibration-pool months. '
             'At M use records published by M-1 20:00 PHT and incident maturity through M-15. '
             'Tuning uses the preceding three months; initial fit ends 14 days before tuning starts. '
             'Refit uses the full mature 12/24/all-history window. Calibration uses genuinely earlier monthly '
             'predictions joined on recloser-day row ID; no current-model in-sample scores. '
             'January–December 17, 2025 are scored for selection. December 18–31 raw predictions remain available '
             'for later calibration without entering selection outcomes. Later 2026 updates may use completed '
             'earlier 2026 outcomes after publication/maturity checks, never current/future target-month labels.', '',
             monthly['rule'], '', monthly['feature_rule'], '',
             'The 7-of-12 feature screen is a disclosed experimental consistency criterion, not an industry standard. '
             'All feature candidate scores are retained even when not selected. Monthly intercept adjustment preserves '
             'within-month ranking and, with corresponding retuned thresholds, alert decisions; it is not independent '
             'evidence of improved detection.', '',
             '## All raw frozen procedures at the primary FPR20 rule', '']
    fs = csv_read(run / '07_frozen_evaluation/overall_metrics.csv')
    subset = fs[(fs['mode'] == 'raw') & (fs.policy == 'fpr20')]
    text += rows_table([(r.model_id, r.to_dict()) for _, r in subset.iterrows()])
    text += ['', '## All raw monthly procedures at the primary FPR20 rule', '']
    ms = csv_read(run / '06_monthly_evaluation/overall_metrics.csv')
    subset = ms[(ms['mode'] == 'raw') & (ms.policy == 'fpr20')]
    text += rows_table([(r.model_id, r.to_dict()) for _, r in subset.iterrows()])
    text += ['', '## Cutoffs, uncertainty and reproducibility', '',
             'Frozen thresholds are saved once per procedure; monthly thresholds are saved per origin. '
             'The 0.50 reference is explicit and is not substituted for the earlier-selected policies. '
             'F1 now uses the separate earlier calibration period; the original legacy reference used its earlier '
             'tuning F1 threshold and is labelled accordingly. Earlier FPR allowances are not guarantees of future FPR. '
             'Precision is stored as zero by the inherited metric convention if no warnings exist; empirical warning '
             'precision is undefined in that case. No-warning accuracy is 97.3% on the primary cohort while recall is zero.', '',
             'Date-block bootstrap resamples seven calendar days with all corresponding available recloser rows '
             'together (300 replicates). It does not treat nine units on one date as independent weather observations. '
             'Intervals are exploratory and conditional on these fitted/selected procedures, not full selection uncertainty. '
             'All model trials, preprocessing, selected rounds, role memberships, probability modes, thresholds, '
             'per-unit metrics, support audits, primary/sensitivity predictions and original reference results are saved.', '',
             'All 2026 results remain exploratory; these dates were repeatedly examined and motivated the suite. '
             'Historical target-day weather and own-month vegetation do not establish operational day-ahead input '
             'availability. Reporting completeness is unverified. Different city/barangay/recloser units and eligibility '
             'must not be compared as though they have the same target prevalence or independently observed events.', '',
             'References: [thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html), '
             '[calibration](https://scikit-learn.org/stable/modules/calibration.html), '
             '[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).', '']
    (run / '08_report/THESIS_RECLOSER_SUITE_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
