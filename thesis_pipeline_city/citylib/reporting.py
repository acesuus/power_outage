"""Thesis report and static scientific figures from completed, frozen predictions."""
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from .common import csv_read,read_json
from lib.reporting import plt,save_figure


def figures(frame,probabilities,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6))
    for name,p in probabilities.items():
        precision,recall,_=precision_recall_curve(frame.outage_label,p)
        ax.step(recall,precision,where='post',label=name.replace('__',' / '))
    ax.axhline(frame.outage_label.mean(),color='gray',linestyle='--',label='Recorded positive fraction')
    ax.set(xlabel='Recall',ylabel='Precision',title='PPC city-day: exploratory 2026 precision–recall',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8,loc='lower left');save_figure(fig,folder / 'precision_recall.png')
    fig,ax=plt.subplots(figsize=(10,6))
    for name,p in probabilities.items():
        observed,predicted=calibration_curve(frame.outage_label,p,n_bins=5,strategy='quantile')
        ax.plot(predicted,observed,marker='o',label=name.replace('__',' / '))
    ax.plot([0,1],[0,1],color='gray',linestyle='--')
    ax.set(xlabel='Mean predicted probability',ylabel='Recorded outage fraction',
        title='PPC city-day: exploratory probability calibration',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8,loc='upper left');save_figure(fig,folder / 'calibration.png')


def write_report(run,config,out):
    audit=read_json(run / '01_audit/report.json');feature=read_json(run / '02_features/report.json')
    choice=read_json(run / '05_selection/selection.json');evaluation=read_json(run / '07_exploratory_2026/report.json')
    scores=csv_read(run / '07_exploratory_2026/overall_metrics.csv')
    warnings=csv_read(run / '07_exploratory_2026/warning_policies.csv')
    cohort=csv_read(run / '03_cohorts/cohort_summary.csv');ranked=csv_read(run / '05_selection/ranked_procedures.csv')
    verification=read_json(out / 'verification.json')
    main=evaluation['main'];result=evaluation['models'][main]
    text=['# PPC city-day broad baseline','',
        f"The procedure selected by mean 2025 development probability loss is **{main}**. Its exploratory 2026 accuracy is {result['accuracy']:.1%}, precision {result['precision']:.1%}, and recall {result['recall']:.1%} at the earlier-validation balanced-accuracy threshold.",'',
        f"It detects **{result['tp']} of {result['positives']} recorded outage days**, misses {result['fn']}, and produces {result['fp']} false warnings across {result['rows']} evaluated city-days. These are city occurrences, not predictions of affected barangays or reclosers.",'',
        '## Target and source','',
        f"All {audit['source_posts']:,} retained unscheduled PPC posts support the target. They collapse to {audit['positive_city_days']} distinct positive dates in {audit['calendar_days']} calendar days ({audit['first_date']} to {audit['last_date']}). Multiple notices or affected areas on the same date create one positive city-day. Posts are not counted as independent physical failures.",'',
        'A positive means at least one retained incident/outage date has a qualifying public record somewhere in PPC. A zero means no qualifying record was found. No record is not independently verified continuous power. Restoration spanning midnight does not automatically create a second positive day. All retained unscheduled causes remain included.','',
        f"The source has {audit['publication_fallback_posts']} publication-date fallback posts and {audit['fallback_only_positive_days']} dates supported only by fallback posts. These retain their accepted dates and quality flags. The known conflict is retained and flagged. All {audit['missing_recloser_posts']} posts without a recloser and {audit['missing_barangay_posts']} posts without a cleaned barangay can still support a city label. No target-day localization mask is used.",'',
        '## Adjusted features','',
        f"The compact full set contains {feature['conceptual_inputs']} inputs: "+', '.join(f'{k}={v}' for k,v in feature['family_counts'].items())+' for the nested comparisons. Each exact definition, unit, timing and family appears in 02_features/feature_dictionary.csv.','',
        'Current historical weather uses equal means across the fixed 58 available weather barangays, plus spatial maxima for rainfall, gust and hourly rainfall peak, and a rainy-barangay fraction. Rainfall is never summed across places. Mean precipitation hours is a daily summary of hourly inputs; the model still has one row per day. Weather code is a categorical mode, with smallest numeric code breaking ties.','',
        'The existing weather registry covers 58 of the city’s barangays; it is an exposure proxy, not a complete area-weighted or network-weighted city measurement. Equal barangay weighting overrepresents densely subdivided urban geography relative to land area and does not count independent weather stations. Spatial maxima may be far from an electrical fault. No official feeder, asset or customer weights are available.','',
        'Prior weather is lag one day and accumulated rainfall / maximum gust over the preceding 3 and 7 calendar days. Wet-day counts use any sampled barangay rainfall >=0.1 mm. Summed daily spatial maxima may come from different barangays on different dates. These are separate exposure summaries, not rainfall at a single site.','',
        'Monthly vegetation uses the same 58-barangay scope: means of 5 m and 30 m proxies, spatial P90 of the 5 m proxy, and mean barangay NDVI. Means use available observations; all-missing observations stay missing. Sampling coverage is audit metadata rather than a physical predictor. Source buffer values are supplied remote-sensing proxies, not verified vegetation touching electrical lines. D’s own monthly composite is retained as requested; observations after D can enter that composite.','',
        'Outage history counts distinct known incident dates over 3/7 days, with recency capped at 7. Mandatory equipment-fault history counts distinct city incident dates over 7/14 days. All history requires incident < D and supporting publication <= D−1 20:00 PHT. Fault history specifically requires a fault-bearing post available by that cutoff. Multiple same-date failures still count as one fault day.','',
        'Calendar uses categorical weekday and annual sine/cosine. Three interactions represent rainfall×vegetation, gust×vegetation and rain presence×prior equipment faults. Recloser/barangay identity is omitted because there is one geographic unit. Overlapping temperature/wind/apparent-temperature measures, redundant month/weekend indicators, very sparse fault-subcategory histories, and the 15 m vegetation proxy are omitted from this compact comparison. This does not establish that omitted features have no predictive value.','',
        '## Chronological development and final fitting','',
        'Four origins in 2025 use five date roles: initial fit, subsequent tuning, refit, separate calibration/threshold period, and outer evaluation. Fourteen-day maturity gaps separate fit–tune, refit–calibration and calibration–outer roles. Thirty initial calendar days are warmup. Expanding training is used. Positive days whose only support was published after a role’s historical as-of are excluded; their counts are reported below. This is a conditional availability rule, not a complete historical reporting reconstruction.','',
        '| Origin | Role | First date | Last date | Rows | Positives | Late-only positives excluded |',
        '|---|---|---|---|---:|---:|---:|']
    for r in cohort.itertuples(index=False):
        text.append(f'| {r.origin} | {r.role} | {r.first_date} | {r.last_date} | {r.rows} | {r.positives} | {r.late_publication_positive_rows_excluded} |')
    text+=['',
        'Logistic regression uses L2 regularization with four C candidates. Spline logistic regression adds fixed low-complexity smooth weather/vegetation functions: three uniform knots, degree two, constant extrapolation, fitted only on training observations. It is a small GAM-like additive extension with selected explicit interactions, not a separate large neural model. CatBoost compares depth 2/3 and leaf L2 regularization 10/30, with at most 300 trees, tuning-only early stopping, chronological ordering and natural class prevalence. Each learned algorithm compares history/calendar, history/weather and full features.','',
        'Inner selection minimizes tuning probability log loss, with average precision breaking ties. Refit uses the selected parameters and tree count. Imputation medians, missingness indicators, categorical encoding, scaling and spline knots are learned from fitting/refitting rows only. The model cannot use raw text, target-day causes, date-quality flags, current affected areas or current supporting-post counts as inputs.','',
        'Both raw and nondecreasing sigmoid-calibrated probability procedures are scored in each outer quarter. The latter is fitted on the separate earlier calibration quarter, with slope constrained >=0 and a fixed 0.001 penalty toward slope 1. A zero slope creates a constant probability; it cannot reverse ranks. Raw probabilities remain genuine comparison candidates. No automatic claim is made that calibration improves performance.','',
        'Warning thresholds are learned from the separate calibration-period predictions of the refitted model. The primary policy maximizes balanced accuracy; higher threshold breaks ties. The secondary policy maximizes recall subject to calibration false-positive rate <=20%. These are experimental operating rules, not a utility-approved warning tolerance. Neither guarantees the same future false-positive rate. Changing a threshold does not improve ranking.','',
        'Probability procedure selection minimizes mean 2025 outer-quarter log loss over learned algorithm/family/mode combinations; average precision then model ID break ties. The best full-input procedure and AP-ranking alternative are frozen separately. Historical-frequency and month-frequency baselines are reported outside learned-candidate selection. Month rates use 30 pseudo-days at the pooled refitting rate.','',
        '| Procedure | Mean 2025 loss | Mean AP | Mean AUC |','|---|---:|---:|---:|']
    for r in ranked.itertuples(index=False):text.append(f'| {r.model_id} | {r.mean_log_loss:.4f} | {r.mean_average_precision:.4f} | {r.mean_roc_auc:.4f} |')
    text+=['',f"Best full-input procedure: **{choice['best_full_candidate']['model_id']}**. AP-ranking alternative: **{choice['ranking_alternative']['model_id']}**.",'',
        '## Exploratory 2026 results','',
        f"Frozen evaluation runs from {evaluation['first_date']} to {evaluation['last_date']}: {evaluation['rows']} eligible dates, {evaluation['positive_city_days']} positive days. The later source end and 14-day reporting buffer define this evaluation window; dates after September 21 are not part of this reported comparison.",'',
        '| Procedure | AP | AUC | Log loss | Accuracy | Balanced accuracy | Precision | Recall | TP | FP | FN | TN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in scores.itertuples(index=False):
        text.append(f'| {r.model_id} | {r.average_precision:.4f} | {r.roc_auc:.4f} | {r.log_loss:.4f} | {r.accuracy:.3f} | {r.balanced_accuracy:.3f} | {r.precision:.3f} | {r.recall:.3f} | {r.tp} | {r.fp} | {r.fn} | {r.tn} |')
    text+=['','Selected model warning-policy comparison:','',
        '| Earlier-validation policy | TP | FP | FN | TN | Precision | Recall | Actual 2026 FPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in warnings[warnings.model_id.eq(main)].itertuples(index=False):
        text.append(f'| {r.policy} | {r.tp} | {r.fp} | {r.fn} | {r.tn} | {r.precision:.3f} | {r.recall:.3f} | {r.false_positive_rate:.3f} |')
    references=csv_read(run / '07_exploratory_2026/constant_decision_reference.csv')
    text+=['', 'Constant decision references on the same evaluation dates: '+', '.join(
        f"{r.rule}: accuracy {r.accuracy:.1%}, recall {r.recall:.1%}" for r in references.itertuples(index=False))+'. These clarify why accuracy alone does not establish useful outage warnings.','',
        '## Interpretation and limits','',
        'The target is whether any recorded interruption occurs somewhere in PPC, so this model provides one probability per date and cannot locate the affected barangay/recloser. City aggregation avoids uncertain assignment of unlocalized posts and duplication of labels across affected barangays. It does not repair unreported interruptions or changing public reporting coverage.','',
        'City-day prevalence, labels and evaluation population differ from barangay-day and recloser-day. A higher AP, recall or accuracy cannot be used as a direct performance improvement claim over those different tasks. Compare city models against city frequency baselines on the same dates. Paired seven-calendar-day block bootstrap intervals and paired development feature comparisons are supplied; neither establishes causal weather/vegetation effects.','',
        'The 2026 period was repeatedly examined in earlier experiments and motivated this city hypothesis. It is exploratory even though none of its rows entered model fitting, calibration, threshold selection or procedure selection. A later unused period is required for independent confirmation. Current historical weather and same-month vegetation make this a retrospective conditional-weather experiment. Operational day-ahead accuracy still needs issue-time forecasts and vegetation available at the issue time.','',
        f"Verification confirmed {verification['preserved_prior_artifacts']} prior artifact hashes, source/code hashes, fitting-only medians/scaling, nonnegative calibration slopes, one calendar row per date, and model reloads for every saved procedure. The runner writes COMPLETION.json only after all eight stages complete, then writes the run artifact manifest.",'',
        '## Primary documentation','',
        '- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api): gridded historical weather inputs.',
        '- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): calibrate using observations separate from model fitting.',
        '- [scikit-learn decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html): probability estimation and warning decisions are separate.',
        '- [scikit-learn spline transformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html): fitting spline bases and extrapolation.',
        '']
    (out / 'THESIS_CITY_DAY_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
