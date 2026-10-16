import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from .common import csv_read,read_json
from lib.reporting import plt,save_figure


def figures(pred,names,main,folder,config):
    folder.mkdir(parents=True,exist_ok=True)
    labels={main:'Selected monthly procedure','static_logistic_history':'Frozen logistic history',
        'static_xgboost_full':'Frozen full XGBoost','recent_frequency':'Recent known outage rate'}
    def title(name):return labels.get(name,name.replace('__',' / '))
    fig,ax=plt.subplots(figsize=(10,6))
    for name in names:
        precision,recall,_=precision_recall_curve(pred.outage_label,pred[name+'_probability'])
        ax.step(recall,precision,where='post',label=title(name))
    ax.axhline(pred.outage_label.mean(),color='gray',linestyle='--',label='Recorded positive fraction')
    ax.set(xlabel='Recall',ylabel='Precision',title='PPC city-day: monthly updating, exploratory 2026',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8,loc='lower left');save_figure(fig,folder / 'precision_recall.png')
    fig,ax=plt.subplots(figsize=(10,6))
    for name in names:
        observed,predicted=calibration_curve(pred.outage_label,pred[name+'_probability'],n_bins=5,strategy='quantile')
        ax.plot(predicted,observed,marker='o',label=title(name))
    ax.plot([0,1],[0,1],color='gray',linestyle='--')
    ax.set(xlabel='Mean predicted probability',ylabel='Recorded outage fraction',
        title='PPC city-day: monthly probability calibration',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8,loc='upper left')
    save_figure(fig,folder / 'calibration.png')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    months=sorted(pred.origin.unique())
    for name in list(dict.fromkeys([main,'static_logistic_history','static_xgboost_full'])):
        recalls=[];fprs=[]
        for origin in months:
            g=pred[pred.origin.eq(origin)];y=g.outage_label.to_numpy();w=g[name+'_'+config['primary_warning_policy']+'_warning'].to_numpy()
            recalls.append(float(w[y==1].mean()));fprs.append(float(w[y==0].mean()))
        axes[0].plot(months,recalls,marker='o',label=title(name));axes[1].plot(months,fprs,marker='o',label=title(name))
    axes[0].set(ylabel='Recall',ylim=(0,1),title='PPC city-day: monthly warning performance')
    axes[1].axhline(config['false_positive_rate_limit'],color='gray',linestyle='--',label='20% reference allowance')
    axes[1].set(ylabel='False-positive rate',xlabel='2026 prediction month (September ends on day 21)',ylim=(0,1))
    axes[1].set_xticks(months,[m[5:7] for m in months]);axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    save_figure(fig,folder / 'monthly_warning_rates.png')


def write_report(run,config,out):
    choice=read_json(run / '04_selection/selection.json');result=read_json(run / '05_exploratory_2026/report.json')
    scores=csv_read(run / '05_exploratory_2026/overall_metrics.csv').set_index('model_id')
    ranked=csv_read(run / '04_selection/ranked_procedures.csv');verified=read_json(out / 'verification.json')
    reporting=csv_read(run / '01_audit/monthly_reporting_audit.csv')
    main=choice['main']['model_id'];selected=scores.loc[main];full=choice['best_full']['model_id']
    text=['# Monthly updating city-day experiment','',
        f"The earlier-development warning rule selected **{main}**. On {result['rows']} exploratory 2026 dates with {result['positives']} recorded outage dates, it detects {int(selected.tp)}, misses {int(selected.fn)}, and issues {int(selected.fp)} false warnings. Precision is {selected.precision:.1%}, recall {selected.recall:.1%}, ordinary accuracy {selected.accuracy:.1%}, and balanced accuracy {selected.balanced_accuracy:.1%}.",'',
        '## What changed','',
        'This experiment refreshes a model at the beginning of each calendar month. It compares logistic regression and shallow XGBoost, 12-month / 24-month / expanding training, an eight-input history/calendar control and the unchanged 36-input full weather/vegetation family. No new physical or weather features, target changes, new source downloads, or removal of accepted incident/publication-fallback dates were introduced.','',
        'The full inputs retain historical D weather, D-month vegetation, weather histories at 1/3/7 days, outage histories at 3/7 days and mandatory equipment history at 7/14 days. Fixed 58-barangay spatial summaries and label semantics remain those of the original city baseline. See its exact feature dictionary and methodology. Current raw text/causes, current post counts, uncertainty flags and later labels remain excluded from predictors.','',
        '## Time ordering and monthly learning','',
        'At origin M, the information cutoff is M−1 day at 20:00 Asia/Manila. Refitting ends M−15 days, giving a 14-day maturity buffer. The 12/24-month training windows begin M minus the selected number of calendar months; expanding begins at the original warmup date. The inner fitting set ends 14 days before a three-month tuning period; refit then uses all eligible window dates through the mature cutoff. A positive date is eligible only if at least one supporting post was published by the origin information cutoff. Historical unknown late-only positive dates are excluded, not relabeled zero. Public-record zeros are still not verified continuous supply.','',
        'Three monthly origins in October–December 2024 create raw out-of-sample predictions for calibration warmup. Models produce full December 2025 predictions even though development scoring ends December 17. Those December 18–31 predictions can be used for calibration of later 2026 updates after their outcomes are available. This avoids an unfilled prequential calibration gap.','',
        'Development scoring covers January 1–December 17, 2025, with label support available by December 31, 2025 at 20:00. The December reporting buffer prevents selecting the first January 2026 procedure using outcomes that were not yet available. The future 2026 procedure specification is selected before 2026 updating/scoring starts.','',
        'During 2026, later months are intentionally allowed to train on completed earlier 2026 dates once publication and maturity requirements are met. This is sequential updating, not one model fitted entirely before 2026. No target-month or later-month label enters that month’s tuning, refitting, calibration or threshold setting. All method/window/family choices are frozen from 2025. Target-day weather and monthly imagery remain retrospective inputs, so this timing discipline does not establish operational day-ahead availability.','',
        '## Tuning, preprocessing and calibration','',
        'Each month tunes four candidates: logistic L2 C={0.01,0.1,1,10}; XGBoost depth={2,3} × L2={10,30}. XGBoost retains child weight 5, learning rate 0.05, row/column sampling 0.8, natural prevalence, at most 300 trees and tuning log-loss early stopping patience 25. Inner model selection prioritizes tuning AP, then lower tuning loss, then trial order. Refit uses the selected XGBoost round count. Medians, category encodings, missing indicators and logistic standardization are learned only from the current fitting/refitting set. Trees have no standardization or splines.','',
        'Calibration uses raw predictions previously issued by the same algorithm/window/family procedure for the earlier three-month period ending M−15. Each of those predictions came from a model fitted before its own target month. The current refitted model can train on these earlier dates, but calibration uses their earlier genuinely out-of-sample scores, not current in-sample fitted predictions. This calibrates a retraining procedure and can be affected by score-distribution changes after refitting.','',
        'The intercept adjustment fixes the logit slope at one and fits only an offset so the average adjusted probability matches the earlier calibration fraction. It cannot create ranking information or flatten a nonconstant raw score to zero slope. Raw and adjusted probabilities are both evaluated. Within each month the ranking is preserved; between-month probability-level changes can change pooled ranking.','',
        'Warning thresholds are also chosen on those earlier out-of-sample predictions: primary maximizes recall subject to calibration FPR <=20%; secondary maximizes balanced accuracy. Higher threshold breaks ties. Re-fitting a threshold after an intercept shift produces identical raw/adjusted warning decisions within a month; that invariant is checked. Therefore any detection gains come from updating, training/window decisions and warning-policy changes, not from the offset itself. The earlier-calibration FPR allowance does not guarantee a future 20% FPR.','',
        '## Registered warning-oriented selection','',
        'Among the 24 learned algorithm/family/window/probability procedures, require pooled 2025 development FPR <=20%, then maximize pooled recall. Mean monthly AP, lower mean monthly probability loss and procedure ID break ties. If no learned procedure meets the allowance, choose lowest pooled FPR, then highest monthly AP, lower loss and ID, and explicitly flag the unmet allowance. The best full-family candidate follows the same rule. Highest-AP and lowest-loss alternatives are retained separately.','',
        f"Selected development FPR allowance met: **{choice['development_fpr_budget_met']}**. Best full-input candidate: **{full}**. AP alternative: **{choice['AP_alternative']['model_id']}**. Probability-loss alternative: **{choice['probability_alternative']['model_id']}**.",'',
        '| Procedure | Development recall | Development FPR | Mean monthly AP | Mean monthly loss | Meets FPR allowance |',
        '|---|---:|---:|---:|---:|---|']
    for r in ranked.itertuples(index=False):
        text.append(f'| {r.model_id} | {r.recall:.3f} | {r.false_positive_rate:.3f} | {r.mean_monthly_AP:.4f} | {r.mean_monthly_log_loss:.4f} | {r.development_budget_met} |')
    text+=['','## Matched exploratory 2026 comparison','',
        'All procedures are scored on the same 264 city dates, January 1–September 21, with 92 recorded positives. The frozen original logistic and XGBoost references use their earlier FPR20 thresholds, not whichever operating point appears most favorable now. Monthly frequency uses a smoothed mature expanding training rate; recent frequency uses a smoothed known mature three-month calibration rate. These rate benchmarks use a stated 0.5 probability decision rule for warnings and are most useful as probability baselines.','',
        '| Procedure | AP | AUC | Log loss | Detected | False warnings | Missed | Precision | Recall | FPR | Accuracy | Balanced accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    names=list(dict.fromkeys([main,full,'static_logistic_history','static_xgboost_full','static_xgboost_history_raw','monthly_frequency','recent_frequency']))
    for name in names:
        r=scores.loc[name]
        text.append(f'| {name} | {r.average_precision:.4f} | {r.roc_auc:.4f} | {r.log_loss:.4f} | {int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {r.precision:.1%} | {r.recall:.1%} | {r.false_positive_rate:.1%} | {r.accuracy:.1%} | {r.balanced_accuracy:.1%} |')
    text+=['',
        'Always predicting no outage has 65.2% accuracy and zero recall. Evaluate improvements using matched counts, actual FPR, precision, AP/AUC and probability loss together. A procedure can improve recall while exceeding its intended warning allowance. A procedure selected in earlier development can underperform a comparison later; do not replace it using the best-looking 2026 score. All candidate results and both warning policies are saved.','',
        '## Reporting continuity and uncertainty','',
        'The monthly source audit counts incident-dated posts, publication dates and recorded positive dates. It identifies changes in observed reporting frequency, not whether the archive is complete or physical outages declined. No month was removed or relabeled using these counts, and these target-month counts are not predictors. Independent source/service records would still be needed to establish missing outages or reporting coverage.','',
        'Paired seven-calendar-day moving-block bootstrap intervals use 300 replicates and include AP, AUC, probability loss, Brier, recall, FPR and precision, plus differences against static/recent-rate references. They are conditional on supplied labels/features and procedure choices. They do not incorporate all selection uncertainty, missing physical outages or all uncertainty from changing reporting.','',
        '2026 was already examined and motivated this change, so these are exploratory sequential-evaluation results. Current historical weather and same-month vegetation make them retrospective conditional-weather results. Later untouched dates and issue-time inputs remain necessary for independent operational day-ahead claims. The geographic target remains a single city occurrence probability, not affected-area prediction.','',
        f"Verification: {verified['monthly_model_bundles_verified']} monthly model bundles, original artifact hashes ({verified['preserved_original_artifacts']}), fitting-only medians/scaling, fixed boosting rounds, earlier out-of-sample calibration, model reloads, no overlap with current target-month training labels, and exact same final 264 dates/92 positives. Maximum reload probability difference: {verified['maximum_reload_probability_error']:.3g}.",'',
        '## References','',
        '- [scikit-learn chronological splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)',
        '- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)',
        '- [scikit-learn threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html)',
        '- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)','']
    (out / 'THESIS_MONTHLY_CITY_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
