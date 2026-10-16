"""Scientific figures and a measured report, with changed target/cohort limitations explicit."""
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from .common import csv_read,read_json,csv_write
from lib.reporting import plt,save_figure


def figures(frame,predictions,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,p in predictions.items():
        precision,recall,_=precision_recall_curve(frame.outage_label,p)
        ax.step(recall,precision,where='post',label=name.replace('__',' / '))
    ax.axhline(frame.outage_label.mean(),color='gray',linestyle='--',label='Recorded positive fraction')
    ax.set(xlabel='Recall',ylabel='Precision',title='Nine reclosers: exploratory 2026 precision–recall',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8);save_figure(fig,folder / 'precision_recall.png')
    fig,ax=plt.subplots(figsize=(9,5))
    maximum=0.1
    for name,p in predictions.items():
        observed,predicted=calibration_curve(frame.outage_label,p,n_bins=5,strategy='quantile')
        ax.plot(predicted,observed,marker='o',label=name.replace('__',' / '))
        maximum=max(maximum,float(max(predicted.max(),observed.max()))*1.1)
    ax.plot([0,maximum],[0,maximum],color='gray',linestyle='--')
    ax.set(xlabel='Mean predicted probability',ylabel='Recorded outage fraction',title='Nine reclosers: exploratory probability calibration',xlim=(0,maximum),ylim=(0,maximum))
    ax.legend(fontsize=8);save_figure(fig,folder / 'calibration.png')


def write_report(run,config,out):
    audit=read_json(run / '01_audit/report.json');features=read_json(run / '02_features/report.json')
    evaluation=read_json(run / '07_exploratory_2026/report.json');choice=read_json(run / '05_selection/selection.json')
    rows=csv_read(run / '07_exploratory_2026/overall_metrics.csv')
    maprows=csv_read(run / '02_features/recloser_barangay_proxy_mapping.csv')
    ablation=csv_read(run / '05_selection/paired_feature_ablation.csv')
    cohort=csv_read(run / '03_cohorts/cohort_summary.csv')
    text=['# Nine-recloser historical-weather experiment','',
        f"Selected by 2025 development log loss: **{evaluation['main']}**. This model was chosen before scoring the already-examined 2026 labels. The unit registry and this hypothesis were motivated by previous results, so this is an exploratory comparison.",'',
        '## Target and source audit','',
        'One pooled model estimates whether at least one recorded unscheduled interruption affects part of a named recloser service area on day D. A positive does not require the recloser hardware itself to fail. A zero means no qualifying named-area report was found; it does not confirm uninterrupted supply. Recloser names are reporting units, not verified unchanged physical devices.','',
        f"The source has {audit['source_posts']:,} posts. The nine names have {audit['candidate_selected_posts']} candidate posts and {audit['candidate_post_recloser_assignments']} post/name assignments. One Lacsamana equipment mention in an explicitly affected Kalikasan notice is excluded from Lacsamana's label. The accepted evidence has {audit['accepted_supporting_posts']} supporting posts and {audit['accepted_assignments']} assignments, collapsed to **{audit['positive_recloser_days']} positive recloser-days** in a {audit['panel_rows']:,}-row calendar panel.",'',
        f"Posts with no extracted recloser occur on {audit['unassigned_recloser_post_dates']} incident dates. The primary experiment excludes {audit['uncertain_negative_rows']:,} otherwise-zero rows on those dates, including the disputed assignment where relevant. Positive rows remain eligible subject to date maturity. This conservative choice changes the evaluation population and may miss unassigned positive events. A separately reported sensitivity evaluates frozen models with all no-record zeros included; it is not used to select models.",'',
        'Corrected incident dates are retained. Publication-date fallback and the known date conflict remain flagged. Multiple notices collapse to a single binary recloser/date positive. Restoration crossing midnight does not expand the target onto the following day. All retained causes remain in the target.','',
        '## Spatial weather and vegetation proxies','',
        f"Mapping uses only single-recloser affected-area notices with incident dates through {config['mapping_incident_end']} and publication timestamps through {config['mapping_publication_asof']}. For each name, take the union of its listed mapped barangays in the preceding {config['mapping_recent_days']} days; use older eligible notices only if no recent ones exist. Freeze this proxy for every development and final model. Multi-recloser circuit notices do not assign their entire area list to each device. Kalikasan preserves the previously accepted inferred barangay mapping. This is an announcement-derived exposure proxy, not an official service-area/GIS map.",'',
        '| Recloser | Frozen proxy barangays |','|---|---|']
    for name,g in maprows.groupby('recloser'):
        text.append('| '+name+' | '+', '.join(sorted(g.barangay.unique()))+' |')
    text += ['',
        'Daily measurements are averaged with equal barangay weights. Rainfall, gust, precipitation-duration and peak-hourly-rain spatial maxima are additional inputs, with rainfall/gust ranges. Wind direction is averaged through sine/cosine components. Weather code uses the modal category, with the smallest numeric category breaking ties; it is never numerically averaged. Full mapped weather coverage is required for means and ranges. These are daily observations/proxies for the target day, not recloser-mounted weather measurements.','',
        'Monthly vegetation uses D’s month, averaged across available mapped barangays, with spatial maximum buffer values as additional exposure proxies. Entirely missing measurements remain missing. Observed-barangay fraction is saved as quality metadata; fitting-only missingness indicators and imputation handle missing predictors. NDVI/vegetation means and maxima are not weighted by line length because asset/network lengths are unavailable. Maxima may lie away from electrical lines. Same-month composites may contain observations after D, so their use remains retrospective.','',
        f"The selected proxies cover {features['unique_mapped_barangays']} distinct barangays. Rainfall is identical across all nine proxies on {features['weather_days_with_identical_rain_for_all_reclosers']} of {features['weather_days']} weather dates. Median distinct rainfall values per date: {features['median_distinct_recloser_rain_values_per_day']:.1f}. Different recloser labels do not manufacture finer weather resolution.",'',
        '## Features and chronological procedure','',
        f"The full family has {features['candidate_inputs']} conceptual inputs. Family sizes: "+', '.join(f"{name}={count}" for name,count in features['family_counts'].items())+'. See the feature dictionary and FEATURE_CATALOG.md for all variables. The internal compatibility column `psgc` contains the recloser name, not a fabricated geographic code.','',
        'Current historical weather and monthly vegetation retain the user-approved retrospective definition. Prior-weather windows are 1/3/7 days; outage history is 3/7 days; equipment/repair/category history is 7/14 days. Histories require incident date < D and publication <= D−1 20:00 Philippine time. Category availability comes from the category-bearing notice. Equipment counts describe previous affected-area fault reports, not failures of the named recloser itself. Other-recloser history counts prior named recloser/date pairs and does not establish network connectivity.','',
        'Four 2025 chronological development origins each have initial fitting, subsequent tuning, refitting, independent probability calibration and an outer quarter. Fourteen-day maturity gaps separate roles. All units share date-role boundaries. Four hyperparameter candidates per learner/family are selected by tuning AP, with tuning log loss as tie-breaker. Booster early stopping uses tuning log loss. Selected tree counts are reused at refit. All medians, encodings, scaling and weather P90 references are learned from fitting/refitting rows only. Sigmoid calibration uses a separate unweighted prior quarter. Warning thresholds maximize tuning F1 and are frozen before calibration and evaluation.','',
        'Learners: regularized pooled logistic regression, shallow XGBoost and native-category CatBoost. Each compares history/calendar, history/weather and full weather/vegetation inputs on identical eligible rows. A smoothed fitting-only per-recloser frequency benchmark is also evaluated. Primary selection minimizes mean calibrated 2025 outer-quarter log loss among the nine learned procedures, with AP as tie-breaker. All nine final learned procedures plus frequency are saved to show the registered ablations. No 2026 rows are passed to training, parameter selection or calibration.','',
        '## Exploratory 2026 results','',
        f"Evaluation: {evaluation['rows']:,} eligible rows on {evaluation['eligible_calendar_dates']} calendar dates, {evaluation['positive_recloser_days']} positive recloser-days across {evaluation['positive_calendar_dates']} positive dates, from {evaluation['first_date']} through {evaluation['last_date']}. The beginning and end correspond to the previous experiment's calendar bounds, but the target and eligible population differ.",'',
        '| Model | AP | ROC AUC | Log loss | Brier | Precision | Recall |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows.itertuples(index=False):
        text.append(f'| {row.model_id} | {row.average_precision:.4f} | {row.roc_auc:.4f} | {row.log_loss:.4f} | {row.brier_score:.4f} | {row.precision:.3f} | {row.recall:.3f} |')
    main=evaluation['models'][evaluation['main']]
    text += ['',f"Main-model warning counts: TP={main['tp']}, FN={main['fn']}, FP={main['fp']}, TN={main['tn']}. Warning precision/recall use a tuning-selected operating point; they should not be interpreted as a universal optimal threshold.",'',
        '## Paired development feature effects','',
        'Each difference is added family minus its matched simpler family on the same quarter. Higher AP/AUC and lower loss/Brier are favorable. Hyperparameters are retuned within each registered family; this is a pipeline comparison, not a causal estimate.','',
        '| Learner | Addition | Mean AP difference | Mean log-loss difference | Quarters with AP gain |','|---|---|---:|---:|---:|']
    for (kind,addition),g in ablation.groupby(['kind','addition']):
        text.append(f'| {kind} | {addition} | {g.average_precision_difference.mean():+.5f} | {g.log_loss_difference.mean():+.5f} | {int(g.average_precision_difference.gt(0).sum())}/4 |')
    text += ['',
        '## What can be concluded','',
        'Use within-recloser matched comparisons and date-block uncertainty intervals. Do not compare this AP directly with barangay AP to claim reclosers are better: prevalence, geographic unit and eligibility changed. Nine unit labels reduce spatial replication but also leave fewer positive examples. The all-year scope excludes later appearing/disappearing names and was selected after looking at the full historical period. Recloser footprints may have changed, and unknown publication-time text revisions limit exact historical availability reconstruction.','',
        'The experiment evaluates recorded named-area interruptions given historical weather and monthly vegetation. It does not verify all physical outages, predict device hardware failures, or establish operational day-ahead accuracy. Operational claims still require issue-time weather forecasts, available vegetation measurements, a verified service-area registry and a later untouched evaluation period.','',
        '## Reproducibility and sources','',
        'Eight separate stage scripts, preflight safeguards, code/source snapshots, cohort archives, fitted preprocessors, calibrated models, predictions and artifact hashes are saved with this run. Original V1/V2/V3 artifacts are checked before and after the experiment. Model reload predictions and fitting-only weather references are verified.','',
        '- [Open-Meteo historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api): gridded reanalysis/weather inputs, not independent stations per barangay.',
        '- [scikit-learn chronological splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): train before later evaluation periods.',
        '- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): calibrate with data separate from base-model fitting.','']
    (out / 'THESIS_RECLOSER_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
