"""Eight separately executed stages with auditable artifacts and fixed decision rules."""
import importlib.metadata as md
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import (ROOT, PROJECT, source, csv_read, csv_write, json_write, read_json,
                     sha, now, versions, code_hashes, check_sources)
from .data import (normalize_posts, make_labels, learn_proxy_mapping, aggregate_daily,
                   aggregate_vegetation, make_features, FEATURES, FAMILIES, GROUPS)
from .modeling import search, fit_final, predict, frequency_bundle, select, metrics
from v2lib.folds import definition, index_role, get_cohorts

PRIOR_RUNS = [('thesis_pipeline/runs/baseline_v1_3_first','11_report'),
    ('thesis_pipeline_v2/runs/v2_chronological_first','09_report'),
    ('thesis_pipeline_v3/runs/v3_rainfall_comparison_first','09_report')]


def audit(run,config):
    out = run / '01_audit'; out.mkdir(exist_ok=True)
    hashes = {rel:sha(PROJECT / rel) for rel in config['sources'].values()}
    json_write(out / 'source_hashes.json',hashes)
    prior = {}
    for folder,report_folder in PRIOR_RUNS:
        manifest = read_json(PROJECT / folder / report_folder / 'artifact_manifest.json')
        for relative,record in manifest.items():
            p = PROJECT / folder / relative
            assert sha(p) == record['sha256'], 'Prior artifact changed: ' + str(p)
            prior[str(p.relative_to(PROJECT))] = record['sha256']
    json_write(out / 'prior_artifact_hashes.json',prior)
    environment = versions(); environment['packages']['catboost'] = md.version('catboost')
    json_write(out / 'environment.json',environment)
    json_write(out / 'code_hashes.json',code_hashes())
    docs = out / 'documentation'; docs.mkdir(exist_ok=True)
    for path in [ROOT / 'README.md',ROOT / 'FEATURE_CATALOG.md',ROOT / 'docs/METHODOLOGY.md',ROOT / 'docs/STAGES.md']:
        shutil.copy2(path,docs / path.name)
    posts = normalize_posts(csv_read(source(run,'posts')))
    assert len(posts)==1369
    panel,evidence,review = make_labels(posts,config)
    csv_write(panel,out / 'recloser_day_labels.csv')
    csv_write(evidence,out / 'recloser_outage_evidence.csv')
    csv_write(review,out / 'assignment_review.csv')
    counts = posts[posts.recloser_clean.map(lambda s: bool(set(str(s).split(';')) & set(config['reclosers'])))].copy()
    annual = evidence.assign(year=evidence.date.str[:4]).groupby(['recloser','year']).agg(posts=('event_id','nunique'),incident_dates=('date','nunique')).reset_index()
    csv_write(annual,out / 'annual_support.csv')
    report = {'source_posts':len(posts),'candidate_selected_posts':len(counts),
        'candidate_post_recloser_assignments':sum(len(set(str(s).split(';')) & set(config['reclosers'])) for s in counts.recloser_clean),
        'accepted_supporting_posts':int(evidence.event_id.nunique()),'accepted_assignments':len(evidence),
        'assignment_exclusions':len(review),'panel_rows':len(panel),'positive_recloser_days':int(panel.outage_label.sum()),
        'positive_incident_dates':int(panel[panel.outage_label.eq(1)].date.nunique()),
        'unassigned_recloser_post_dates':int(panel[panel.unassigned_recloser_posts_on_date.gt(0)].date.nunique()),
        'uncertain_negative_rows':int(panel.unlocalized_negative_uncertainty.sum()),
        'publication_fallback_positive_days':int((panel.publication_fallback_post_count.gt(0)&panel.outage_label.eq(1)).sum()),
        'preserved_prior_artifacts':len(prior),'period':[panel.date.min(),panel.date.max()],
        'scope_chosen_after_examining_2026':True,'independent_test_claim':False}
    json_write(out / 'report.json',report)
    return report


def features(run,config):
    out = run / '02_features'; out.mkdir(exist_ok=True)
    posts = normalize_posts(csv_read(source(run,'posts')))
    panel = csv_read(run / '01_audit/recloser_day_labels.csv')
    evidence = csv_read(run / '01_audit/recloser_outage_evidence.csv')
    mapping,provenance = learn_proxy_mapping(posts,csv_read(source(run,'barangay_mapping')),config)
    assert provenance.date.max() <= config['mapping_incident_end']
    assert pd.to_datetime(provenance.published_at_local,utc=True).max() <= pd.Timestamp(config['mapping_publication_asof']).tz_convert('UTC')
    csv_write(mapping,out / 'recloser_barangay_proxy_mapping.csv')
    csv_write(provenance,out / 'mapping_source_evidence.csv')
    daily = aggregate_daily(csv_read(source(run,'weather')),csv_read(source(run,'rainfall')),mapping)
    vegetation = aggregate_vegetation(csv_read(source(run,'vegetation')),mapping)
    csv_write(daily,out / 'recloser_daily_weather_proxy.csv')
    csv_write(vegetation,out / 'recloser_monthly_vegetation_proxy.csv')
    f = make_features(panel,daily,vegetation,evidence,csv_read(source(run,'equipment')),config)
    csv_write(f,out / 'recloser_feature_panel.csv')
    catalog = [{'variable':n,'group':g,'families':';'.join(k for k,v in FAMILIES.items() if n in v)} for g,fields in GROUPS.items() for n in fields]
    csv_write(pd.DataFrame(catalog),out / 'feature_dictionary.csv')
    unique_rain = daily.groupby('date').precipitation_mm.nunique()
    test = daily[daily.date.between(config['test_start'],config['test_end'])]
    correlation = test.pivot(index='date',columns='psgc',values='precipitation_mm').corr()
    csv_write(correlation.rename_axis('recloser').reset_index(),out / 'rainfall_correlations_seen2026.csv')
    report = {'candidate_inputs':len(FEATURES),'family_counts':{k:len(v) for k,v in FAMILIES.items()},
        'mapping_barangays_per_recloser':mapping.groupby('recloser').size().to_dict(),
        'unique_mapped_barangays':int(mapping.psgc.nunique()),
        'mapping_incident_end':config['mapping_incident_end'],'mapping_publication_asof':config['mapping_publication_asof'],
        'weather_days_with_identical_rain_for_all_reclosers':int(unique_rain.eq(1).sum()),
        'weather_days':len(unique_rain),'median_distinct_recloser_rain_values_per_day':float(unique_rain.median()),
        'vegetation_missing_fraction':{n:float(f[n].isna().mean()) for n in ['veg_buffer_5m','veg_buffer_15m','veg_buffer_30m']},
        'official_service_area_mapping':False,'future_test_posts_used_for_mapping':False}
    json_write(out / 'report.json',report)
    return report


def cohorts(run,config):
    out = run / '03_cohorts'; out.mkdir(exist_ok=True)
    f = csv_read(run / '02_features/recloser_feature_panel.csv')
    e = csv_read(run / '01_audit/recloser_outage_evidence.csv')
    records=[]
    for origin in config['development_origins']+[config['test_start']]:
        defs = definition(origin)
        if origin == config['test_start']:
            defs['outer'].update(end=config['test_end'],asof=config['truth_asof'])
        indices={role:index_role(f,e,defs[role],'all',config) for role in ['fit','tune','refit','calibrate','outer']}
        for role,idx in indices.items():
            frame=f.iloc[idx]
            records.append({'origin':origin,'role':role,'first_date':frame.date.min(),'last_date':frame.date.max(),
                'rows':len(frame),'positives':int(frame.outage_label.sum()),'positive_dates':int(frame[frame.outage_label.eq(1)].date.nunique()),
                'asof':defs[role]['asof']})
        for earlier,later in [('fit','tune'),('refit','calibrate'),('calibrate','outer')]:
            assert set(f.iloc[indices[earlier]].date).isdisjoint(set(f.iloc[indices[later]].date))
            assert (pd.Timestamp(f.iloc[indices[later]].date.min())-pd.Timestamp(f.iloc[indices[earlier]].date.max())).days>=15
        assert f.iloc[indices['fit']].date.max()>=config['mapping_incident_end']
        np.savez_compressed(out / (origin+'.npz'),**indices)
        json_write(out / (origin+'.json'),defs)
    csv_write(pd.DataFrame(records),out / 'cohort_summary.csv')
    json_write(out / 'policy.json',{'roles':['fit','tune','refit','calibrate','outer'],'reporting_buffer_days':14,
        'uncertain_negative_rows_excluded':True,'2026_exploratory':True,'mapping_frozen_before_first_tuning_period':True})
    return {'folds':5,'2026_used_for_training_or_selection':False}


def development(run,config):
    out=run / '04_development'; out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/recloser_feature_panel.csv'); records=[]
    fit_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT / 'rclib/modeling.py',ROOT / 'rclib/data.py',ROOT / 'rclib/stages.py']}
    for origin in config['development_origins']:
        c=get_cohorts(f,run / '03_cohorts' / (origin+'.npz'))
        procedures=[(kind,family) for kind in config['algorithms'] for family in config['families']]+[('frequency','history_calendar')]
        for kind,family in procedures:
            name=kind if kind=='frequency' else kind+'__'+family
            folder=out / origin / name; complete=folder / 'complete.json'
            if complete.exists():
                saved=read_json(complete); assert saved['fitting_code_sha256']==fit_hashes
                for path,expected in saved['artifact_sha256'].items(): assert sha(folder / path)==expected
                records.append(read_json(folder / 'outer_metrics.json')); continue
            print('Development: '+origin+' '+name,flush=True)
            if kind=='frequency': bundle=frequency_bundle(c,folder)
            else:
                selected=search(c,config,kind,family,folder / 'search')
                bundle=fit_final(c,config,kind,family,selected,folder)
            raw,p,warning=predict(bundle,c['outer']); score=metrics(c['outer'].outage_label,p,decision=warning)
            record={'fold':origin,'model_id':name,'kind':kind,'family':family,**score}
            pred=c['outer'][['row_id','date','recloser','outage_label']].copy()
            pred['raw_probability'],pred['probability'],pred['warning']=raw,p,warning
            csv_write(pred,folder / 'outer_predictions.csv'); json_write(folder / 'outer_metrics.json',record)
            json_write(complete,{'fitting_code_sha256':fit_hashes,'artifact_sha256':{n:sha(folder / n) for n in ['model.joblib','outer_metrics.json','outer_predictions.csv']}})
            records.append(record); csv_write(pd.DataFrame(records),out / 'development_metrics.csv')
            print(f'  AP={score["average_precision"]:.4f}, loss={score["log_loss"]:.4f}, positives={score["positives"]}',flush=True)
    csv_write(pd.DataFrame(records),out / 'development_metrics.csv')
    assert len(records)==40
    return {'development_results':40,'inner_tuning_trials':144,'2026_used':False}


def selection(run,config):
    out=run / '05_selection'; out.mkdir(exist_ok=True)
    d=csv_read(run / '04_development/development_metrics.csv')
    ranked,main=select(d); csv_write(ranked,out / 'ranked_procedures.csv')
    comparisons=[]
    for kind in config['algorithms']:
        pivot=d[d.kind.eq(kind)].pivot(index='fold',columns='family',values=['average_precision','log_loss','brier_score','roc_auc'])
        for fold in pivot.index:
            for label,added,base in [('weather','history_weather','history_calendar'),('vegetation','full','history_weather')]:
                comparisons.append({'kind':kind,'fold':fold,'addition':label,
                    **{metric+'_difference':float(pivot.loc[fold,(metric,added)]-pivot.loc[fold,(metric,base)]) for metric in ['average_precision','log_loss','brier_score','roc_auc']}})
    csv_write(pd.DataFrame(comparisons),out / 'paired_feature_ablation.csv')
    choice={'main':main,'ranking_alternative':ranked.sort_values(['mean_average_precision','mean_log_loss'],ascending=[False,True]).iloc[0].to_dict(),
        'rule':'minimum mean calibrated 2025 development log loss among nine learned procedures; AP breaks ties',
        '2026_used':False,'scope_and_hypothesis_motivated_by_previously_seen_2026':True}
    json_write(out / 'selection.json',choice)
    return choice


def final_fitting(run,config):
    out=run / '06_final_models'; out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/recloser_feature_panel.csv')
    c=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))
    training={key:c[key] for key in ['fit','tune','refit','calibrate']}; specs={}
    for kind in config['algorithms']:
        for family in config['families']:
            name=kind+'__'+family; folder=out / name
            print('Final fitting: '+name,flush=True)
            selected=search(training,config,kind,family,folder / 'search')
            fit_final(training,config,kind,family,selected,folder)
            specs[name]={'kind':kind,'family':family,'path':str((folder / 'model.joblib').relative_to(run))}
    frequency_bundle(training,out / 'frequency')
    specs['frequency']={'kind':'frequency','family':'history_calendar','path':str((out / 'frequency/model.joblib').relative_to(run))}
    json_write(out / 'models.json',specs)
    return {'saved_models':len(specs),'2026_rows_passed_to_fitting':False}


def bootstrap(frame,predictions,main,comparators,config):
    calendar=pd.date_range(frame.date.min(),frame.date.max(),freq='D').strftime('%Y-%m-%d').to_numpy()
    groups={d:np.flatnonzero(frame.date.to_numpy()==d) for d in calendar}
    rng=np.random.default_rng(config['seed']); width=config['bootstrap_block_days']; rows=[]
    for repeat in range(config['bootstrap_replicates']):
        starts=rng.integers(0,len(calendar)-width+1,size=int(np.ceil(len(calendar)/width)))
        days=np.concatenate([calendar[s:s+width] for s in starts])[:len(calendar)]
        take=np.concatenate([groups[d] for d in days]); y=frame.outage_label.to_numpy()[take]
        if len(np.unique(y))<2: continue
        scores={name:metrics(y,p[take]) for name,p in predictions.items() if name in [main]+comparators}
        for name in scores:
            rows.append({'repeat':repeat,'comparison':name,**{k:scores[name][k] for k in ['average_precision','roc_auc','log_loss','brier_score']}})
        for name in comparators:
            if name==main: continue
            rows.append({'repeat':repeat,'comparison':'main_minus_'+name,**{k:scores[main][k]-scores[name][k] for k in ['average_precision','roc_auc','log_loss','brier_score']}})
    draws=pd.DataFrame(rows); result=[]
    for name,g in draws.groupby('comparison'):
        for key in ['average_precision','roc_auc','log_loss','brier_score']:
            lo,hi=g[key].quantile([.025,.975])
            result.append({'comparison':name,'metric':key,'lower95':lo,'upper95':hi,'valid_replicates':len(g),
                'method':'paired seven-calendar-day block bootstrap; exploratory, nine areas remain grouped'})
    return pd.DataFrame(result)


def evaluation(run,config):
    out=run / '07_exploratory_2026'; out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/recloser_feature_panel.csv')
    frame=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))['outer'].reset_index(drop=True)
    specs=read_json(run / '06_final_models/models.json'); choice=read_json(run / '05_selection/selection.json')
    predictions=frame[['row_id','date','recloser','outage_label']].copy(); scores=[]; probabilities={}; groups=[]
    sensitivity=f[f.date.between(config['test_start'],config['test_end'])].reset_index(drop=True)
    sensitivity_rows=[]
    for name,info in specs.items():
        bundle=joblib.load(run / info['path']); raw,p,w=predict(bundle,frame); probabilities[name]=p
        predictions[name+'_raw_probability'],predictions[name+'_probability'],predictions[name+'_warning']=raw,p,w
        scores.append({'model_id':name,'family':info['family'],'kind':info['kind'],**metrics(frame.outage_label,p,decision=w)})
        for recloser,g in frame.groupby('recloser'):
            idx=g.index.to_numpy(); groups.append({'model_id':name,'recloser':recloser,**metrics(g.outage_label,p[idx],decision=w[idx])})
        _,sp,sw=predict(bundle,sensitivity)
        sensitivity_rows.append({'model_id':name,**metrics(sensitivity.outage_label,sp,decision=sw),
            'policy':'includes otherwise-zero rows on dates with unassigned recloser posts; descriptive sensitivity only'})
    csv_write(predictions,out / 'predictions.csv'); csv_write(pd.DataFrame(scores),out / 'overall_metrics.csv')
    csv_write(pd.DataFrame(groups),out / 'recloser_metrics.csv'); csv_write(pd.DataFrame(sensitivity_rows),out / 'all_no_record_zeros_sensitivity.csv')
    main=choice['main']['model_id']; kind=choice['main']['kind']
    comparators=list(dict.fromkeys(['frequency']+[kind+'__'+family for family in config['families']]))
    csv_write(bootstrap(frame,probabilities,main,comparators,config),out / 'block_bootstrap_intervals.csv')
    from .reporting import figures
    figures(frame,{name:probabilities[name] for name in comparators},out / 'figures')
    report={'main':main,'models':{row['model_id']:{k:v for k,v in row.items() if k not in ['model_id','family','kind']} for row in scores},
        'rows':len(frame),'positive_recloser_days':int(frame.outage_label.sum()),'positive_calendar_dates':int(frame[frame.outage_label.eq(1)].date.nunique()),
        'first_date':frame.date.min(),'last_date':frame.date.max(),'eligible_calendar_dates':int(frame.date.nunique()),
        '2026_used_for_selection':False,'independent_untouched_test':False,'operational_day_ahead_test':False,
        'direct_comparison_with_barangay_AP_is_valid':False}
    json_write(out / 'report.json',report)
    return {'main':main,'main_metrics':report['models'][main],'frequency_metrics':report['models']['frequency'],'exploratory_only':True}


def report(run,config):
    out=run / '08_report'; out.mkdir(exist_ok=True)
    check_sources(run)
    prior=read_json(run / '01_audit/prior_artifact_hashes.json')
    for path,expected in prior.items(): assert sha(PROJECT / path)==expected
    initial=read_json(run / '01_audit/code_hashes.json'); assert initial==code_hashes()
    f=csv_read(run / '02_features/recloser_feature_panel.csv')
    frame=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))['outer'].reset_index(drop=True)
    saved=csv_read(run / '07_exploratory_2026/predictions.csv'); specs=read_json(run / '06_final_models/models.json')
    reload_errors={}
    for name,info in specs.items():
        bundle=joblib.load(run / info['path']); take=frame.iloc[:200]; raw,p,w=predict(bundle,take)
        error=float(np.max(np.abs(p-saved[name+'_probability'].iloc[:len(take)].to_numpy())))
        assert error<1e-6 and np.array_equal(w,saved[name+'_warning'].iloc[:len(take)])
        reload_errors[name]=error
        if info['kind']!='frequency':
            refit=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))['refit']
            for raw,values in bundle['transformer'].p90.items():
                expected=refit.groupby('psgc')[raw].quantile(config['weather_stress_quantile']).clip(lower=1).to_dict()
                assert expected==values
            assert not set(bundle['transformer'].fields) & {'outage_label','supporting_post_count','published_at_local','raw_text','unlocalized_negative_uncertainty'}
    verification={'sources_unchanged':True,'code_unchanged_since_audit':True,'preserved_prior_artifacts':len(prior),
        'saved_model_probability_reload_max_errors':reload_errors,'fitting_only_weather_references_verified':True,
        'preflight_tests':read_json(run / 'preflight_tests.json'),'stage_statuses':{str(i):read_json(run / 'status' / f'{i:02}.json')['status'] for i in range(1,8)},
        '2026_independent_test':False,'target_unit_matches_original_barangay_models':False}
    json_write(out / 'verification.json',verification)
    from .reporting import write_report
    write_report(run,config,out)
    manifest={str(p.relative_to(run)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in run.rglob('*')
        if p.is_file() and p!=out / 'artifact_manifest.json' and p.parent.name not in ['logs','status']}
    json_write(out / 'artifact_manifest.json',manifest)
    return {'artifacts':len(manifest),'preserved_prior_artifacts':len(prior),'report':str(out / 'THESIS_RECLOSER_REPORT.md')}
