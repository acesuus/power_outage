"""Eight independent executable stages; registered rules precede evaluation."""
import importlib.metadata as md
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import (ROOT,PROJECT,source,csv_read,csv_write,json_write,read_json,sha,versions,
    code_hashes,check_inputs)
from .data import (normalize_posts,make_labels,weather_registry,aggregate_weather,
    aggregate_vegetation,make_features,dictionary,FEATURES,FAMILIES)
from .modeling import search,fit_final,baseline_bundle,predict,select,metrics
from v2lib.folds import definition,index_role,get_cohorts

PRIOR_RUNS = [('thesis_pipeline/runs/baseline_v1_3_first','11_report'),
    ('thesis_pipeline_v2/runs/v2_chronological_first','09_report'),
    ('thesis_pipeline_v3/runs/v3_rainfall_comparison_first','09_report'),
    ('thesis_pipeline_recloser/runs/recloser9_validated_first','08_report')]


def audit(run,config):
    out=run / '01_audit';out.mkdir(exist_ok=True)
    json_write(out / 'source_hashes.json',{rel:sha(PROJECT / rel) for rel in config['sources'].values()})
    prior={}
    for relative,report_dir in PRIOR_RUNS:
        manifest=read_json(PROJECT / relative / report_dir / 'artifact_manifest.json')
        for path,record in manifest.items():
            full=PROJECT / relative / path
            assert sha(full)==record['sha256'],'Prior artifact changed: '+str(full)
            prior[str(full.relative_to(PROJECT))]=record['sha256']
    json_write(out / 'preserved_artifact_hashes.json',prior)
    json_write(out / 'code_hashes.json',code_hashes())
    env=versions();env['packages']['catboost']=md.version('catboost');json_write(out / 'environment.json',env)
    for folder,files in [('documentation',[ROOT / 'README.md',ROOT / 'FEATURE_CATALOG.md',ROOT / 'docs/METHODOLOGY.md',ROOT / 'docs/STAGES.md']),
        ('source_code',list(ROOT.glob('*.py'))+list((ROOT / 'citylib').glob('*.py')))]:
        for path in files:
            dest=out / folder / (path.relative_to(ROOT) if folder=='source_code' else path.name)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    posts=normalize_posts(csv_read(source(run,'posts')))
    assert len(posts)==1369,'This registered run uses the final 1,369-post source.'
    flags=csv_read(source(run,'equipment'))
    linked=posts[['event_id','raw_text']].merge(flags[['event_id','raw_text','equipment_fault_stated']],on='event_id',validate='one_to_one')
    assert len(linked)==len(posts) and linked.raw_text_x.equals(linked.raw_text_y)
    assert linked.equipment_fault_stated.isin([0,1]).all()
    panel,evidence=make_labels(posts)
    csv_write(panel,out / 'city_day_labels.csv');csv_write(evidence,out / 'city_outage_evidence.csv')
    annual=panel.assign(year=panel.date.str[:4]).groupby('year').agg(calendar_days=('date','count'),
        positive_days=('outage_label','sum'),supporting_posts=('supporting_post_count','sum')).reset_index()
    csv_write(annual,out / 'annual_label_support.csv')
    assert panel.outage_label.sum()==posts.date.nunique()
    report={'source_posts':len(posts),'calendar_days':len(panel),'positive_city_days':int(panel.outage_label.sum()),
        'first_date':panel.date.min(),'last_date':panel.date.max(),
        'publication_fallback_posts':int(evidence.publication_fallback.sum()),
        'fallback_only_positive_days':int(panel.fallback_only_positive.sum()),
        'missing_recloser_posts':int(posts.recloser_clean.eq('').sum()),
        'missing_barangay_posts':int(posts.barangay_clean.eq('').sum()),
        'known_date_conflict_posts':int(evidence.date_conflict.sum()),
        'all_retained_posts_support_city_labels':True,'negative_means_verified_power':False,
        'preserved_prior_artifacts':len(prior)}
    json_write(out / 'report.json',report);return report


def features(run,config):
    out=run / '02_features';out.mkdir(exist_ok=True)
    weather=csv_read(source(run,'weather'));rain=csv_read(source(run,'rainfall'))
    manifest=read_json(source(run,'rainfall_manifest'))
    assert manifest['complete'] and manifest['timezone']=='Asia/Manila'
    assert manifest['output_sha256']==sha(source(run,'rainfall'))
    assert len(rain)==len(weather)
    registry=weather_registry(weather)
    assert len(registry)==58 and np.isclose(registry.equal_weight.sum(),1)
    daily=aggregate_weather(weather,rain,registry,config)
    vegetation=aggregate_vegetation(csv_read(source(run,'vegetation')),registry)
    panel=csv_read(run / '01_audit/city_day_labels.csv');evidence=csv_read(run / '01_audit/city_outage_evidence.csv')
    f=make_features(panel,daily,vegetation,evidence,csv_read(source(run,'equipment')),config)
    assert f[FEATURES].select_dtypes('number').apply(lambda s:np.isinf(s).any()).eq(False).all()
    csv_write(registry,out / 'fixed_weather_barangay_registry.csv');csv_write(daily,out / 'city_daily_weather.csv')
    csv_write(vegetation,out / 'city_monthly_vegetation.csv');csv_write(f,out / 'city_feature_panel.csv')
    csv_write(dictionary(),out / 'feature_dictionary.csv')
    report={'conceptual_inputs':len(FEATURES),'family_counts':{k:len(v) for k,v in FAMILIES.items()},
        'weather_barangays':len(registry),'weather_dates':len(daily),'equal_barangay_weights':True,
        'weather_source_is_full_city_boundary_coverage':False,
        'vegetation_missing_fraction':{n:float(f[n].isna().mean()) for n in FAMILIES['full'] if n.startswith('veg') or n=='ndvi_barangay_mean'},
        'weather_source':'historical gridded barangay-coordinate series; 58 available barangays',
        'no_spatial_rainfall_sum':True,'hourly_predictor_rows':False,
        'new_network_downloads':False,'current_day_text_or_causes_used_as_predictors':False}
    json_write(out / 'report.json',report);return report


def cohorts(run,config):
    out=run / '03_cohorts';out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/city_feature_panel.csv');e=csv_read(run / '01_audit/city_outage_evidence.csv')
    rows=[]
    for origin in config['development_origins']+[config['test_start']]:
        defs=definition(origin)
        if origin==config['test_start']: defs['outer'].update(end=config['test_end'],asof=config['truth_asof'])
        indices={role:index_role(f,e,defs[role],'all',config) for role in ['fit','tune','refit','calibrate','outer']}
        for role,idx in indices.items():
            g=f.iloc[idx]
            initial=defs[role]['start'] or config['warmup_start']
            candidate=f[f.date.between(initial,defs[role]['end'])]
            rows.append({'origin':origin,'role':role,'first_date':g.date.min(),'last_date':g.date.max(),
                'rows':len(g),'positives':int(g.outage_label.sum()),'prevalence':float(g.outage_label.mean()),
                'late_publication_positive_rows_excluded':len(candidate)-len(g),'truth_asof':defs[role]['asof']})
        for earlier,later in [('fit','tune'),('refit','calibrate'),('calibrate','outer')]:
            a,b=f.iloc[indices[earlier]],f.iloc[indices[later]]
            assert set(a.date).isdisjoint(set(b.date))
            assert (pd.Timestamp(b.date.min())-pd.Timestamp(a.date.max())).days>=15
        for role in ['fit','tune','refit','calibrate']:
            assert f.iloc[indices[role]].date.max()<origin
        np.savez_compressed(out / (origin+'.npz'),**indices)
        json_write(out / (origin+'.json'),defs)
    csv_write(pd.DataFrame(rows),out / 'cohort_summary.csv')
    json_write(out / 'policy.json',{'development_year':2025,'maturity_gap_days':14,
        'source_warmup_days':30,'negative_location_mask':False,
        'late_only_positive_training_days':'excluded when all supporting notices were published after role as-of; counts reported',
        '2026_is_exploratory':True,'day_ahead_operational_test':False})
    return {'origins':5,'2026_rows_used_for_fitting_or_selection':False}


def evaluate_bundle(bundle,c,config,base_id,fold=None):
    kind=bundle['kind'];family=bundle.get('family','history_calendar')
    modes=['raw'] if kind in ['frequency','seasonal_frequency'] else config['probability_modes']
    records=[];warnings=[];pred=c['outer'][['row_id','date','outage_label']].copy()
    for mode in modes:
        name=base_id+'__'+mode
        p,w=predict(bundle,c['outer'],mode,config['primary_warning_policy'])
        pred[name+'_probability']=p
        for policy in config['warning_policies']:
            _,decision=predict(bundle,c['outer'],mode,policy)
            pred[name+'_'+policy+'_warning']=decision
            warnings.append({'fold':fold,'model_id':name,'policy':policy,**metrics(c['outer'].outage_label,p,decision)})
        records.append({'fold':fold,'model_id':name,'base_id':base_id,'kind':kind,'family':family,'mode':mode,
            **metrics(c['outer'].outage_label,p,w)})
    return records,warnings,pred


def procedures(config):
    return [(kind,family,kind+'__'+family) for kind in config['algorithms'] for family in config['families']]+[
        ('frequency','history_calendar','frequency'),('seasonal_frequency','history_calendar','seasonal_frequency')]


def development(run,config):
    out=run / '04_development';out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/city_feature_panel.csv');records=[];warnings=[]
    for origin in config['development_origins']:
        c=get_cohorts(f,run / '03_cohorts' / (origin+'.npz'))
        for kind,family,name in procedures(config):
            print('Development: '+origin+' '+name,flush=True)
            folder=out / origin / name
            if kind in ['frequency','seasonal_frequency']: bundle=baseline_bundle(c,config,kind,folder)
            else:
                chosen=search(c,config,kind,family,folder / 'search')
                bundle=fit_final(c,config,kind,family,chosen,folder)
            scored,operating,pred=evaluate_bundle(bundle,c,config,name,origin)
            records+=scored;warnings+=operating
            csv_write(pred,folder / 'outer_predictions.csv');csv_write(pd.DataFrame(scored),folder / 'outer_metrics.csv')
            csv_write(pd.DataFrame(records),out / 'development_metrics.csv')
            csv_write(pd.DataFrame(warnings),out / 'development_warning_metrics.csv')
            print('  '+', '.join(f"{r['mode']}: AP={r['average_precision']:.3f}, loss={r['log_loss']:.3f}" for r in scored),flush=True)
    assert len(records)==80
    return {'learned_initial_tuning_trials':144,'refitted_learned_models':36,'scored_procedures':80,
        '2026_rows_used':False}


def selection(run,config):
    out=run / '05_selection';out.mkdir(exist_ok=True)
    d=csv_read(run / '04_development/development_metrics.csv')
    ranked,main=select(d);csv_write(ranked,out / 'ranked_procedures.csv')
    ranker=ranked.sort_values(['mean_average_precision','mean_log_loss','model_id'],ascending=[False,True,True]).iloc[0].to_dict()
    full=ranked[ranked.family.eq('full')].iloc[0].to_dict()
    baseline=d[d.kind.isin(['frequency','seasonal_frequency'])].groupby('model_id')[['log_loss','average_precision','roc_auc','brier_score']].mean().reset_index()
    csv_write(baseline,out / 'baseline_development_means.csv')
    differences=[]
    for kind in config['algorithms']:
        for mode in config['probability_modes']:
            sub=d[d.kind.eq(kind)&d['mode'].eq(mode)].pivot(index='fold',columns='family',values=['log_loss','average_precision'])
            for fold in sub.index:
                for label,base,added in [('weather','history_calendar','history_weather'),('vegetation_interactions','history_weather','full')]:
                    differences.append({'kind':kind,'mode':mode,'fold':fold,'addition':label,
                        **{metric+'_difference':float(sub.loc[fold,(metric,added)]-sub.loc[fold,(metric,base)]) for metric in ['log_loss','average_precision']}})
    csv_write(pd.DataFrame(differences),out / 'paired_feature_comparisons.csv')
    choice={'main':main,'ranking_alternative':ranker,'best_full_candidate':full,
        'rule':'minimum mean 2025 outer-quarter log loss among learned algorithm/family/probability-mode procedures; AP then model ID break ties',
        'warning_policy':config['primary_warning_policy'],'2026_rows_used':False,
        'hypothesis_motivated_by_previously_seen_2026':True}
    json_write(out / 'selection.json',choice);return choice


def final_fitting(run,config):
    out=run / '06_final_models';out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/city_feature_panel.csv')
    c=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))
    training={role:c[role] for role in ['fit','tune','refit','calibrate']}
    assert all(g.date.max()<'2026-01-01' for g in training.values())
    specs={}
    for kind,family,name in procedures(config):
        print('Final fitting: '+name,flush=True);folder=out / name
        if kind in ['frequency','seasonal_frequency']: baseline_bundle(training,config,kind,folder)
        else:
            selected=search(training,config,kind,family,folder / 'search')
            fit_final(training,config,kind,family,selected,folder)
        specs[name]={'kind':kind,'family':family,'path':str((folder / 'model.joblib').relative_to(run))}
    json_write(out / 'models.json',specs)
    return {'saved_base_models':len(specs),'learned_tuning_trials':36,'2026_rows_used':False}


def bootstrap(frame,probabilities,selected,config):
    calendar=pd.date_range(frame.date.min(),frame.date.max(),freq='D').strftime('%Y-%m-%d').to_numpy()
    lookup={d:np.flatnonzero(frame.date.to_numpy()==d) for d in calendar}
    rng=np.random.default_rng(config['seed']);width=config['bootstrap_block_days'];draws=[]
    for repeat in range(config['bootstrap_replicates']):
        starts=rng.integers(0,len(calendar)-width+1,size=int(np.ceil(len(calendar)/width)))
        days=np.concatenate([calendar[s:s+width] for s in starts])[:len(calendar)]
        take=np.concatenate([lookup[d] for d in days]);y=frame.outage_label.to_numpy()[take]
        if len(np.unique(y))<2:continue
        scores={name:metrics(y,p[take]) for name,p in probabilities.items()}
        for name,score in scores.items():
            draws.append({'repeat':repeat,'comparison':name,**{k:score[k] for k in ['average_precision','roc_auc','log_loss','brier_score']}})
        for name in scores:
            if name==selected:continue
            draws.append({'repeat':repeat,'comparison':'main_minus_'+name,
                **{k:scores[selected][k]-scores[name][k] for k in ['average_precision','roc_auc','log_loss','brier_score']}})
    rows=[]
    for name,g in pd.DataFrame(draws).groupby('comparison'):
        for metric in ['average_precision','roc_auc','log_loss','brier_score']:
            lo,hi=g[metric].quantile([.025,.975])
            rows.append({'comparison':name,'metric':metric,'lower95':float(lo),'upper95':float(hi),
                'replicates':len(g),'method':'paired seven-calendar-day moving-block bootstrap; exploratory'})
    return pd.DataFrame(rows)


def evaluation(run,config):
    out=run / '07_exploratory_2026';out.mkdir(exist_ok=True)
    f=csv_read(run / '02_features/city_feature_panel.csv')
    frame=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))['outer'].reset_index(drop=True)
    c={'outer':frame};specs=read_json(run / '06_final_models/models.json');choice=read_json(run / '05_selection/selection.json')
    records=[];warnings=[];pred=frame[['row_id','date','outage_label']].copy()
    for name,info in specs.items():
        bundle=joblib.load(run / info['path'])
        scored,operating,block=evaluate_bundle(bundle,c,config,name)
        records+=scored;warnings+=operating
        pred=pred.merge(block,on=['row_id','date','outage_label'],validate='one_to_one')
    csv_write(pd.DataFrame(records),out / 'overall_metrics.csv');csv_write(pd.DataFrame(warnings),out / 'warning_policies.csv')
    csv_write(pred,out / 'predictions.csv')
    baselines=[]
    for label,decision in [('always_no_outage',np.zeros(len(frame),dtype=int)),('always_outage',np.ones(len(frame),dtype=int))]:
        baselines.append({'rule':label,**metrics(frame.outage_label,np.repeat(.5,len(frame)),decision)})
    csv_write(pd.DataFrame(baselines),out / 'constant_decision_reference.csv')
    selected=choice['main']['model_id']
    compare=list(dict.fromkeys([selected,choice['best_full_candidate']['model_id'],choice['ranking_alternative']['model_id'],
        'frequency__raw','seasonal_frequency__raw']))
    probabilities={name:pred[name+'_probability'].to_numpy() for name in compare}
    csv_write(bootstrap(frame,probabilities,selected,config),out / 'block_bootstrap_intervals.csv')
    sensitivity=[]
    reliable=frame.fallback_only_positive.eq(0)&frame.date_conflict_post_count.eq(0)
    for row in records:
        name=row['model_id'];p=pred.loc[reliable,name+'_probability']
        w=pred.loc[reliable,name+'_'+config['primary_warning_policy']+'_warning']
        sensitivity.append({'model_id':name,**metrics(frame.loc[reliable,'outage_label'],p,w),
            'policy':'exclude fallback-only/conflicting positive dates; no retraining or threshold changes'})
    csv_write(pd.DataFrame(sensitivity),out / 'date_quality_sensitivity.csv')
    from .reporting import figures
    figures(frame,probabilities,out / 'figures')
    report={'main':selected,'rows':len(frame),'positive_city_days':int(frame.outage_label.sum()),
        'first_date':frame.date.min(),'last_date':frame.date.max(),'models':{r['model_id']:r for r in records},
        '2026_rows_used_for_selection':False,'independent_untouched_test':False,
        'operational_day_ahead_test':False,'area_location_prediction':False,
        'direct_AP_comparison_to_barangay_or_recloser_is_valid':False}
    json_write(out / 'report.json',report)
    return {'main':selected,'main_metrics':report['models'][selected],'exploratory':True}


def report(run,config):
    out=run / '08_report';out.mkdir(exist_ok=True);check_inputs(run)
    prior=read_json(run / '01_audit/preserved_artifact_hashes.json')
    for path,expected in prior.items():assert sha(PROJECT / path)==expected
    f=csv_read(run / '02_features/city_feature_panel.csv')
    c=get_cohorts(f,run / '03_cohorts' / (config['test_start']+'.npz'))
    frame=c['outer'].reset_index(drop=True);saved=csv_read(run / '07_exploratory_2026/predictions.csv')
    specs=read_json(run / '06_final_models/models.json');errors={}
    for base,info in specs.items():
        bundle=joblib.load(run / info['path'])
        modes=['raw'] if info['kind'] in ['frequency','seasonal_frequency'] else config['probability_modes']
        if info['kind'] not in ['frequency','seasonal_frequency']:
            prep=bundle['transformer'];expected=c['refit'][prep.numeric].median().to_dict()
            assert np.allclose(list(prep.medians.values()),list(expected.values()))
            if prep.scaler is not None:
                assert np.allclose(prep.scaler.mean_,prep.numeric_values(c['refit']).mean(axis=0))
            assert bundle['calibrator'].slope>=0
            if info['kind']=='catboost':assert bundle['model'].get_all_params()['has_time']
        for mode in modes:
            name=base+'__'+mode;p,w=predict(bundle,frame,mode,config['primary_warning_policy'])
            error=float(np.abs(p-saved[name+'_probability'].to_numpy()).max())
            assert error<1e-8 and np.array_equal(w,saved[name+'_'+config['primary_warning_policy']+'_warning'])
            errors[name]=error
    assert not f.date.duplicated().any()
    assert set(f[f.outage_label.eq(1)].date)==set(csv_read(source(run,'posts')).event_date)
    for n in ['equipment_fault_days_prev7d','equipment_fault_days_prev14d']:assert n in FAMILIES['full']
    json_write(out / 'verification.json',{'preserved_prior_artifacts':len(prior),'source_and_code_hashes_verified':True,
        'model_reload_max_errors':errors,'fitting_only_medians_scaling_verified':True,
        'nonnegative_calibration_slopes_verified':True,'one_row_per_date_verified':True,
        'all_retained_post_dates_supported':True,'2026_exploratory':True})
    from .reporting import write_report
    write_report(run,config,out)
    return {'verified_base_models':len(specs),'preserved_prior_artifacts':len(prior),
        'report':'08_report/THESIS_CITY_DAY_REPORT.md'}
