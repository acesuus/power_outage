import importlib.metadata as md
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import ROOT,PROJECT,city_run,xgb_run,csv_read,csv_write,json_write,read_json,sha,code_hashes,check
from .data import load_inputs,origins,specs,role_definition,cohorts,target_month,earlier_predictions
from .modeling import fit_month,raw_predict,attach_calibration,score_month,predict,rank_procedures,choose,metrics


def audit(run,config):
    out=run / '01_audit';out.mkdir(exist_ok=True)
    hashes={}
    for base,report in [(city_run(config),'08_report'),(xgb_run(config),'06_report')]:
        manifest=read_json(base / report / 'artifact_manifest.json')
        for path,record in manifest.items():
            p=base / path;assert sha(p)==record['sha256'],str(p)
            hashes[str(p.relative_to(PROJECT))]=record['sha256']
    json_write(out / 'preserved_artifact_hashes.json',hashes)
    json_write(out / 'code_hashes.json',code_hashes());json_write(out / 'config_hash.json',{'sha256':sha(run / 'config.json')})
    json_write(out / 'environment.json',{n:md.version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','matplotlib']})
    for p in list(ROOT.glob('*.py'))+list((ROOT / 'monthlib').glob('*.py'))+[ROOT / 'README.md',ROOT / 'METHODOLOGY.md']:
        destination=out / 'snapshot' / p.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    base=city_run(config)
    for relative in ['01_audit/city_day_labels.csv','01_audit/city_outage_evidence.csv','02_features/city_feature_panel.csv',
        '02_features/feature_dictionary.csv','02_features/fixed_weather_barangay_registry.csv']:
        json_write(out / 'reused' / (relative.replace('/','__')+'.json'),{'path':str(base / relative),'sha256':sha(base / relative)})
    f=load_inputs(config)
    source_config=read_json(base / 'config.json')
    posts=csv_read(PROJECT / source_config['sources']['posts'])
    months=pd.period_range(f.date.min()[:7],f.date.max()[:7],freq='M')
    incident=posts.groupby(posts.event_date.str[:7]).size()
    published=pd.to_datetime(posts.published_at,utc=True).dt.tz_convert('Asia/Manila').dt.strftime('%Y-%m')
    pub=posts.groupby(published).size()
    rows=[]
    for month in months:
        name=str(month);g=f[f.date.str.startswith(name)]
        rows.append({'month':name,'calendar_days_in_source':len(g),'recorded_outage_days':int(g.outage_label.sum()),
            'incident_dated_posts':int(incident.get(name,0)),'published_posts':int(pub.get(name,0)),
            'positive_date_fraction':float(g.outage_label.mean()),'source_completeness_independently_verified':False})
    csv_write(pd.DataFrame(rows),out / 'monthly_reporting_audit.csv')
    json_write(out / 'report.json',{'preserved_city_and_extension_artifacts':len(hashes),'unchanged_features':36,
        'history_control_features':8,'new_weather_features':False,'source_posts':len(posts),
        'reporting_frequency_is_not_proof_of_physical_outage_frequency':True,
        'new_external_downloads':False,'2026_exploratory':True})
    return {'preserved_artifacts':len(hashes),'unchanged_full_features':36}


def prepare(run,config):
    out=run / '02_cohorts';out.mkdir(exist_ok=True);f=load_inputs(config);rows=[]
    for phase in ['development','evaluation']:
        for origin in origins(config,phase):
            for window in config['training_windows']:
                d=role_definition(origin,window,config);c=cohorts(f,d)
                target=target_month(f,origin,config,'pool' if phase=='development' else 'evaluation')
                for role,g in c.items():
                    rows.append({'phase':phase,'origin':origin,'window':window,'role':role,'rows':len(g),
                        'positives':int(g.outage_label.sum()),'first_date':g.date.min(),'last_date':g.date.max(),'asof':d['asof']})
                assert (pd.Timestamp(origin)-pd.Timestamp(c['refit'].date.max())).days>=15
                assert set(c['refit'].row_id).isdisjoint(target.row_id)
                assert set(c['fit'].row_id).isdisjoint(c['tune'].row_id)
                assert c['fit'].date.max()<c['tune'].date.min()
                membership={role:g.row_id.to_numpy(dtype=np.int64) for role,g in c.items()}
                membership['prediction']=target.row_id.to_numpy(dtype=np.int64)
                folder=out / phase / origin;folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder / (window+'.npz'),**membership);json_write(folder / (window+'.json'),d)
    csv_write(pd.DataFrame(rows),out / 'cohort_summary.csv')
    json_write(out / 'protocol.json',{'prediction_unit':'city-day','monthly_refresh':True,
        'training_windows':config['training_windows'],'calibration_months':3,'reporting_buffer_days':14,
        'development_end':config['development_end'],'selection_asof':config['selection_asof'],
        'late_only_positive_dates_excluded_by_role_asof':True,'future_month_labels_used_for_fitting':False,
        'earlier_2026_labels_may_be_used_in_later_2026_updates':True})
    return {'monthly_origins':24,'windowed_cohort_definitions':72}


def get_roles(run,config,f,phase,origin,window):
    folder=run / '02_cohorts' / phase / origin
    with np.load(folder / (window+'.npz')) as archive:
        c={role:f.iloc[archive[role]].copy() for role in ['fit','tune','refit','calibration']}
        target=f.iloc[archive['prediction']].copy()
    return c,target,read_json(folder / (window+'.json'))


def monthly_predictions(run,config,phase):
    out=run / ('03_development' if phase=='development' else '05_exploratory_2026');out.mkdir(exist_ok=True)
    f=load_inputs(config);all_scores=[];all_warnings=[];merged_by_origin={};pool_index={};fit_records=[]
    for spec in specs(config):
        name=spec['base_id']
        pool=pd.DataFrame(columns=['row_id','date','origin','raw_probability']) if phase=='development' else csv_read(run / '03_development/raw_pools' / (name+'.csv'))
        spec_predictions=[]
        print(phase+': '+name,flush=True)
        for origin in origins(config,phase):
            folder=out / 'models' / origin / name
            c,target,d=get_roles(run,config,f,phase,origin,spec['window'])
            bundle=fit_month(c,config,spec,folder)
            scored_month=phase=='evaluation' or origin>=config['development_start']
            if scored_month:
                cal=earlier_predictions(pool,c['calibration'])
                assert cal.origin.max()<origin and cal.date.max()<origin
                bundle=attach_calibration(bundle,cal,config)
                csv_write(cal,folder / 'prequential_calibration.csv')
            raw=raw_predict(bundle,target)
            current=target[['row_id','date']].copy();current['origin']=origin;current['raw_probability']=raw
            pool=current if pool.empty else pd.concat([pool,current],ignore_index=True)
            assert not pool.row_id.duplicated().any()
            joblib.dump(bundle,folder / 'model.joblib')
            if scored_month:
                score_target=target_month(f,origin,config,'development' if phase=='development' else 'evaluation')
                scores,warnings,pred=score_month(bundle,score_target,origin,config)
                all_scores.append(scores);all_warnings.append(warnings);spec_predictions.append(pred)
                json_write(folder / 'calibration_and_warnings.json',{'adjustment':bundle['adjustment'].summary(),
                    'warnings':bundle['warnings'],'asof':d['asof'],'prequential_origin_max':cal.origin.max(),
                    'calibration_end':cal.date.max(),'model_refit_end':c['refit'].date.max()})
            fit_records.append({'phase':phase,'origin':origin,**spec,'refit_rows':len(c['refit']),
                'refit_end':c['refit'].date.max(),'asof':d['asof'],'new_month_targets_used':False})
            print('  '+origin+(' scored' if scored_month else ' calibration warmup'),flush=True)
        csv_write(pool,out / 'raw_pools' / (name+'.csv'));pool_index[name]=str((out / 'raw_pools' / (name+'.csv')).relative_to(run))
        joined=pd.concat(spec_predictions,ignore_index=True)
        for origin,part in joined.groupby('origin',sort=True):
            keys=['row_id','date','outage_label','origin']
            if origin not in merged_by_origin:merged_by_origin[origin]=part.copy()
            else:merged_by_origin[origin]=merged_by_origin[origin].merge(part,on=keys,validate='one_to_one')
        csv_write(pd.concat(all_scores,ignore_index=True),out / 'monthly_metrics.csv')
    combined=pd.concat([merged_by_origin[k] for k in sorted(merged_by_origin)],ignore_index=True)
    assert not combined.row_id.duplicated().any()
    csv_write(combined,out / 'predictions.csv');csv_write(pd.concat(all_warnings,ignore_index=True),out / 'monthly_warning_policies.csv')
    csv_write(pd.DataFrame(fit_records),out / 'monthly_fitting_summary.csv');json_write(out / 'raw_pools.json',pool_index)
    return combined,pd.concat(all_scores,ignore_index=True)


def development(run,config):
    pred,monthly=monthly_predictions(run,config,'development')
    assert pred.date.max()<=config['development_end'] and pred.date.max()<config['test_start']
    assert len(monthly)==12*12*2
    return {'monthly_learned_scores':len(monthly),'development_refits_including_warmup':180,
        'tuning_trials':720,'development_dates':len(pred),'2026_rows_used':False}


def selection(run,config):
    out=run / '04_selection';out.mkdir(exist_ok=True)
    ranked=rank_procedures(csv_read(run / '03_development/monthly_metrics.csv'),csv_read(run / '03_development/predictions.csv'),config)
    main,met=choose(ranked);full,full_met=choose(ranked[ranked.family.eq('full')])
    csv_write(ranked,out / 'ranked_procedures.csv')
    probability=ranked.sort_values(['mean_monthly_log_loss','mean_monthly_AP','model_id'],ascending=[True,False,True]).iloc[0].to_dict()
    ranking=ranked.sort_values(['mean_monthly_AP','mean_monthly_log_loss','model_id'],ascending=[False,True,True]).iloc[0].to_dict()
    choice={'main':main,'development_fpr_budget_met':met,'best_full':full,'full_development_fpr_budget_met':full_met,
        'probability_alternative':probability,'AP_alternative':ranking,
        'rule':'highest pooled development recall among learned procedures with pooled development FPR <=20%; mean monthly AP, lower mean monthly loss, ID break ties',
        'fallback_if_none_feasible':'lowest pooled FPR, then highest mean monthly AP, then lowest mean monthly loss, then ID; flag budget unmet',
        '2026_rows_used':False,'selection_asof':config['selection_asof'],'2026_is_independent_test':False}
    json_write(out / 'selection.json',choice);return choice


def references(run,config,pred):
    f=load_inputs(config);base=city_run(config);xbase=xgb_run(config)
    c_static=csv_read(base / '07_exploratory_2026/predictions.csv')
    x_static=csv_read(xbase / '05_exploratory_2026/predictions.csv')
    assert pred.row_id.equals(c_static.row_id) and pred.outage_label.equals(c_static.outage_label)
    labels=['static_logistic_history','static_xgboost_full','static_xgboost_history_raw']
    for label,source,name in [(labels[0],c_static,'logistic__history_calendar__monotone_sigmoid'),
        (labels[1],x_static,'xgboost__full__monotone_sigmoid'),(labels[2],x_static,'xgboost__history_calendar__raw')]:
        pred[label+'_probability']=source[name+'_probability'].to_numpy()
        for policy in config['warning_policies']:pred[label+'_'+policy+'_warning']=source[name+'_'+policy+'_warning'].to_numpy()
    for origin,g in pred.groupby('origin'):
        c,_,_=get_roles(run,config,f,'evaluation',origin,'all')
        # The recent rate uses the same known, mature earlier calibration dates; not a new input feature.
        for label,train in [('monthly_frequency',c['refit']),('recent_frequency',c['calibration'])]:
            rate=float((train.outage_label.sum()+.5)/(len(train)+1))
            pred.loc[g.index,label+'_probability']=rate
            # Frozen 0.5 probability rule for constant-rate reference warnings.
            for policy in config['warning_policies']:pred.loc[g.index,label+'_'+policy+'_warning']=int(rate>=.5)
    return pred,labels+['monthly_frequency','recent_frequency']


def probability_metrics(y,p,w):
    from sklearn.metrics import average_precision_score,roc_auc_score
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7);w=np.asarray(w,dtype=int)
    positive=y==1;negative=~positive
    tp=int((positive&(w==1)).sum());fp=int((negative&(w==1)).sum())
    return {'average_precision':float(average_precision_score(y,p)),
        'roc_auc':float(roc_auc_score(y,p)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        'brier_score':float(np.mean((y-p)**2)),'recall':tp/max(positive.sum(),1),
        'false_positive_rate':fp/max(negative.sum(),1),'precision':tp/max(tp+fp,1)}


def intervals(pred,names,main,config):
    dates=pd.date_range(pred.date.min(),pred.date.max(),freq='D').strftime('%Y-%m-%d').to_numpy()
    groups={d:np.flatnonzero(pred.date.to_numpy()==d) for d in dates}
    rng=np.random.default_rng(config['seed']);width=config['bootstrap_block_days'];rows=[]
    for repeat in range(config['bootstrap_replicates']):
        starts=rng.integers(0,len(dates)-width+1,size=int(np.ceil(len(dates)/width)))
        sampled=np.concatenate([dates[s:s+width] for s in starts])[:len(dates)]
        take=np.concatenate([groups[d] for d in sampled]);y=pred.outage_label.to_numpy()[take]
        if len(np.unique(y))<2:continue
        scores={name:probability_metrics(y,pred[name+'_probability'].to_numpy()[take],
            pred[name+'_'+config['primary_warning_policy']+'_warning'].to_numpy()[take]) for name in names}
        for name,s in scores.items():rows.append({'repeat':repeat,'comparison':name,**s})
        for name,s in scores.items():
            if name!=main:rows.append({'repeat':repeat,'comparison':'main_minus_'+name,**{k:scores[main][k]-v for k,v in s.items()}})
    result=[]
    for name,g in pd.DataFrame(rows).groupby('comparison'):
        for metric in ['average_precision','roc_auc','log_loss','brier_score','recall','false_positive_rate','precision']:
            lo,hi=g[metric].quantile([.025,.975]);result.append({'comparison':name,'metric':metric,
                'lower95':float(lo),'upper95':float(hi),'replicates':len(g),'method':'paired seven-calendar-day moving-block bootstrap; exploratory'})
    return pd.DataFrame(result)


def evaluation(run,config):
    out=run / '05_exploratory_2026';pred,monthly=monthly_predictions(run,config,'evaluation')
    pred,reference_names=references(run,config,pred)
    choice=read_json(run / '04_selection/selection.json')
    names=monthly.model_id.drop_duplicates().tolist()+reference_names;rows=[];policy_rows=[]
    for name in names:
        p=pred[name+'_probability']
        for policy in config['warning_policies']:
            score=metrics(pred.outage_label,p,pred[name+'_'+policy+'_warning'])
            policy_rows.append({'model_id':name,'policy':policy,**score})
            if policy==config['primary_warning_policy']:rows.append({'model_id':name,**score})
    csv_write(pred,out / 'predictions.csv');csv_write(pd.DataFrame(rows),out / 'overall_metrics.csv')
    csv_write(pd.DataFrame(policy_rows),out / 'overall_warning_policies.csv')
    main=choice['main']['model_id'];full=choice['best_full']['model_id']
    compared=list(dict.fromkeys([main,full,'static_logistic_history','static_xgboost_full','recent_frequency']))
    csv_write(intervals(pred,compared,main,config),out / 'block_bootstrap_intervals.csv')
    from .reporting import figures
    figures(pred,compared,main,out / 'figures',config)
    result={'main':main,'best_full':full,'rows':len(pred),'positives':int(pred.outage_label.sum()),
        'earlier_2026_labels_used_only_after_availability_and_maturity':True,
        'future_target_month_labels_used_for_fitting':False,'2026_used_for_procedure_selection':False,
        'independent_test':False,'operational_day_ahead_test':False,'models':{r['model_id']:r for r in rows}}
    json_write(out / 'report.json',result)
    return {'rows':len(pred),'positives':int(pred.outage_label.sum()),'main':main,'main_metrics':result['models'][main]}


def report(run,config):
    out=run / '06_report';out.mkdir(exist_ok=True);check(run)
    prior=read_json(run / '01_audit/preserved_artifact_hashes.json')
    for path,expected in prior.items():assert sha(PROJECT / path)==expected
    f=load_inputs(config);max_error=0;checked=0
    final_pred=csv_read(run / '05_exploratory_2026/predictions.csv')
    for phase,folder in [('development','03_development'),('evaluation','05_exploratory_2026')]:
        stage_pred=csv_read(run / folder / 'predictions.csv')
        for spec in specs(config):
            for origin in origins(config,phase):
                path=run / folder / 'models' / origin / spec['base_id']
                bundle=joblib.load(path / 'model.joblib');c,target,d=get_roles(run,config,f,phase,origin,spec['window'])
                prep=bundle['transformer'];expected=c['refit'][prep.numeric].median().to_dict()
                assert np.allclose(list(prep.medians.values()),list(expected.values()))
                if prep.scaler is not None:assert np.allclose(prep.scaler.mean_,prep.numeric_values(c['refit']).mean(axis=0))
                if spec['kind']=='xgboost':assert bundle['model'].get_booster().num_boosted_rounds()==bundle['selection']['trees']
                assert c['refit'].date.max()<origin and d['asof']<origin+'T00:00:00+08:00'
                assert set(c['refit'].row_id).isdisjoint(target.row_id)
                if 'adjustment' in bundle:
                    cal=csv_read(path / 'prequential_calibration.csv');assert cal.origin.max()<origin and cal.date.max()<origin
                    assert np.isclose(bundle['adjustment'].predict(cal.raw_probability).mean(),cal.outage_label.mean())
                    take=stage_pred[stage_pred.origin.eq(origin)]
                    frame=f.iloc[take.row_id.to_numpy()]
                    for mode in config['probability_modes']:
                        p,w=predict(bundle,frame,mode,config['primary_warning_policy']);name=spec['base_id']+'__'+mode
                        error=float(np.abs(p-take[name+'_probability'].to_numpy()).max());max_error=max(max_error,error)
                        assert error<1e-7 and np.array_equal(w,take[name+'_'+config['primary_warning_policy']+'_warning'])
                checked+=1
    assert checked==288 and len(final_pred)==264 and final_pred.outage_label.sum()==92
    json_write(out / 'verification.json',{'monthly_model_bundles_verified':checked,'maximum_reload_probability_error':max_error,
        'preserved_original_artifacts':len(prior),'fitting_only_medians_scaling_and_boosting_rounds_verified':True,
        'earlier_out_of_sample_calibration_verified':True,'no_future_month_labels_in_fitting':True,
        'earlier_2026_data_used_for_later_monthly_updates':True,'same_264_dates_and_92_positives_as_static_comparison':True})
    from .reporting import write_report
    write_report(run,config,out)
    return {'verified_monthly_models':checked,'preserved_original_artifacts':len(prior)}
