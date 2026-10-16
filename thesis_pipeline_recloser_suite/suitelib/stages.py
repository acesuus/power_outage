import importlib.metadata
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, PROJECT, csv_read, csv_write, json_write, read_json, sha, code_hashes
from .data import (FAMILIES, build, load, origins, monthly_roles, target_month,
                   frozen_roles, frozen_specs, monthly_specs, eligible, known, calibration_rows)
from .modeling import fit_model, raw_predict, attach, score, metrics


def audit(run, config):
    out = run / '01_audit'; out.mkdir(exist_ok=True)
    prior = {}
    references = [
        (config['reference_run'], '08_report'),
        ('thesis_pipeline_city/runs/city_baseline_first', '08_report'),
        ('thesis_pipeline_city/xgboost_extension/runs/city_xgboost_first', '06_report'),
        ('thesis_pipeline_city_monthly/runs/city_monthly_first', '06_report'),
        ('thesis_pipeline_city_features/runs/feature_groups_first', '06_report'),
        ('thesis_pipeline_city_frozen_features/runs/frozen_features_verified', '07_report')]
    for base, report in references:
        for relative, record in read_json(PROJECT / base / report / 'artifact_manifest.json').items():
            path = PROJECT / base / relative
            assert sha(path) == record['sha256'], str(path)
            prior[str(path.relative_to(PROJECT))] = record['sha256']
    json_write(out / 'preserved_artifacts.json', prior)
    reference = PROJECT / config['reference_run']
    source_config = read_json(reference / 'config.json')
    paths = [PROJECT / p for p in source_config['sources'].values()]
    paths += list((reference / '03_cohorts').glob('*'))
    paths += [reference / relative for relative in ['config.json', '01_audit/recloser_outage_evidence.csv',
              '02_features/recloser_feature_panel.csv', '02_features/recloser_barangay_proxy_mapping.csv',
              '02_features/mapping_source_evidence.csv', '02_features/feature_dictionary.csv']]
    for directory in ['thesis_pipeline/lib', 'thesis_pipeline_v2/v2lib', 'thesis_pipeline_v3/v3lib',
                      'thesis_pipeline_recloser/rclib', 'thesis_pipeline_city/citylib',
                      'thesis_pipeline_city_monthly/monthlib', 'thesis_pipeline_city_features/featurelib']:
        paths += list((PROJECT / directory).glob('*.py'))
    json_write(out / 'input_hashes.json', {str(p.relative_to(PROJECT)): sha(p) for p in sorted(set(paths))})
    json_write(out / 'code_hashes.json', code_hashes())
    json_write(out / 'config_hash.json', {'sha256': sha(run / 'config.json')})
    for relative in code_hashes():
        target = out / 'snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    json_write(out / 'environment.json', {n: importlib.metadata.version(n) for n in
               ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'joblib', 'matplotlib']})
    for filename in ['annual_support.csv', 'report.json', 'assignment_review.csv']:
        shutil.copy2(reference / '01_audit' / filename, out / filename)
    return dict(preserved_artifacts=len(prior), source_posts=1369, labels_changed=False, new_downloads=False)


def prepare(run, config):
    from citylib.data import feature_description
    from featurelib.features import dictionary as city_extra_dictionary
    out = run / '02_features'; out.mkdir(exist_ok=True)
    frame, spatial = build(config)
    csv_write(frame, out / 'recloser_feature_panel.csv')
    csv_write(spatial, out / 'spatial_feature_audit.csv')
    json_write(out / 'families.json', FAMILIES)
    extra = city_extra_dictionary().set_index('variable').to_dict('index')
    catalog = []
    for n in dict.fromkeys(n for fields in FAMILIES.values() for n in fields):
        if n == 'recloser_id':
            definition, unit, timing = 'Named recloser service-area category; not a geographic coordinate', 'category', 'fixed'
        elif n in extra:
            definition = extra[n]['definition'].replace('city', 'mapped recloser area').replace('58 sites', 'mapped sites')
            unit, timing = extra[n]['unit'].replace('58 sites', 'mapped sites'), extra[n]['timing']
        else:
            definition, unit, timing = 'Original 88-input recloser feature; see preserved FEATURE_CATALOG.md', 'original source units', 'original timing'
        if n in ['wet_days_prev3d', 'wet_days_prev7d']:
            definition = 'Prior days with mapped-mean recloser rainfall >=0.1 mm; inherited original recloser definition'
        catalog.append(dict(variable=n, definition=definition, unit=unit, timing=timing,
                            families=';'.join(k for k, v in FAMILIES.items() if n in v)))
    csv_write(pd.DataFrame(catalog), out / 'feature_dictionary.csv')
    mapping = PROJECT / config['reference_run'] / '02_features/recloser_barangay_proxy_mapping.csv'
    shutil.copy2(mapping, out / mapping.name)
    rows = []
    for origin in config['development_origins'] + [config['test_start']]:
        archive = PROJECT / config['reference_run'] / '03_cohorts' / (origin+'.npz')
        target = out / 'cohorts/frozen' / archive.name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, target)
        for role, group in frozen_roles(frame, origin, config).items():
            rows.append(dict(protocol='frozen', phase='evaluation' if origin == config['test_start'] else 'development',
                             origin=origin, window='all', role=role, rows=len(group), positives=int(group.outage_label.sum()),
                             first_date=group.date.min(), last_date=group.date.max()))
    for phase in ['development', 'evaluation']:
        for origin in origins(config, phase):
            for window in ['12m', '24m', 'all']:
                roles, definition = monthly_roles(frame, origin, window, config)
                folder = out / 'cohorts/monthly' / phase / origin; folder.mkdir(parents=True, exist_ok=True)
                members = {role: g.row_id.to_numpy(dtype=np.int64) for role, g in roles.items()}
                members['pool'] = target_month(frame, origin, config, phase, full_pool=True).row_id.to_numpy(dtype=np.int64)
                np.savez_compressed(folder / (window+'.npz'), **members)
                json_write(folder / (window+'.json'), definition)
                for role, group in roles.items():
                    rows.append(dict(protocol='monthly', phase=phase, origin=origin, window=window, role=role,
                                     rows=len(group), positives=int(group.outage_label.sum()),
                                     first_date=group.date.min(), last_date=group.date.max()))
    csv_write(pd.DataFrame(rows), out / 'cohort_summary.csv')
    json_write(out / 'specifications.json', dict(frozen=frozen_specs(), monthly=monthly_specs()))
    test = frozen_roles(frame, config['test_start'], config)['outer']
    assert (len(test), int(test.outage_label.sum())) == (2098, 57)
    full = frame[frame.date.between(config['test_start'], config['test_end']) & known(frame, config['test_truth_asof'])]
    assert (len(full), int(full.outage_label.sum())) == (2376, 57)
    return dict(panel_rows=len(frame), positives=418, families={n: len(v) for n, v in FAMILIES.items()},
                frozen_procedures=27, monthly_procedures=23, primary_test_rows=2098, primary_positives=57,
                sensitivity_rows=2376, uncertainty_mask_is_retrospective=True)


def checkpoint(folder):
    marker = folder / 'COMPLETE.json'
    if not marker.exists(): return False
    for relative, expected in read_json(marker).items(): assert sha(folder / relative) == expected, str(folder)
    return True


def complete(folder):
    json_write(folder / 'COMPLETE.json', {p.name: sha(p) for p in folder.iterdir()
               if p.is_file() and p.name != 'COMPLETE.json'})


def frozen_development(run, config):
    out = run / '03_frozen_development'; out.mkdir(exist_ok=True)
    frame, all_scores, per_origin = load(run), [], []
    for origin in config['development_origins']:
        roles = frozen_roles(frame, origin, config)
        parts = []
        for spec in frozen_specs():
            folder = out / origin / spec['base_id']
            print('Frozen development '+origin+' '+spec['base_id'], flush=True)
            if checkpoint(folder):
                part, records = csv_read(folder / 'predictions.csv'), read_json(folder / 'scores.json')
            else:
                bundle = fit_model(roles, config, spec, folder, 'frozen')
                attach(bundle, roles['calibrate'], config)
                part, records = score(bundle, roles['outer'], origin, config)
                joblib.dump(bundle, folder / 'model.joblib')
                csv_write(part, folder / 'predictions.csv'); json_write(folder / 'scores.json', records)
                csv_write(roles['calibrate'][['row_id', 'date', 'recloser', 'outage_label']].assign(
                    raw_probability=raw_predict(bundle, roles['calibrate'])), folder / 'calibration.csv')
                json_write(folder / 'calibration_and_warnings.json', dict(calibrator=bundle['calibrator'].summary(), warnings=bundle['warnings']))
                complete(folder)
            parts.append(part); all_scores += records
        per_origin.append(merge_parts(parts))
        csv_write(pd.DataFrame(all_scores), out / 'metrics.csv')
    csv_write(pd.concat(per_origin, ignore_index=True), out / 'predictions.csv')
    return dict(fitted_models=108, inner_trials=432, scored_probability_procedures=54, target_year_used=False)


def merge_parts(parts):
    keys = ['row_id', 'date', 'recloser', 'outage_label', 'origin']
    result = parts[0].set_index(keys)
    for part in parts[1:]: result = result.join(part.set_index(keys), how='inner', validate='one_to_one')
    return result.reset_index()


def frozen_select_fit(run, config):
    out = run / '04_frozen_selection_fit'; out.mkdir(exist_ok=True)
    data = csv_read(run / '03_frozen_development/metrics.csv')
    primary = data[data.policy == config['primary_warning_policy']]
    ranked = primary.groupby(['model_id', 'base_id', 'kind', 'family', 'window', 'mode']).agg(
        mean_log_loss=('log_loss', 'mean'), mean_AP=('average_precision', 'mean'),
        mean_AUC=('roc_auc', 'mean'), mean_recall=('recall', 'mean'), mean_FPR=('false_positive_rate', 'mean'),
        folds=('origin', 'nunique')).reset_index()
    assert ranked.folds.eq(4).all()
    ranked = ranked.sort_values(['mean_log_loss', 'mean_AP', 'model_id'], ascending=[True, False, True])
    csv_write(ranked, out / 'ranked_procedures.csv')
    full = ranked[ranked.family.isin(['full', 'full_local', 'full_dry', 'full_local_dry', 'legacy_full'])]
    raw = ranked[ranked['mode'] == 'raw']
    choice = dict(main=ranked.iloc[0].to_dict(), best_full=full.iloc[0].to_dict(), best_raw=raw.iloc[0].to_dict(),
                  AP_alternative=ranked.sort_values(['mean_AP', 'mean_log_loss', 'model_id'], ascending=[False, True, True]).iloc[0].to_dict(),
                  rule='Minimum mean 2025 outer-quarter probability loss; higher AP then model ID ties',
                  evaluated_2026_to_choose=False, independent_test=False)
    json_write(out / 'selection.json', choice)
    roles = frozen_roles(load(run), config['test_start'], config)
    training = {name: roles[name] for name in ['fit', 'tune', 'refit', 'calibrate']}
    assert all(g.date.max() < config['test_start'] for g in training.values())
    for spec in frozen_specs():
        folder = out / 'models' / spec['base_id']
        print('Frozen final '+spec['base_id'], flush=True)
        if checkpoint(folder): continue
        bundle = fit_model(training, config, spec, folder, 'frozen')
        attach(bundle, training['calibrate'], config)
        joblib.dump(bundle, folder / 'model.joblib')
        csv_write(training['calibrate'][['row_id', 'date', 'recloser', 'outage_label']].assign(
            raw_probability=raw_predict(bundle, training['calibrate'])), folder / 'calibration.csv')
        json_write(folder / 'calibration_and_warnings.json', dict(calibrator=bundle['calibrator'].summary(), warnings=bundle['warnings']))
        complete(folder)
    return dict(choice=choice, final_models=27, inner_trials=108, final_fitting_uses_2026=False)


def monthly_data(run, frame, origin, phase, window):
    folder = run / '02_features/cohorts/monthly' / phase / origin
    with np.load(folder / (window+'.npz')) as archive:
        roles = {role: frame.iloc[archive[role]].copy() for role in ['fit', 'tune', 'refit', 'calibration']}
        pool_target = frame.iloc[archive['pool']].copy()
    return roles, pool_target, read_json(folder / (window+'.json'))


def monthly_predictions(run, config, phase):
    out = run / ('05_monthly_development' if phase == 'development' else '06_monthly_evaluation')
    out.mkdir(exist_ok=True)
    frame, all_scores, parts_by_origin, fitting = load(run), [], {}, []
    for spec in monthly_specs():
        name = spec['base_id']
        pool = pd.DataFrame(columns=['row_id', 'date', 'origin', 'raw_probability']) if phase == 'development' else csv_read(run / '05_monthly_development/raw_pools' / (name+'.csv'))
        print('Monthly '+phase+' '+name, flush=True)
        for origin in origins(config, phase):
            folder = out / 'models' / origin / name
            roles, pool_target, definition = monthly_data(run, frame, origin, phase, spec['window'])
            scored = phase == 'evaluation' or origin >= config['development_start']
            if checkpoint(folder):
                current = csv_read(folder / 'raw_predictions.csv')
                if scored:
                    part = csv_read(folder / 'predictions.csv'); records = read_json(folder / 'scores.json')
            else:
                bundle = fit_model(roles, config, spec, folder, 'monthly')
                if scored:
                    cal = calibration_rows(pool, roles['calibration'], origin, config['minimum_calibration_rows'])
                    attach(bundle, cal, config, raw=cal.raw_probability)
                    csv_write(cal, folder / 'prequential_calibration.csv')
                    target = target_month(frame, origin, config, phase)
                    part, records = score(bundle, target, origin, config)
                    csv_write(part, folder / 'predictions.csv'); json_write(folder / 'scores.json', records)
                    json_write(folder / 'calibration_and_warnings.json', dict(calibrator=bundle['calibrator'].summary(),
                        warnings=bundle['warnings'], asof=definition['asof'], refit_end=roles['refit'].date.max(),
                        calibration_end=cal.date.max(), max_prediction_origin=cal.origin.max()))
                current = pool_target[['row_id', 'date']].copy()
                current['origin'], current['raw_probability'] = origin, raw_predict(bundle, pool_target)
                csv_write(current, folder / 'raw_predictions.csv')
                joblib.dump(bundle, folder / 'model.joblib'); complete(folder)
            pool = current if pool.empty else pd.concat([pool, current], ignore_index=True)
            assert pool.row_id.is_unique
            if scored:
                all_scores += records; parts_by_origin.setdefault(origin, []).append(part)
            fitting.append(dict(origin=origin, **spec, refit_rows=len(roles['refit']),
                                refit_end=roles['refit'].date.max(), target_month_used=False))
            print('  '+origin+(' scored' if scored else ' warmup'), flush=True)
        csv_write(pool, out / 'raw_pools' / (name+'.csv'))
        csv_write(pd.DataFrame(all_scores), out / 'monthly_metrics.csv')
        csv_write(pd.DataFrame(fitting), out / 'fitting_summary.csv')
    combined = pd.concat([merge_parts(parts_by_origin[o]) for o in sorted(parts_by_origin)], ignore_index=True)
    assert combined.row_id.is_unique
    if phase == 'evaluation':
        expected = frozen_roles(frame, config['test_start'], config)['outer']
        np.testing.assert_array_equal(combined.row_id, expected.row_id)
        assert (len(combined), int(combined.outage_label.sum())) == (2098, 57)
    csv_write(combined, out / 'predictions.csv')
    return combined, pd.DataFrame(all_scores)


def monthly_development(run, config):
    pred, monthly = monthly_predictions(run, config, 'development')
    assert pred.date.max() <= config['development_end']
    return dict(fitted_models=345, inner_trials=1380, scored_rows=len(pred), positives=int(pred.outage_label.sum()))


def aggregate(pred, monthly, config):
    rows = []
    for name, group in monthly.groupby('model_id'):
        reference = group.iloc[0]
        for policy in config['warning_policies']:
            g = group[group.policy == policy]
            rows.append(dict(model_id=name, base_id=reference.base_id, kind=reference.kind, family=reference.family,
                window=reference.window, mode=reference['mode'], policy=policy,
                mean_monthly_AP=float(g.average_precision.mean()), mean_monthly_loss=float(g.log_loss.mean()),
                **metrics(pred.outage_label, pred[name+'_probability'], pred[name+'_'+policy+'_warning'])))
    return pd.DataFrame(rows)


def choose_warning(ranked):
    feasible = ranked[ranked.development_budget_met]
    if len(feasible): return feasible.iloc[0].to_dict(), True
    fallback = ranked.sort_values(['false_positive_rate', 'mean_monthly_AP', 'mean_monthly_loss', 'model_id'],
                                  ascending=[True, False, True, True])
    return fallback.iloc[0].to_dict(), False


def monthly_select_evaluate(run, config):
    out = run / '06_monthly_evaluation'; out.mkdir(exist_ok=True)
    pred = csv_read(run / '05_monthly_development/predictions.csv')
    monthly = csv_read(run / '05_monthly_development/monthly_metrics.csv')
    scores = aggregate(pred, monthly, config)
    csv_write(scores, out / 'development_overall_metrics.csv')
    primary = scores[scores.policy == config['primary_warning_policy']].copy()
    primary['development_budget_met'] = primary.false_positive_rate.le(.2+1e-12)
    ranked = primary.sort_values(['development_budget_met', 'recall', 'mean_monthly_AP', 'mean_monthly_loss', 'model_id'],
                                 ascending=[False, False, False, True, True])
    selected, budget = choose_warning(ranked)
    csv_write(ranked, out / 'ranked_development.csv')
    # Explicit feature-screen analogue to the city monthly feature experiment.
    features = primary[(primary.kind == 'xgboost') & (primary.window == '12m') &
                       (primary['mode'] == 'raw') & (primary.family != 'legacy_full')].copy()
    ref_name = 'xgboost__full__12m__raw'
    base_months = monthly[(monthly.model_id == ref_name) & (monthly.policy == 'fpr20')]
    counts = {}
    for name in features.model_id:
        part = monthly[(monthly.model_id == name) & (monthly.policy == 'fpr20')]
        pair = part.merge(base_months, on='origin', suffixes=('', '_full'), validate='one_to_one')
        counts[name] = int((pair.average_precision > pair.average_precision_full+1e-12).sum())
    features['AP_better_months'] = features.model_id.map(counts)
    reference = features[features.model_id == ref_name].iloc[0]
    features['replacement_qualified'] = (features.development_budget_met & features.AP_better_months.ge(7) &
                                         features.mean_monthly_loss.le(reference.mean_monthly_loss))
    eligible_features = features[features.replacement_qualified]
    feature_choice = (eligible_features.sort_values(['recall', 'mean_monthly_AP', 'mean_monthly_loss', 'model_id'],
                    ascending=[False, False, True, True]).iloc[0] if len(eligible_features) else reference).to_dict()
    csv_write(features, out / 'feature_development_comparison.csv')
    choice = dict(main=selected, development_budget_met=budget,
                  probability_alternative=primary.sort_values(['mean_monthly_loss', 'mean_monthly_AP', 'model_id'],
                    ascending=[True, False, True]).iloc[0].to_dict(),
                  AP_alternative=primary.sort_values(['mean_monthly_AP', 'mean_monthly_loss', 'model_id'],
                    ascending=[False, True, True]).iloc[0].to_dict(),
                  feature_choice=feature_choice, feature_replacement_qualified=bool(len(eligible_features)),
                  rule='Highest pooled 2025 recall with pooled FPR<=20%; higher mean monthly AP, lower loss, ID ties. '
                       'If none feasible lowest FPR then AP/loss/ID; flag allowance unmet.',
                  feature_rule='XGBoost 12m raw feature replacement: FPR<=20%, AP better than full in >=7/12 months, '
                               'no worse mean monthly loss; highest recall, AP, lower loss, ID ties. Else keep full reference.',
                  evaluated_2026_to_choose=False, independent_test=False)
    json_write(out / 'selection.json', choice)
    choice_hash = sha(out / 'selection.json')
    final, months = monthly_predictions(run, config, 'evaluation')
    csv_write(aggregate(final, months, config), out / 'overall_metrics.csv')
    # Sensitivity keeps all no-record rows; models/calibration/warnings do not change.
    frame, sensitivity_parts = load(run), {}
    per_recloser = []
    thresholds = []
    for origin in origins(config, 'evaluation'):
        target = target_month(frame, origin, config, 'evaluation', sensitivity=True)
        blocks = []
        for spec in monthly_specs():
            folder = out / 'models' / origin / spec['base_id']
            bundle = joblib.load(folder / 'model.joblib')
            part, _ = score(bundle, target, origin, config); blocks.append(part)
            for mode, rules in bundle['warnings'].items():
                for policy, rule in rules.items():
                    thresholds.append(dict(origin=origin, base_id=spec['base_id'], mode=mode, policy=policy,
                                           threshold=rule['threshold']))
        sensitivity_parts[origin] = merge_parts(blocks)
    sensitivity = pd.concat([sensitivity_parts[o] for o in sorted(sensitivity_parts)], ignore_index=True)
    assert (len(sensitivity), int(sensitivity.outage_label.sum())) == (2376, 57)
    csv_write(sensitivity, out / 'sensitivity_all_no_record_predictions.csv')
    csv_write(pd.DataFrame(thresholds), out / 'monthly_thresholds.csv')
    rows = []
    for name in months.model_id.unique():
        for policy in config['warning_policies']:
            rows.append(dict(model_id=name, policy=policy,
                             **metrics(sensitivity.outage_label, sensitivity[name+'_probability'],
                                       sensitivity[name+'_'+policy+'_warning'])))
    csv_write(pd.DataFrame(rows), out / 'sensitivity_all_no_record_metrics.csv')
    assert sha(out / 'selection.json') == choice_hash
    return dict(choice=choice, fitted_models=207, inner_trials=828, evaluated_rows=2098, positives=57,
                sensitivity_rows=2376, test_used_to_select=False)


def all_metrics(pred, spec_list, modes, config, origin):
    rows = []
    for spec in spec_list:
        for mode in modes:
            name = spec['base_id']+'__'+mode
            for policy in config['warning_policies']:
                rows.append(dict(origin=origin, model_id=name, **spec, mode=mode, policy=policy,
                                 **metrics(pred.outage_label, pred[name+'_probability'], pred[name+'_'+policy+'_warning'])))
    return pd.DataFrame(rows)


def per_unit(pred, scores, config):
    rows = []
    for name in scores.model_id.unique():
        for recloser, group in pred.groupby('recloser'):
            for policy in config['warning_policies']:
                rows.append(dict(model_id=name, recloser=recloser, policy=policy,
                    **metrics(group.outage_label, group[name+'_probability'], group[name+'_'+policy+'_warning'])))
    return pd.DataFrame(rows)


def frozen_evaluate(run, config):
    out = run / '07_frozen_evaluation'; out.mkdir(exist_ok=True)
    frame = load(run)
    target = frozen_roles(frame, config['test_start'], config)['outer'].reset_index(drop=True)
    sensitivity = frame[frame.date.between(config['test_start'], config['test_end']) & known(frame, config['test_truth_asof'])].reset_index(drop=True)
    parts, sens_parts, records, thresholds = [], [], [], []
    for spec in frozen_specs():
        bundle_path = run / '04_frozen_selection_fit/models' / spec['base_id'] / 'model.joblib'
        before = sha(bundle_path)
        bundle = joblib.load(bundle_path)
        part, scores = score(bundle, target, '2026_frozen', config)
        parts.append(part); records += scores
        sens, _ = score(bundle, sensitivity, '2026_frozen', config); sens_parts.append(sens)
        assert before == sha(bundle_path)
        for mode, rules in bundle['warnings'].items():
            for policy, rule in rules.items():
                thresholds.append(dict(base_id=spec['base_id'], mode=mode, policy=policy,
                                       threshold=rule['threshold'], calibration_slope=bundle['calibrator'].slope))
    pred, sens = merge_parts(parts), merge_parts(sens_parts)
    scored = pd.DataFrame(records)
    csv_write(pred, out / 'predictions.csv'); csv_write(scored, out / 'overall_metrics.csv')
    csv_write(sens, out / 'sensitivity_all_no_record_predictions.csv')
    csv_write(all_metrics(sens, frozen_specs(), ['raw', 'monotone_sigmoid'], config, '2026_sensitivity'),
              out / 'sensitivity_all_no_record_metrics.csv')
    csv_write(pd.DataFrame(thresholds), out / 'frozen_thresholds.csv')
    csv_write(per_unit(pred, scored, config), out / 'per_recloser_metrics.csv')
    monthly = csv_read(run / '06_monthly_evaluation/predictions.csv')
    monthly_scores = csv_read(run / '06_monthly_evaluation/overall_metrics.csv')
    csv_write(per_unit(monthly, monthly_scores, config), run / '06_monthly_evaluation/per_recloser_metrics.csv')
    # Retain original published-run references under their original F1 warning rule.
    old = csv_read(PROJECT / config['reference_run'] / '07_exploratory_2026/predictions.csv')
    np.testing.assert_array_equal(old.row_id, pred.row_id)
    legacy = []
    for column in old:
        if column.endswith('_raw_probability'):
            name = column[:-len('_raw_probability')]
            for mode, probability_column in [('raw', column), ('original_sigmoid', name+'_probability')]:
                legacy.append(dict(model_id=name+'__'+mode, policy='original_tuning_F1',
                    **metrics(old.outage_label, old[probability_column], old[name+'_warning'])))
    csv_write(pd.DataFrame(legacy), out / 'original_recloser_reference_metrics.csv')
    # Date-block intervals keep all available recloser rows together.
    from monthlib.stages import intervals
    frozen_choice = read_json(run / '04_frozen_selection_fit/selection.json')
    monthly_choice = read_json(run / '06_monthly_evaluation/selection.json')
    frozen_names = list(dict.fromkeys([frozen_choice['main']['model_id'], frozen_choice['best_full']['model_id'],
                                      frozen_choice['best_raw']['model_id'], 'xgboost__full__all__raw']))
    monthly_names = list(dict.fromkeys([monthly_choice['main']['model_id'], monthly_choice['AP_alternative']['model_id'],
                                       'xgboost__full__12m__raw', 'xgboost__history__12m__raw']))
    csv_write(intervals(pred, frozen_names, frozen_choice['main']['model_id'], config), out / 'block_bootstrap_intervals.csv')
    csv_write(intervals(monthly, monthly_names, monthly_choice['main']['model_id'], config),
              run / '06_monthly_evaluation/block_bootstrap_intervals.csv')
    # Preserve reporting-frequency and per-unit source support without interpreting quiet periods as complete.
    source = csv_read(PROJECT / config['reference_run'] / '01_audit/recloser_outage_evidence.csv')
    support = frame.assign(year=frame.date.str[:4]).groupby(['recloser', 'year']).agg(
        calendar_rows=('row_id', 'size'), positive_unit_days=('outage_label', 'sum'),
        uncertain_no_record_rows=('unlocalized_negative_uncertainty', 'sum')).reset_index()
    csv_write(support, out / 'annual_unit_label_audit.csv')
    assert (len(pred), int(pred.outage_label.sum())) == (2098, 57)
    return dict(frozen_warning_comparisons=len(scored), monthly_warning_comparisons=len(monthly_scores),
                primary_rows=2098, positives=57, sensitivity_rows=2376, original_reference_procedures=len(legacy))


def verify_report(run, config):
    from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
    from .reporting import figures, write_report
    out = run / '08_report'; out.mkdir(exist_ok=True)
    frame, verified, max_reload_error = load(run), 0, 0.
    role_cache = {}
    paths = list((run / '03_frozen_development').glob('*/*/model.joblib'))
    paths += list((run / '04_frozen_selection_fit/models').glob('*/model.joblib'))
    paths += list((run / '05_monthly_development/models').glob('*/*/model.joblib'))
    paths += list((run / '06_monthly_evaluation/models').glob('*/*/model.joblib'))
    assert len(paths) == 687
    for path in paths:
        bundle = joblib.load(path)
        folder, spec = path.parent, bundle['spec']
        if bundle['protocol'] == 'monthly':
            phase = 'development' if path.is_relative_to(run / '05_monthly_development') else 'evaluation'
            origin = folder.parent.name
            key = ('monthly', phase, origin, spec['window'])
            if key not in role_cache: role_cache[key] = monthly_data(run, frame, origin, phase, spec['window'])
            roles, target, _ = role_cache[key]
            saved = csv_read(folder / 'raw_predictions.csv')
            expected = saved.raw_probability.to_numpy()
        else:
            origin = folder.parent.name if path.is_relative_to(run / '03_frozen_development') else config['test_start']
            key = ('frozen', origin)
            if key not in role_cache: role_cache[key] = frozen_roles(frame, origin, config)
            roles = role_cache[key]
            target = roles['outer']
            saved = csv_read(folder / 'predictions.csv') if path.is_relative_to(run / '03_frozen_development') else csv_read(run / '07_frozen_evaluation/predictions.csv')
            expected = saved[spec['base_id']+'__raw_probability'].to_numpy()
        got = raw_predict(bundle, target)
        error = float(np.max(np.abs(got-expected)))
        assert error < 1e-12, str(path)
        max_reload_error = max(max_reload_error, error)
        prep = bundle['transformer']
        expected_medians = prep.augment(roles['refit'])[prep.numeric].median().to_dict()
        np.testing.assert_allclose([prep.medians[n] for n in prep.numeric],
                                   [expected_medians[n] for n in prep.numeric], rtol=1e-12, atol=1e-12)
        for raw, references in prep.p90.items():
            expected_p90 = roles['refit'].groupby('psgc')[raw].quantile(.9).clip(lower=1).to_dict()
            assert references == expected_p90
        if spec['kind'] == 'xgboost': assert bundle['model'].get_booster().num_boosted_rounds() == bundle['selection']['trees']
        if spec['kind'] == 'catboost': assert bundle['model'].tree_count_ == bundle['selection']['trees']
        if spec['kind'] in ['xgboost', 'random_forest', 'catboost']: assert prep.scaler is None
        if 'calibrator' in bundle:
            if bundle['protocol'] == 'frozen': assert bundle['calibrator'].slope >= 0
        verified += 1
    count = 0
    for relative in ['06_monthly_evaluation', '07_frozen_evaluation']:
        pred = csv_read(run / relative / 'predictions.csv')
        scores = csv_read(run / relative / 'overall_metrics.csv')
        for _, row in scores.iterrows():
            p, w = pred[row.model_id+'_probability'], pred[row.model_id+'_'+row.policy+'_warning']
            np.testing.assert_array_equal(confusion_matrix(pred.outage_label, w, labels=[0, 1]).ravel(),
                                          [row.tn, row.fp, row.fn, row.tp])
            assert np.isclose(average_precision_score(pred.outage_label, p), row.average_precision)
            assert np.isclose(roc_auc_score(pred.outage_label, p), row.roc_auc)
            count += 1
    preserved = read_json(run / '01_audit/preserved_artifacts.json')
    for relative, expected in preserved.items(): assert sha(PROJECT / relative) == expected
    figures(run)
    write_report(run, config)
    json_write(out / 'verification.json', dict(model_bundles_verified=verified, maximum_reload_error=max_reload_error,
               fitting_only_medians_p90_and_rounds_verified=True, aggregate_metrics_independently_recomputed=count,
               earlier_artifacts_preserved=len(preserved), original_recloser_labels_preserved=True,
               primary_test_rows=2098, positives=57, sensitivity_rows=2376))
    manifest = {str(p.relative_to(run)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(run.rglob('*'))
                if p.is_file() and not p.is_relative_to(run / 'status') and p.name not in ['artifact_manifest.json', 'COMPLETION.json']}
    json_write(out / 'artifact_manifest.json', manifest)
    json_write(out / 'COMPLETION.json', dict(all_eight_stages_completed=True, model_bundles=verified, artifacts=len(manifest)))
    return dict(verified_models=verified, artifacts=len(manifest), preserved_earlier_artifacts=len(preserved))
