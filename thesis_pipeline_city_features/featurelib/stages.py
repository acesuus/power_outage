import importlib.metadata
import shutil
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, PROJECT, csv_read, csv_write, json_write, read_json, sha, own_hashes, verify_parent
from .features import build, dictionary, FAMILIES, CONTROLS, NEW_FEATURES
from .modeling import fit_month, warning_rules, score, rank, choose, raw_predict, metrics
from monthlib.data import origins, earlier_predictions, target_month
from monthlib.stages import get_roles, intervals


def audit(run, config):
    out = run / '01_audit'
    out.mkdir(exist_ok=True)
    parent = PROJECT / config['parent_run']
    assert read_json(parent / '06_report/COMPLETION.json')['all_six_stages_completed']
    preserved = verify_parent(parent)
    json_write(out / 'parent_artifact_hashes.json', preserved)
    original_preserved = read_json(parent / '01_audit/preserved_artifact_hashes.json')
    for path, expected in original_preserved.items():
        assert sha(PROJECT / path) == expected, path
    json_write(out / 'earlier_artifact_hashes.json', original_preserved)
    city = PROJECT / config['city_run']
    source_config = read_json(city / 'config.json')
    paths = [PROJECT / value for value in source_config['sources'].values()]
    paths += [city / relative for relative in ['config.json', '02_features/city_feature_panel.csv',
              '02_features/fixed_weather_barangay_registry.csv', '01_audit/city_outage_evidence.csv']]
    paths += [parent / 'config.json', parent / '04_selection/selection.json']
    # Imported helpers are immutable dependencies; record their source, too.
    paths += list((PROJECT / 'thesis_pipeline_city_monthly/monthlib').glob('*.py'))
    paths += list((PROJECT / 'thesis_pipeline_city/citylib').glob('*.py'))
    paths += list((PROJECT / 'thesis_pipeline/lib').glob('*.py'))
    paths += list((PROJECT / 'thesis_pipeline_v2/v2lib').glob('*.py'))
    json_write(out / 'input_hashes.json', {str(p.relative_to(PROJECT)): sha(p) for p in sorted(set(paths))})
    json_write(out / 'code_hashes.json', own_hashes())
    json_write(out / 'config_hash.json', {'sha256': sha(run / 'config.json')})
    for relative in own_hashes():
        destination = out / 'snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    json_write(out / 'environment.json', {name: importlib.metadata.version(name) for name in
               ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'matplotlib']})
    return dict(preserved_parent_artifacts=len(preserved), preserved_earlier_artifacts=len(original_preserved),
                new_external_downloads=False, label_changes=False)


def prepare(run, config):
    out = run / '02_features'
    out.mkdir(exist_ok=True)
    frame, daily = build(config)
    csv_write(frame, out / 'city_feature_panel.csv')
    csv_write(daily, out / 'spatial_feature_audit.csv')
    csv_write(dictionary(), out / 'feature_dictionary.csv')
    json_write(out / 'families.json', FAMILIES)
    summaries = []
    parent = PROJECT / config['parent_run']
    for phase in ['development', 'evaluation']:
        for origin in origins(config, phase):
            roles, target, definition = get_roles(parent, config, frame, phase, origin, config['window'])
            for role, group in roles.items():
                assert group.date.max() < origin
                summaries.append(dict(phase=phase, origin=origin, role=role, rows=len(group),
                                      positives=int(group.outage_label.sum()), first_date=group.date.min(),
                                      last_date=group.date.max(), asof=definition['asof']))
            assert set(roles['refit'].row_id).isdisjoint(target.row_id)
            assert set(roles['fit'].row_id).isdisjoint(roles['tune'].row_id)
            assert (pd.Timestamp(origin)-pd.Timestamp(roles['refit'].date.max())).days >= 15
    csv_write(pd.DataFrame(summaries), out / 'cohort_summary.csv')
    rows = [dict(family=k, inputs=len(v), newly_trained=k not in CONTROLS) for k, v in FAMILIES.items()]
    csv_write(pd.DataFrame(rows), out / 'family_summary.csv')
    return dict(city_dates=len(frame), recorded_positive_dates=int(frame.outage_label.sum()),
                added_features=NEW_FEATURES, families=len(FAMILIES), new_families=len(FAMILIES)-len(CONTROLS))


def predict_phase(run, config, phase):
    out = run / ('03_development' if phase == 'development' else '05_exploratory_2026')
    out.mkdir(exist_ok=True)
    parent = PROJECT / config['parent_run']
    frame = csv_read(run / '02_features/city_feature_panel.csv')
    frame['first_support_utc'] = pd.to_datetime(frame.first_support_utc, utc=True)
    all_predictions, all_monthly = [], []
    for family in FAMILIES:
        print(f'{phase}: {family} ({len(FAMILIES[family])} inputs)', flush=True)
        pool = (pd.DataFrame(columns=['row_id', 'date', 'origin', 'raw_probability']) if phase == 'development'
                else csv_read(run / '03_development/raw_pools' / (family + '.csv')))
        if family in CONTROLS:
            relative = '03_development' if phase == 'development' else '05_exploratory_2026'
            parent_pool = csv_read(parent / relative / 'raw_pools' /
                                   ('xgboost__' + CONTROLS[family] + '__12m.csv'))
        pieces, monthly_records = [], []
        for origin in origins(config, phase):
            folder = out / 'models' / origin / family
            roles, target, definition = get_roles(parent, config, frame, phase, origin, config['window'])
            scored = phase == 'evaluation' or origin >= config['development_start']
            marker = folder / 'COMPLETE.json'
            if marker.exists():
                record = read_json(marker)
                for filename, expected in record['output_sha256'].items():
                    assert sha(folder / filename) == expected
                current = csv_read(folder / 'raw_predictions.csv')
                if scored:
                    part = csv_read(folder / 'scored_predictions.csv')
                    monthly_records += read_json(folder / 'scores.json')
                    pieces.append(part)
            else:
                folder.mkdir(parents=True, exist_ok=True)
                if family in CONTROLS:
                    lookup = parent_pool.set_index('row_id')
                    raw = lookup.loc[target.row_id, 'raw_probability'].to_numpy(dtype=float)
                    assert lookup.loc[target.row_id, 'date'].tolist() == target.date.tolist()
                    json_write(folder / 'reused_control.json', dict(parent_run=config['parent_run'],
                               parent_family=CONTROLS[family], parent_origin=origin, newly_fitted=False))
                    bundle = None
                else:
                    bundle = fit_month(roles, config, family, folder)
                    raw = raw_predict(bundle, target)
                current = target[['row_id', 'date']].copy()
                current['origin'], current['raw_probability'] = origin, raw
                csv_write(current, folder / 'raw_predictions.csv')
                if scored:
                    cal = earlier_predictions(pool, roles['calibration'])
                    assert len(cal) >= config['minimum_calibration_rows'] and cal.origin.max() < origin
                    adjustment, rules = warning_rules(cal)
                    score_target = target_month(frame, origin, config, 'development' if phase == 'development' else 'evaluation')
                    raw_scores = current.set_index('row_id').loc[score_target.row_id, 'raw_probability'].to_numpy()
                    part, records = score(score_target, raw_scores, origin, family, adjustment, rules)
                    pieces.append(part)
                    monthly_records += records
                    csv_write(part, folder / 'scored_predictions.csv')
                    csv_write(cal, folder / 'prequential_calibration.csv')
                    json_write(folder / 'scores.json', records)
                    json_write(folder / 'calibration_and_warnings.json', dict(adjustment=adjustment.summary(),
                               warnings=rules, asof=definition['asof'], calibration_end=cal.date.max(),
                               prequential_origin_max=cal.origin.max(), refit_end=roles['refit'].date.max()))
                    if bundle is not None:
                        bundle['adjustment'], bundle['warnings'] = adjustment, rules
                if bundle is not None:
                    joblib.dump(bundle, folder / 'model.joblib')
                json_write(marker, dict(family=family, origin=origin, refit_end=roles['refit'].date.max(),
                           output_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
            pool = current if pool.empty else pd.concat([pool, current], ignore_index=True)
            assert not pool.row_id.duplicated().any()
            print('  ' + origin + (' scored' if scored else ' warmup'), flush=True)
        csv_write(pool, out / 'raw_pools' / (family + '.csv'))
        all_predictions.append(pd.concat(pieces, ignore_index=True))
        all_monthly += monthly_records
        csv_write(pd.DataFrame(all_monthly), out / 'monthly_metrics.csv')
    keys = ['row_id', 'date', 'outage_label', 'origin']
    combined = all_predictions[0]
    for prediction in all_predictions[1:]:
        combined = combined.merge(prediction, on=keys, validate='one_to_one')
    assert combined.date.is_unique
    expected = (351, 109) if phase == 'development' else (264, 92)
    assert (len(combined), int(combined.outage_label.sum())) == expected
    reference = csv_read(parent / ('03_development' if phase == 'development' else '05_exploratory_2026') / 'predictions.csv')
    assert combined.row_id.equals(reference.row_id)
    for family, original in CONTROLS.items():
        for mode in ['raw', 'intercept_adjusted']:
            a, b = family + '__' + mode, 'xgboost__' + original + '__12m__' + mode
            np.testing.assert_allclose(combined[a + '_probability'], reference[b + '_probability'], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(combined[a + '_fpr20_warning'], reference[b + '_fpr20_warning'])
    csv_write(combined, out / 'predictions.csv')
    return combined, pd.DataFrame(all_monthly)


def development(run, config):
    pred, monthly = predict_phase(run, config, 'development')
    assert pred.date.max() <= config['development_end']
    return dict(dates=len(pred), monthly_scores=len(monthly), newly_fitted_models=8*15, tuning_trials=8*15*4)


def selection(run, config):
    out = run / '04_selection'
    out.mkdir(exist_ok=True)
    pred = csv_read(run / '03_development/predictions.csv')
    monthly = csv_read(run / '03_development/monthly_metrics.csv')
    ranked = rank(pred, monthly)
    stricter = rank(pred, monthly, 'fpr10')
    main, replacement = choose(ranked)
    conservative, conservative_replacement = choose(stricter)
    ap_best = ranked.sort_values(['mean_monthly_AP', 'mean_monthly_log_loss', 'family'],
                                ascending=[False, True, True]).iloc[0].to_dict()
    csv_write(ranked, out / 'ranked_features.csv')
    csv_write(stricter, out / 'ranked_features_fpr10.csv')
    result = dict(main=main, replacement_qualified=replacement,
                  conservative_alternative=conservative, conservative_replacement_qualified=conservative_replacement,
                  AP_alternative=ap_best, selection_asof=config['selection_asof'],
                  rule='Require pooled development FPR<=20%, AP better than full control in at least 7/12 '
                       'months, and no worse mean monthly log loss. Among qualifying candidates choose highest '
                       'pooled recall, then higher mean monthly AP, lower mean loss, family name. '
                       'If none qualify retain original full control and flag unmet replacement criteria.',
                  evaluated_2026_to_choose=False, independent_2026_test=False)
    json_write(out / 'selection.json', result)
    return result


def evaluate(run, config):
    out = run / '05_exploratory_2026'
    choice = read_json(run / '04_selection/selection.json')
    choice_hash = sha(run / '04_selection/selection.json')
    pred, monthly = predict_phase(run, config, 'evaluation')
    records = []
    for family in FAMILIES:
        for mode in ['raw', 'intercept_adjusted']:
            name = family + '__' + mode
            for policy in config['warning_policies']:
                records.append(dict(family=family, mode=mode, policy=policy,
                                    **metrics(pred.outage_label, pred[name + '_probability'], pred[name + '_' + policy + '_warning'])))
    frozen = csv_read(PROJECT / config['parent_run'] / '05_exploratory_2026/predictions.csv')
    assert pred.row_id.equals(frozen.row_id)
    name = 'static_xgboost_full'
    pred[name + '_probability'] = frozen[name + '_probability']
    pred[name + '_fpr20_warning'] = frozen[name + '_fpr20_warning']
    records.append(dict(family=name, mode='preserved', policy='fpr20',
                        **metrics(pred.outage_label, pred[name + '_probability'], pred[name + '_fpr20_warning'])))
    for name in ['monthly_frequency', 'recent_frequency']:
        pred[name + '_probability'] = frozen[name + '_probability']
        pred[name + '_fpr20_warning'] = frozen[name + '_fpr20_warning']
        records.append(dict(family=name, mode='preserved', policy='fixed050',
                            **metrics(pred.outage_label, pred[name + '_probability'], pred[name + '_fpr20_warning'])))
    csv_write(pd.DataFrame(records), out / 'overall_metrics.csv')
    csv_write(pred, out / 'predictions.csv')
    selected = choice['main']['family'] + '__raw'
    names = list(dict.fromkeys([selected, 'history_control__raw', 'full_control__raw',
                               'static_xgboost_full', 'recent_frequency']))
    csv_write(intervals(pred, names, selected, config), out / 'block_bootstrap_intervals.csv')
    assert sha(run / '04_selection/selection.json') == choice_hash
    return dict(selected=selected, dates=len(pred), positives=int(pred.outage_label.sum()),
                selection_unchanged=True, newly_fitted_models=8*9, tuning_trials=8*9*4)


def report(run, config):
    from .reporting import write_report, figures
    out = run / '06_report'
    out.mkdir(exist_ok=True)
    frame = csv_read(run / '02_features/city_feature_panel.csv')
    verified = 0
    for phase, relative in [('development', '03_development'), ('evaluation', '05_exploratory_2026')]:
        pool_by_family = {family: csv_read(run / relative / 'raw_pools' / (family + '.csv')) for family in FAMILIES}
        for model_path in sorted((run / relative / 'models').glob('*/*/model.joblib')):
            folder = model_path.parent
            family, origin = folder.name, folder.parent.name
            roles, target, _ = get_roles(PROJECT / config['parent_run'], config, frame, phase, origin, config['window'])
            bundle = joblib.load(model_path)
            p = raw_predict(bundle, target)
            saved = pool_by_family[family].set_index('row_id').loc[target.row_id, 'raw_probability'].to_numpy()
            np.testing.assert_allclose(p, saved, rtol=0, atol=1e-14)
            medians = roles['refit'][bundle['transformer'].numeric].median().to_dict()
            assert medians == bundle['transformer'].medians
            assert bundle['model'].get_booster().num_boosted_rounds() == bundle['selection']['trees']
            assert bundle['transformer'].scaler is None
            verified += 1
    assert verified == 192
    parent_manifest = verify_parent(PROJECT / config['parent_run'])
    for path, expected in read_json(run / '01_audit/earlier_artifact_hashes.json').items():
        assert sha(PROJECT / path) == expected
    # Independently check every aggregate warning confusion matrix.
    from sklearn.metrics import confusion_matrix
    pred = csv_read(run / '05_exploratory_2026/predictions.csv')
    scores = csv_read(run / '05_exploratory_2026/overall_metrics.csv')
    for _, r in scores.iterrows():
        name = r.family + '__' + r['mode'] if r['mode'] != 'preserved' else r.family
        policy = r.policy if r['mode'] != 'preserved' else 'fpr20'
        got = confusion_matrix(pred.outage_label, pred[name + '_' + policy + '_warning'], labels=[0, 1]).ravel()
        np.testing.assert_array_equal(got, [r.tn, r.fp, r.fn, r.tp])
    figures(run)
    write_report(run, config)
    json_write(out / 'verification.json', dict(model_bundles_verified=verified,
               original_monthly_artifacts_preserved=len(parent_manifest),
               earlier_city_extension_artifacts_preserved=len(read_json(run / '01_audit/earlier_artifact_hashes.json')),
               confusion_matrices_independently_checked=len(scores), reused_controls_exactly_reproduced=True,
               all_original_36_columns_unchanged=True, future_month_targets_excluded=True))
    # Completion closes the immutable run; visual QA is stored separately under analysis/.
    manifest = {str(p.relative_to(run)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(run.rglob('*')) if p.is_file() and
                p != out / 'artifact_manifest.json' and p != out / 'COMPLETION.json' and
                not p.is_relative_to(run / 'status')}
    json_write(out / 'artifact_manifest.json', manifest)
    json_write(out / 'COMPLETION.json', dict(all_six_stages_completed=True, verified_models=verified,
               result='exploratory chronological feature comparison; see report', artifacts=len(manifest)))
    return dict(verified_models=verified, artifacts=len(manifest))
