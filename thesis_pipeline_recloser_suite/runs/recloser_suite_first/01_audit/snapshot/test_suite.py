import unittest
import numpy as np
import pandas as pd
from suitelib.common import ROOT, PROJECT, read_json, csv_read
from suitelib.data import FAMILIES, FORBIDDEN, monthly_roles, frozen_roles, calibration_rows, known
from suitelib.modeling import Transformer, create, candidates, fit_one, raw_predict, warning_rules
from monthlib.modeling import InterceptAdjustment
from citylib.modeling import MonotoneCalibrator


def fixture(n=80):
    fields = list(dict.fromkeys(field for values in FAMILIES.values() for field in values))
    frame = pd.DataFrame({name: np.arange(n, dtype=float) % 13 for name in fields})
    frame['psgc'] = ['UNIT_'+str(i % 9) for i in range(n)]
    frame['recloser_id'] = frame.psgc
    frame['day_of_week'] = np.arange(n) % 7
    frame['weather_code'] = 51
    frame['outage_label'] = (np.arange(n) % 7 == 0).astype(int)
    frame['row_id'] = np.arange(n)
    frame['date'] = pd.date_range('2025-01-01', periods=n).strftime('%Y-%m-%d')
    return frame


class SuiteSafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(ROOT / 'config.json')

    def test_publication_availability_is_per_recloser_day(self):
        frame = pd.DataFrame(dict(outage_label=[1, 1, 0], first_support_utc=pd.to_datetime(
            ['2025-12-31T11:00:00Z', '2025-12-31T13:00:00Z', None], utc=True)))
        self.assertEqual(known(frame, '2025-12-31T20:00:00+08:00').tolist(), [True, False, True])

    def test_prequential_calibration_rejects_missing_and_future_predictions(self):
        cal = pd.DataFrame(dict(row_id=[1, 2], date=['2025-12-01', '2025-12-02'], recloser=['a', 'b'], outage_label=[1, 0]))
        pool = pd.DataFrame(dict(row_id=[1], origin=['2025-12-01'], raw_probability=[.1]))
        with self.assertRaises(AssertionError): calibration_rows(pool, cal, '2026-01-01', 2)
        pool = pd.DataFrame(dict(row_id=[1, 2], origin=['2026-01-01']*2, raw_probability=[.1, .2]))
        with self.assertRaises(AssertionError): calibration_rows(pool, cal, '2026-01-01', 2)

    def test_identity_and_imputation_are_fitting_only(self):
        frame = fixture()
        frame.loc[0, 'veg5_mean'] = np.nan
        prep = Transformer('full', 'xgboost').fit(frame)
        medians = prep.medians.copy()
        future = frame.iloc[:3].copy()
        future[prep.numeric] = 100000.
        future.recloser_id = 'UNSEEN'; future.weather_code = 999
        self.assertTrue(np.isfinite(prep.transform(future)).all())
        self.assertEqual(prep.medians, medians)
        self.assertIsNone(prep.scaler)

    def test_legacy_stress_reference_is_fitting_only(self):
        frame = fixture()
        prep = Transformer('legacy_full', 'logistic').fit(frame)
        expected = max(float(frame.precipitation_mm.quantile(.9)), 1.)
        self.assertEqual(prep.global_p90['precipitation_mm'], expected)
        references = prep.p90.copy()
        future = frame.iloc[:2].copy(); future.precipitation_mm = 10000.
        self.assertTrue(np.isfinite(prep.transform(future)).all())
        self.assertEqual(prep.p90, references)

    def test_every_family_keeps_required_histories_and_excludes_target(self):
        for fields in FAMILIES.values():
            self.assertTrue(set(fields).isdisjoint(FORBIDDEN))
            self.assertTrue({'equipment_fault_days_prev7d', 'equipment_fault_days_prev14d'} <= set(fields))
        self.assertEqual(len(FAMILIES['full']), 37)
        self.assertEqual(len(FAMILIES['legacy_full']), 88)

    def test_monthly_adjustment_preserves_all_learned_warning_decisions(self):
        p = np.linspace(.005, .4, 80); y = (np.arange(80) % 7 == 0).astype(int)
        calibrator = InterceptAdjustment().fit(p, y)
        q = calibrator.predict(p)
        first, second = warning_rules(y, p), warning_rules(y, q)
        for policy in ['fpr20', 'fpr10', 'balanced_accuracy', 'f1']:
            np.testing.assert_array_equal(p >= first[policy]['threshold'], q >= second[policy]['threshold'])

    def test_nonnegative_calibration_cannot_reverse_ranking(self):
        p = np.linspace(.01, .9, 80); y = (np.arange(80) < 20).astype(int)
        fitted = MonotoneCalibrator().fit(p, y)
        self.assertGreaterEqual(fitted.slope, 0)
        self.assertTrue((np.diff(fitted.predict(p)) >= 0).all())

    def test_original_evaluation_population_is_preserved(self):
        frame = csv_read(PROJECT / self.config['reference_run'] / '02_features/recloser_feature_panel.csv')
        roles = frozen_roles(frame, '2026-01-01', self.config)
        self.assertEqual((len(roles['outer']), int(roles['outer'].outage_label.sum())), (2098, 57))
        self.assertEqual(roles['refit'].date.max(), '2025-09-16')
        self.assertEqual(roles['calibrate'].date.max(), '2025-12-17')

    def test_all_algorithms_accept_the_shared_transform_and_predict(self):
        frame = fixture(100)
        fit, tune = frame.iloc[:70], frame.iloc[70:]
        for kind in ['logistic', 'spline_logistic', 'xgboost', 'catboost', 'random_forest']:
            with self.subTest(kind=kind):
                prep = Transformer('full', kind).fit(fit)
                params = candidates(kind)[0]
                model = create(kind, params, self.config, initial=True)
                fit_one(model, prep, fit, self.config, tune)
                p = raw_predict(dict(model=model, transformer=prep), tune)
                self.assertTrue(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())


if __name__ == '__main__': unittest.main()
