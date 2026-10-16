"""Per-fit weather baselines, imputation, category encoding and numerical scaling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import FEATURES, BASE_FEATURES, STRESS


class Transformer:
    def __init__(self, config, family):
        self.config, self.family = config, family
        self.fields = list(BASE_FEATURES if family == 'base45' else FEATURES)
        assert family in ['base45', 'expanded62']

    def augment(self, frame):
        f = frame.copy()
        if self.family == 'expanded62':
            for raw, field in zip(['precipitation_mm', 'wind_gust_max_kmh'], STRESS):
                reference = f.psgc.map(self.weather_p90[raw]).fillna(self.global_p90[raw])
                f[field] = f[raw] / reference
        return f

    def fit(self, frame):
        if self.family == 'expanded62':
            q, floor = self.config['weather_stress_quantile'], self.config['weather_stress_denominator_floor']
            self.weather_p90, self.global_p90 = {}, {}
            for raw in ['precipitation_mm', 'wind_gust_max_kmh']:
                self.weather_p90[raw] = frame.groupby('psgc')[raw].quantile(q).clip(lower=floor).to_dict()
                self.global_p90[raw] = max(float(frame[raw].quantile(q)), floor)
        f = self.augment(frame)
        self.categorical = ['psgc', 'weather_code']
        numeric = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in numeric if f[n].isna().all()]
        self.numeric = [n for n in numeric if n not in self.dropped]
        self.medians = f[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if f[n].isna().any()]
        x = f[self.numeric].fillna(self.medians).to_numpy(dtype=np.float32)
        self.scaler = StandardScaler().fit(x)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        self.encoder.fit(self.categories(f))
        cat_names = list(self.encoder.get_feature_names_out(self.categorical))
        self.names = self.numeric + [n + '_missing' for n in self.indicators] + cat_names
        self.concepts = self.numeric + self.indicators + [next(c for c in self.categorical if n.startswith(c + '_')) for n in cat_names]
        return self

    def categories(self, f):
        return pd.DataFrame({'psgc': f.psgc.fillna('__MISSING__').astype(str),
            'weather_code': f.weather_code.map(lambda x: '__MISSING__' if pd.isna(x) else str(int(x)))}, index=f.index)

    def transform(self, frame, scale=False):
        f = self.augment(frame)
        x = f[self.numeric].fillna(self.medians).to_numpy(dtype=np.float32)
        if scale:
            x = self.scaler.transform(x).astype(np.float32)
        parts = [x]
        if self.indicators:
            parts.append(f[self.indicators].isna().to_numpy(dtype=np.float32))
        parts.append(self.encoder.transform(self.categories(f)))
        matrix = np.concatenate(parts, axis=1)
        assert np.isfinite(matrix).all(), 'Non-finite transformed input.'
        return matrix

    def unknown_counts(self, frame):
        cats = self.categories(frame)
        return {n: int((~cats[n].isin(values)).sum()) for n, values in zip(self.categorical, self.encoder.categories_)}

    def summary(self):
        return {'family': self.family, 'conceptual_fields': self.fields, 'encoded_columns': self.names,
            'matrix_columns': len(self.names), 'column_to_concept': dict(zip(self.names, self.concepts)),
            'medians': self.medians, 'missingness_indicators': self.indicators,
            'dropped_all_missing': self.dropped,
            'categories': {n: list(values) for n, values in zip(self.categorical, self.encoder.categories_)},
            'weather_p90_by_barangay': getattr(self, 'weather_p90', {}),
            'weather_p90_global_fallback': getattr(self, 'global_p90', {})}
