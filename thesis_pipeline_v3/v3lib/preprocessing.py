"""Fitting-only references and encodings, native CatBoost categories and weather splines."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from .features import FAMILIES
from v2lib.features import STRESS

WEATHER_CURVES = ['precipitation_mm','wind_gust_max_kmh','humidity_mean_pct','temperature_max_c']


class Transformer:
    def __init__(self,config,family,kind,knots=4):
        self.config,self.family,self.kind,self.knots = config,family,kind,knots
        self.fields = list(FAMILIES[family])

    def augment(self,frame):
        f = frame.copy()
        if any(n in self.fields for n in STRESS):
            for raw,field in zip(['precipitation_mm','wind_gust_max_kmh'],STRESS):
                reference = f.psgc.map(self.weather_p90[raw]).fillna(self.global_p90[raw])
                f[field] = f[raw]/reference
        return f

    def categories(self,f):
        result = pd.DataFrame(index=f.index)
        for n in self.categorical:
            result[n] = f[n].fillna('__MISSING__').astype(str) if n == 'psgc' else f[n].map(
                lambda x:'__MISSING__' if pd.isna(x) else str(int(x)))
        return result

    def numeric_matrix(self,f):
        fields = self.numeric if self.kind != 'spline_logistic' else [n for n in self.numeric if n not in self.curves]
        pieces = [f[fields].fillna(self.medians).to_numpy(dtype=np.float32)]
        if self.kind == 'spline_logistic':
            pieces.append(self.spline.transform(f[self.curves].fillna(self.medians)).astype(np.float32))
        return np.concatenate(pieces,axis=1)

    def fit(self,frame):
        self.weather_p90,self.global_p90 = {},{}
        if any(n in self.fields for n in STRESS):
            for raw in ['precipitation_mm','wind_gust_max_kmh']:
                self.weather_p90[raw] = frame.groupby('psgc')[raw].quantile(self.config['weather_stress_quantile']).clip(lower=1.0).to_dict()
                self.global_p90[raw] = max(float(frame[raw].quantile(self.config['weather_stress_quantile'])),1.0)
        f = self.augment(frame)
        self.categorical = [n for n in ['psgc','weather_code'] if n in self.fields]
        numeric = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in numeric if f[n].isna().all()]
        self.numeric = [n for n in numeric if n not in self.dropped]
        self.medians = f[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if f[n].isna().any()]
        names,concepts = list(self.numeric),list(self.numeric)
        if self.kind == 'spline_logistic':
            self.curves = [n for n in WEATHER_CURVES if n in self.numeric]
            self.spline = SplineTransformer(n_knots=self.knots,degree=3,knots='quantile',include_bias=False,extrapolation='constant')
            self.spline.fit(f[self.curves].fillna(self.medians))
            curve_names = list(self.spline.get_feature_names_out(self.curves))
            names = [n for n in self.numeric if n not in self.curves] + curve_names
            concepts = [n for n in self.numeric if n not in self.curves] + [next(n for n in self.curves if c.startswith(n+'_')) for c in curve_names]
            self.scaler = StandardScaler().fit(self.numeric_matrix(f))
        names += [n+'_missing' for n in self.indicators]
        concepts += self.indicators
        if self.kind == 'catboost':
            names += self.categorical
            concepts += self.categorical
        else:
            self.encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32).fit(self.categories(f))
            cats = list(self.encoder.get_feature_names_out(self.categorical))
            names += cats
            concepts += [next(n for n in self.categorical if c.startswith(n+'_')) for c in cats]
        self.names,self.concepts = names,concepts
        return self

    def transform(self,frame):
        f = self.augment(frame)
        numeric = self.numeric_matrix(f)
        if self.kind == 'spline_logistic':
            numeric = self.scaler.transform(numeric).astype(np.float32)
        if self.indicators:
            numeric = np.concatenate([numeric,f[self.indicators].isna().to_numpy(dtype=np.float32)],axis=1)
        assert np.isfinite(numeric).all()
        if self.kind == 'catboost':
            count = len(self.names)-len(self.categorical)
            result = pd.DataFrame(numeric,columns=self.names[:count],index=f.index)
            return pd.concat([result,self.categories(f)],axis=1)
        return np.concatenate([numeric,self.encoder.transform(self.categories(f))],axis=1)

    def summary(self):
        result = {'family':self.family,'kind':self.kind,'fields':self.fields,'names':self.names,'concepts':self.concepts,
            'matrix_columns':len(self.names),'medians':self.medians,'missingness_indicators':self.indicators,
            'dropped_all_missing':self.dropped,'categorical_fields':self.categorical,
            'weather_p90_by_barangay':self.weather_p90,'weather_p90_global_fallback':self.global_p90}
        if self.kind == 'spline_logistic':
            result.update(weather_curve_fields=self.curves,spline_knots=self.knots,
                fitted_knots=[s.t.tolist() for s in self.spline.bsplines_],spline_extrapolation='constant')
        return result
