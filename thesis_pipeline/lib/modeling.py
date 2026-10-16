"""Fitting-only preprocessing, comparable models, metrics and calibration."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, roc_auc_score,
                             precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)
from xgboost import XGBClassifier

from .features import FEATURES


class FeatureTransformer:
    """Serializable transformer with medians/categories learned only from fit rows."""
    def fit(self, frame, fields=None):
        self.fields = list(FEATURES if fields is None else fields)
        if not set(self.fields) <= set(FEATURES):
            raise ValueError('Predictor allowlist violation.')
        self.categorical = [n for n in self.fields if n in ['psgc', 'weather_code']]
        numeric = [n for n in self.fields if n not in self.categorical]
        self.dropped = [n for n in numeric if frame[n].isna().all()]
        self.numeric = [n for n in numeric if n not in self.dropped]
        self.medians = frame[self.numeric].median().to_dict()
        self.indicators = [n for n in self.numeric if frame[n].isna().any()]
        numeric_values = frame[self.numeric].fillna(self.medians).to_numpy(dtype=np.float32)
        self.scaler = StandardScaler().fit(numeric_values)
        self.encoder = None
        if self.categorical:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            self.encoder.fit(self.categories(frame))
        cats = list(self.encoder.get_feature_names_out(self.categorical)) if self.encoder else []
        self.names = self.numeric + [n + '_missing' for n in self.indicators] + cats
        self.concepts = self.numeric + self.indicators + [
            next(c for c in self.categorical if n.startswith(c + '_')) for n in cats]
        return self

    def categories(self, frame):
        values = pd.DataFrame(index=frame.index)
        for name in self.categorical:
            if name == 'weather_code':
                values[name] = frame[name].map(lambda x: '__MISSING__' if pd.isna(x) else str(int(x)))
            else:
                values[name] = frame[name].fillna('__MISSING__').astype(str)
        return values

    def transform(self, frame, scale=False):
        base = frame[self.numeric].fillna(self.medians).to_numpy(dtype=np.float32)
        if scale:
            base = self.scaler.transform(base).astype(np.float32)
        pieces = [base]
        if self.indicators:
            pieces.append(frame[self.indicators].isna().to_numpy(dtype=np.float32))
        if self.encoder:
            pieces.append(self.encoder.transform(self.categories(frame)))
        output = np.concatenate(pieces, axis=1)
        if not np.isfinite(output).all():
            raise ValueError('Non-finite values after preprocessing.')
        return output

    def summary(self):
        return {'conceptual_fields': self.fields, 'numeric_fields': self.numeric,
                'dropped_all_missing_fit_fields': self.dropped, 'medians': self.medians,
                'missingness_indicator_fields': self.indicators, 'encoded_columns': self.names,
                'column_to_concept': dict(zip(self.names, self.concepts)),
                'matrix_columns': len(self.names),
                'categorical_levels': {n: list(v) for n, v in zip(self.categorical, self.encoder.categories_)}
                if self.encoder else {}}

    def unknown_counts(self, frame):
        if self.encoder is None:
            return {}
        cats = self.categories(frame)
        return {n: int((~cats[n].isin(levels)).sum()) for n, levels in
                zip(self.categorical, self.encoder.categories_)}


def frequency_fit(frame, prior=30):
    rate = float(frame.outage_label.mean())
    agg = frame.groupby('psgc').outage_label.agg(['sum', 'count'])
    return {'pooled_rate': rate, 'prior_days': prior,
            'rates': ((agg['sum'] + prior * rate) / (agg['count'] + prior)).to_dict()}


def frequency_predict(model, frame):
    return frame.psgc.map(model['rates']).fillna(model['pooled_rate']).to_numpy(dtype=float)


def xgb_model(config, parameters, trees=None, early=True, label=None):
    settings = config['xgboost']
    ratio = float((label == 0).sum() / (label == 1).sum()) if label is not None else 1.0
    return XGBClassifier(objective='binary:logistic', tree_method='hist', eval_metric='logloss',
                         n_jobs=config['threads'], random_state=config['seed'],
                         learning_rate=settings['learning_rate'], subsample=settings['subsample'],
                         colsample_bytree=settings['colsample_bytree'],
                         n_estimators=trees or settings['max_trees'],
                         early_stopping_rounds=settings['early_stopping_rounds'] if early else None,
                         max_depth=parameters['max_depth'], min_child_weight=parameters['min_child_weight'],
                         reg_lambda=parameters['reg_lambda'],
                         scale_pos_weight=ratio if parameters['class_weight_mode'] == 'training_ratio' else 1.0)


def freeze_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                  out=np.zeros(len(thresholds)), where=(precision[:-1] + recall[:-1]) > 0)
    # Highest threshold wins an F1 tie. No threshold is chosen on calibration or test labels.
    selected = np.flatnonzero(np.isclose(f, f.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[selected])


def metrics(y, probability, threshold=0.5, decision=None):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int) if decision is None else np.asarray(decision, dtype=int)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'rows': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'brier_score': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'roc_auc': float(roc_auc_score(y, p)) if both else None,
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)), 'accuracy': float(accuracy_score(y, pred)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


class SigmoidCalibrator:
    """Sigmoid of the raw probability logit; fit on separate natural-prevalence data."""
    def fit(self, raw_probability, y):
        if len(np.unique(y)) != 2:
            raise ValueError('Calibration requires both outcome classes.')
        self.model = LogisticRegression(C=1e6, max_iter=2000, solver='lbfgs')
        self.model.fit(logit(np.clip(raw_probability, 1e-7, 1 - 1e-7)).reshape(-1, 1), y)
        return self

    def predict(self, raw_probability):
        return self.model.predict_proba(logit(np.clip(raw_probability, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]

    def summary(self):
        return {'method': 'logit-sigmoid, unweighted separate calibration period',
                'slope': float(self.model.coef_[0, 0]), 'intercept': float(self.model.intercept_[0]),
                'increases_with_raw_score': bool(self.model.coef_[0, 0] > 0)}


def raw_predictions(bundle, frame):
    if bundle['kind'] == 'frequency':
        return frequency_predict(bundle['model'], frame)
    X = bundle['transformer'].transform(frame, scale=bundle['kind'] == 'logistic')
    return bundle['model'].predict_proba(X)[:, 1]
