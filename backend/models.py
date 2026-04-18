"""
ML Models Module (XGBoost + Random Forest)
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


class XGBoostPredictor:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        use_gpu = self._use_gpu()
        self.feature_names_: list[str] | None = None
        self.model: Any = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=0,
        )

    def _use_gpu(self) -> bool:
        value = os.getenv("ML_USE_GPU", "auto").strip().lower()
        if value in {"0", "false", "cpu", "off", "no"}:
            return False
        if value in {"1", "true", "gpu", "cuda", "on", "yes"}:
            return True
        # auto mode: default to CPU for portability (no CUDA required)
        return False

    def train(self, X_train, y_train, X_val, y_val, feature_names=None, **_kwargs):
        neg = max(int((y_train == 0).sum()), 1)
        pos = max(int((y_train == 1).sum()), 1)
        self.model.set_params(scale_pos_weight=neg / pos)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return {"trained": True}

    def predict(self, X):
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.error(f"Prediction failed: {exc}")
            return np.zeros(len(X))

    def get_feature_importance(self) -> dict[str, float]:
        try:
            booster = self.model.get_booster()
            scores = booster.get_score(importance_type="gain")
            if not scores:
                return {}

            feature_map: dict[str, float] = {}
            if self.feature_names_:
                for key, value in scores.items():
                    if key.startswith("f") and key[1:].isdigit():
                        index = int(key[1:])
                        if 0 <= index < len(self.feature_names_):
                            feature_map[self.feature_names_[index]] = _safe_float(value)
                        else:
                            feature_map[key] = _safe_float(value)
                    else:
                        feature_map[key] = _safe_float(value)
            else:
                for key, value in scores.items():
                    feature_map[key] = _safe_float(value)

            return dict(
                sorted(feature_map.items(), key=lambda item: item[1], reverse=True)
            )
        except Exception as exc:
            logger.error(f"Failed to extract feature importance: {exc}")
            return {}

    def save(self, path):
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names_,
            },
            path,
        )
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path):
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.feature_names_ = payload.get("feature_names")
        else:
            self.model = payload
        logger.info(f"XGBoost model loaded from {path}")


class RandomForestPredictor:
    def __init__(
        self, n_estimators: int = 200, max_depth: int = 12, random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )

    def train(self, X, y, feature_names=None, **_kwargs):
        self.model.fit(X, y)
        return {"trained": True, "feature_names": feature_names}

    def predict(self, X):
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.error(f"Random forest prediction failed: {exc}")
            return np.zeros(len(X))

    def save(self, path):
        joblib.dump(self.model, path)
        logger.info(f"Random forest model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        logger.info(f"Random forest model loaded from {path}")


class EnsemblePredictor:
    def __init__(self, xgb_weight: float = 0.6, rf_weight: float = 0.4):
        self.weights = {"xgb": xgb_weight, "rf": rf_weight}
        self.xgb = None
        self.rf = None

    def fit(self, xgb_model, rf_model=None, lstm_model=None):
        self.xgb = xgb_model
        self.rf = rf_model
        return self

    def get_individual_probs(
        self, X_test: np.ndarray, X_seq: np.ndarray | None = None
    ) -> dict[str, np.ndarray | None]:
        probs: dict[str, np.ndarray | None] = {
            "xgb_prob": None,
            "rf_prob": None,
        }

        if self.xgb is not None:
            probs["xgb_prob"] = np.asarray(self.xgb.predict(X_test), dtype=float)

        if self.rf is not None:
            probs["rf_prob"] = np.asarray(self.rf.predict(X_test), dtype=float)

        return probs

    def predict_proba(
        self, X_test: np.ndarray, X_seq: np.ndarray | None = None
    ) -> np.ndarray:
        probs = self.get_individual_probs(X_test, X_seq)
        valid = [prob for prob in probs.values() if prob is not None and len(prob) > 0]
        if not valid:
            return np.array([])

        n = min(len(prob) for prob in valid)
        weighted_sum = np.zeros(n)
        total_weight = 0.0

        for model_name, prob in probs.items():
            if prob is None or len(prob) == 0:
                continue
            weight = float(self.weights.get(model_name.split("_")[0], 0.0))
            if weight <= 0:
                continue
            aligned = (
                prob[-n:]
                if len(prob) >= n
                else np.pad(prob, (n - len(prob), 0), mode="edge")
            )
            weighted_sum += weight * aligned
            total_weight += weight

        if total_weight == 0:
            return np.clip(weighted_sum, 0, 1)

        return np.clip(weighted_sum / total_weight, 0, 1)


def evaluate_predictions(y_true, y_pred_proba):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    if y_true.size == 0 or y_pred_proba.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.5,
            "auc_roc": 0.5,
            "log_loss": 0.0,
            "brier_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    if len(np.unique(y_true)) < 2:
        y_pred = (y_pred_proba > 0.5).astype(int)
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.5,
            "auc_roc": 0.5,
            "log_loss": 0.0,
            "brier_score": 0.0,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    y_pred = (y_pred_proba > 0.5).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        roc_auc = 0.5

    try:
        loss = float(log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1 - 1e-7)))
    except Exception:
        loss = 0.0

    brier = float(
        np.mean(
            (np.asarray(y_true, dtype=float) - np.asarray(y_pred_proba, dtype=float))
            ** 2
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "auc_roc": roc_auc,
        "log_loss": loss,
        "brier_score": brier,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
