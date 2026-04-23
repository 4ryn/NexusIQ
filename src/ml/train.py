# ══════════════════════════════════════════════════════════════════════
# ML Models — XGBoost Forecast + Churn (SAFE + STABLE)
# ══════════════════════════════════════════════════════════════════════

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np
import pandas as pd
import joblib
import shap
from loguru import logger
from typing import Dict, List

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
import xgboost as xgb

from src.utils.config import cfg


# ══════════════════════════════════════════════════════════════════════
# SALES FORECAST MODEL
# ══════════════════════════════════════════════════════════════════════

class SalesForecastModel:

    XGB_PARAMS = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    BASE_FEATURES = [
        "lag_1","lag_2","lag_3","lag_6","lag_12",
        "roll_mean_3","roll_mean_6","roll_mean_12",
        "roll_std_3","roll_std_6",
        "mom_growth","yoy_growth","Orders",
        "month_sin","month_cos","qtr_sin","qtr_cos",
        "trend","lag1_x_roll3"
    ]

    def __init__(self):
        self.model = xgb.XGBRegressor(**self.XGB_PARAMS)
        self.feature_cols: List[str] = []
        self.cv_metrics = {}
        self.explainer = None

    def _engineer_features(self, df):
        df = df.copy()
        df["lag_12"] = df["Revenue"].shift(12)
        df["trend"] = np.arange(len(df))
        df["lag1_x_roll3"] = df["lag_1"] * df["roll_mean_3"]
        return df

    def _features(self, df):
        base = [c for c in self.BASE_FEATURES if c in df.columns]
        dummies = [c for c in df.columns if c.startswith("Description_") or c.startswith("Country_")]
        return base + dummies

    def train(self, df, n_splits=5):

        df = self._engineer_features(df)
        df = pd.get_dummies(df, columns=["Description","Country"], dtype=int)

        self.feature_cols = self._features(df)

        X = df[self.feature_cols].fillna(0)
        y = np.log1p(df["Revenue"])

        tscv = TimeSeriesSplit(n_splits=n_splits)

        metrics = {"mape": [], "mae": [], "rmse": [], "r2": []}

        for tr, va in tscv.split(X):
            self.model.fit(X.iloc[tr], y.iloc[tr], verbose=False)

            pred = np.expm1(self.model.predict(X.iloc[va]))
            y_true = np.expm1(y.iloc[va])

            metrics["mape"].append(mean_absolute_percentage_error(y_true, pred))
            metrics["mae"].append(mean_absolute_error(y_true, pred))
            metrics["rmse"].append(np.sqrt(np.mean((y_true - pred) ** 2)))
            metrics["r2"].append(r2_score(y_true, pred))

        self.model.fit(X, y, verbose=False)

        self.cv_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        self.explainer = shap.TreeExplainer(self.model)

        logger.success(f"Forecast → {self.cv_metrics}")
        return self.cv_metrics

    def predict(self, X):
        Xf = X.reindex(columns=self.feature_cols, fill_value=0)
        return np.expm1(self.model.predict(Xf))

    def save(self):
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.models_dir / "forecast.joblib"

        # 🔥 CRITICAL FIX (remove SHAP before saving)
        explainer = self.explainer
        self.explainer = None

        joblib.dump(self, path)

        self.explainer = explainer

        logger.success(f"Forecast model → {path}")
        return path


# ══════════════════════════════════════════════════════════════════════
# CHURN MODEL
# ══════════════════════════════════════════════════════════════════════

class ChurnModel:

    XGB_PARAMS = dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.8,
        scale_pos_weight=2.5,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )

    def __init__(self):
        self.model = xgb.XGBClassifier(**self.XGB_PARAMS)
        self.feature_cols = []
        self.auc_roc = 0.0
        self.explainer = None

    def _features(self, df):
        exclude = {"Churn"}
        return [c for c in df.columns if c not in exclude and df[c].dtype != "object"]

    def train(self, df):

        df = pd.get_dummies(df)
        df = df.astype(np.float32)

        self.feature_cols = self._features(df)

        X = df[self.feature_cols].fillna(0)
        y = df["Churn"]

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

        self.model.fit(Xtr, ytr, verbose=False)

        probs = self.model.predict_proba(Xte)[:, 1]
        self.auc_roc = float(roc_auc_score(yte, probs))

        self.explainer = shap.TreeExplainer(self.model)

        logger.success(f"Churn → AUC={self.auc_roc:.4f}")
        return {"auc_roc": self.auc_roc}

    def save(self):
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.models_dir / "churn.joblib"

        # 🔥 CRITICAL FIX
        explainer = self.explainer
        self.explainer = None

        joblib.dump(self, path)

        self.explainer = explainer

        logger.success(f"Churn model → {path}")
        return path


# ══════════════════════════════════════════════════════════════════════
# TRAINING ENTRY
# ══════════════════════════════════════════════════════════════════════

def train_all():

    processed = cfg.processed_dir

    if not (processed / "sales.parquet").exists():
        from src.pipeline.etl import run_pipeline
        run_pipeline()

    sales = pd.read_parquet(processed / "sales.parquet")
    churn = pd.read_parquet(processed / "churn.parquet")

    fm = SalesForecastModel()
    fm.train(sales)
    fm.save()

    cm = ChurnModel()
    cm.train(churn)
    cm.save()

    return fm, cm


if __name__ == "__main__":
    train_all()