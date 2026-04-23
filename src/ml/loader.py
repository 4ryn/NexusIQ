"""
Safe model loader — prevents torch DLL crash by rebuilding SHAP at runtime
"""

import joblib
from loguru import logger
from src.utils.config import cfg

_forecast_model = None
_churn_model = None


def get_forecast_model():
    global _forecast_model

    if _forecast_model is not None:
        return _forecast_model

    path = cfg.models_dir / "forecast.joblib"

    try:
        import src.ml.train
        model = joblib.load(path)

        # rebuild SHAP safely
        try:
            import shap
            model.explainer = shap.TreeExplainer(model.model)
        except Exception as e:
            logger.warning(f"SHAP skipped: {e}")

        _forecast_model = model
        logger.info("Forecast model loaded")
        return model

    except Exception as e:
        logger.error(f"Forecast load failed: {e}")
        return None


def get_churn_model():
    global _churn_model

    if _churn_model is not None:
        return _churn_model

    path = cfg.models_dir / "churn.joblib"

    try:
        import src.ml.train
        model = joblib.load(path)

        # rebuild SHAP safely
        try:
            import shap
            model.explainer = shap.TreeExplainer(model.model)
        except Exception as e:
            logger.warning(f"SHAP skipped: {e}")

        _churn_model = model
        logger.info(f"Churn model loaded (AUC={model.auc_roc:.4f})")
        return model

    except Exception as e:
        logger.error(f"Churn load failed: {e}")
        return None


def retrain_all():
    from src.ml.train import train_all
    return train_all()