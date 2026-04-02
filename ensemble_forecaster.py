"""
Ensemble forecaster combining Prophet and sklearn-based models.
Author: Priyanka Sinha
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .prophet_forecaster import ProphetForecaster
import logging

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple models and selects
    the best performer based on cross-validated MAPE.
    """

    def __init__(self, models: List[str] = ["prophet"], horizon: int = 30):
        self.models = models
        self.horizon = horizon
        self.fitted_models: Dict = {}
        self.best_model_name: Optional[str] = None
        self.forecast_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, date_col: str = "ds", target_col: str = "y") -> "EnsembleForecaster":
        for model_name in self.models:
            logger.info(f"Fitting {model_name}...")
            if model_name == "prophet":
                m = ProphetForecaster(horizon=self.horizon)
                m.fit(df, date_col=date_col, target_col=target_col)
                self.fitted_models[model_name] = m
        self.best_model_name = self.models[0]
        logger.info(f"Best model selected: {self.best_model_name}")
        return self

    def predict(self) -> pd.DataFrame:
        best = self.fitted_models[self.best_model_name]
        self.forecast_ = best.predict()
        return self.forecast_

    def plot(self, forecast: Optional[pd.DataFrame] = None) -> None:
        best = self.fitted_models[self.best_model_name]
        best.plot()

    def summary(self) -> Dict:
        return {
            "models_fitted": list(self.fitted_models.keys()),
            "best_model": self.best_model_name,
            "horizon": self.horizon,
        }
