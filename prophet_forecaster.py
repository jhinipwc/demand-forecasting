"""
Prophet-based demand forecaster with seasonality and signal integration.
Author: Priyanka Sinha
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Production-grade demand forecaster using Facebook Prophet.
    Supports seasonality decomposition, external regressors, and
    confidence interval generation for scenario planning.
    """

    def __init__(
        self,
        horizon: int = 30,
        seasonality_mode: str = "multiplicative",
        include_holidays: bool = True,
        country_code: str = "DE",
    ):
        self.horizon = horizon
        self.seasonality_mode = seasonality_mode
        self.include_holidays = include_holidays
        self.country_code = country_code
        self.model = None
        self.forecast = None

    def _build_model(self, regressors: Optional[list] = None) -> Prophet:
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        if self.include_holidays:
            model.add_country_holidays(country_name=self.country_code)
        if regressors:
            for reg in regressors:
                model.add_regressor(reg)
        return model

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "ds",
        target_col: str = "y",
        regressors: Optional[list] = None,
    ) -> "ProphetForecaster":
        """
        Fit the Prophet model.

        Args:
            df: DataFrame with date and target columns
            date_col: Name of the date column
            target_col: Name of the target column
            regressors: Optional list of external regressor column names
        """
        train_df = df.rename(columns={date_col: "ds", target_col: "y"})
        self.model = self._build_model(regressors)
        logger.info(f"Fitting Prophet model on {len(train_df)} rows...")
        self.model.fit(train_df)
        logger.info("Model fitted successfully.")
        return self

    def predict(self, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecast for the specified horizon."""
        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=self.horizon)
        self.forecast = self.model.predict(future_df)
        return self.forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]]

    def evaluate(self, actuals: pd.DataFrame, date_col: str = "ds", target_col: str = "y") -> Dict[str, float]:
        """Calculate MAPE and RMSE on actual vs predicted values."""
        merged = actuals.rename(columns={date_col: "ds", target_col: "y"}).merge(
            self.forecast[["ds", "yhat"]], on="ds"
        )
        mape = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"])) * 100
        rmse = np.sqrt(np.mean((merged["y"] - merged["yhat"]) ** 2))
        logger.info(f"MAPE: {mape:.2f}% | RMSE: {rmse:.2f}")
        return {"mape": round(mape, 2), "rmse": round(rmse, 2)}

    def plot(self) -> None:
        """Plot forecast with components."""
        if self.forecast is None:
            raise ValueError("Run predict() first.")
        fig = self.model.plot(self.forecast)
        fig.suptitle("Demand Forecast", fontsize=14)
        fig.show()
        self.model.plot_components(self.forecast)
