"""
Seasonality and scarcity signal feature engineering for demand forecasting.
Author: Priyanka Sinha
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_time_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Add calendar-based features to support seasonality modelling."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["days_to_year_end"] = (
        pd.to_datetime(df[date_col].dt.year.astype(str) + "-12-31") - df[date_col]
    ).dt.days
    return df


def add_scarcity_signal(
    df: pd.DataFrame,
    capacity_col: str,
    booked_col: str,
    signal_col: str = "scarcity_signal",
) -> pd.DataFrame:
    """
    Add inventory scarcity signal: ratio of booked to total capacity.
    High scarcity (> 0.8) indicates strong demand pressure.
    """
    df = df.copy()
    df[signal_col] = (df[booked_col] / df[capacity_col]).clip(0, 1)
    df["high_scarcity"] = (df[signal_col] > 0.8).astype(int)
    return df


def add_competitor_signal(
    df: pd.DataFrame,
    own_price_col: str,
    competitor_price_col: str,
    signal_col: str = "price_gap_pct",
) -> pd.DataFrame:
    """
    Add relative price gap vs competitor as a demand signal.
    Positive gap = we are more expensive than competitor.
    """
    df = df.copy()
    df[signal_col] = (
        (df[own_price_col] - df[competitor_price_col]) / df[competitor_price_col] * 100
    )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list = [7, 14, 28],
) -> pd.DataFrame:
    """Add rolling mean and std features for the target variable."""
    df = df.copy()
    for w in windows:
        df[f"rolling_mean_{w}d"] = df[target_col].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}d"] = df[target_col].shift(1).rolling(w).std()
    return df
