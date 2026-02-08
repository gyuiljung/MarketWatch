"""
Volatility analysis module.

Provides realized volatility and percentile calculations.
"""

from typing import Optional
import logging

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import RV_DEFAULT_WINDOW, RV_ANNUALIZATION_FACTOR

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Realized volatility analyzer with percentile rankings.

    Calculates rolling realized volatility and historical percentiles
    for each asset.
    """

    def __init__(self, config: Config):
        """
        Initialize volatility analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.rv_window = config.analysis.rv_window
        self.lookback = config.analysis.lookback

    def compute_rv(self, series: pd.Series, annualize: bool = True) -> pd.Series:
        """
        Compute realized volatility.

        Args:
            series: Return series
            annualize: Whether to annualize (default: True)

        Returns:
            Rolling realized volatility series
        """
        rv = series.rolling(self.rv_window).std()
        if annualize:
            rv = rv * np.sqrt(RV_ANNUALIZATION_FACTOR) * 100
        return rv

    def compute_rv_percentile(self, series: pd.Series) -> pd.Series:
        """
        Compute realized volatility percentile ranking.

        Args:
            series: Return series

        Returns:
            Series of percentile rankings (0-100)
        """
        rv = self.compute_rv(series)
        return rv.rolling(self.lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
        )

    def compute_all(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RV percentiles for all assets.

        Args:
            returns: Returns DataFrame

        Returns:
            DataFrame with RV percentiles for each asset
        """
        logger.info("Computing volatility percentiles...")
        result = pd.DataFrame(index=returns.index)

        for col in returns.columns:
            result[col] = self.compute_rv_percentile(returns[col])

        return result

    def get_extreme_assets(
        self,
        rv_pct: pd.DataFrame,
        threshold: float = 90.0
    ) -> list:
        """
        Get assets with extreme volatility.

        Args:
            rv_pct: RV percentile DataFrame
            threshold: Percentile threshold

        Returns:
            List of asset names exceeding threshold
        """
        if len(rv_pct) == 0:
            return []
        current = rv_pct.iloc[-1]
        return [a for a, v in current.items() if v > threshold]

    def get_elevated_assets(
        self,
        rv_pct: pd.DataFrame,
        lower: float = 75.0,
        upper: float = 90.0
    ) -> list:
        """
        Get assets with elevated (but not extreme) volatility.

        Args:
            rv_pct: RV percentile DataFrame
            lower: Lower threshold
            upper: Upper threshold

        Returns:
            List of asset names in range
        """
        if len(rv_pct) == 0:
            return []
        current = rv_pct.iloc[-1]
        return [a for a, v in current.items() if lower < v <= upper]
