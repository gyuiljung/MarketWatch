"""
Impulse response analysis module.

Analyzes how hub asset movements propagate to connected assets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import pandas as pd
import numpy as np

from ..core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class LeadLagResult:
    """Lead-lag cross-correlation result."""
    lag_correlations: Dict[int, float]
    best_lag: int
    best_corr: float
    contemporaneous: float
    hub_leads: bool


@dataclass
class ConditionalResponse:
    """Conditional response to hub movements."""
    n_events: int
    past_5d_avg: float
    same_day_avg: float
    future_5d_avg: float


class ImpulseResponseAnalyzer:
    """
    Impulse response analyzer.

    Analyzes:
    - Lead/Lag cross-correlations between hub and neighbors
    - Conditional response when hub moves > threshold
    """

    def __init__(self, config: Config):
        """
        Initialize impulse analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        analysis = config.analysis
        self.window = getattr(analysis, 'impulse_window', 60)
        self.max_lag = getattr(analysis, 'impulse_max_lag', 5)

    def compute_lead_lag(
        self,
        returns: pd.DataFrame,
        hub: str,
        neighbors: List[str]
    ) -> Dict[str, LeadLagResult]:
        """
        Compute lead-lag cross-correlations.

        Args:
            returns: Returns DataFrame
            hub: Hub asset name
            neighbors: List of neighbor asset names

        Returns:
            Dict of neighbor -> LeadLagResult
        """
        recent = returns.iloc[-self.window:]
        results = {}

        if hub not in recent.columns:
            logger.warning(f"Hub {hub} not in data")
            return results

        for neighbor in neighbors:
            if neighbor not in recent.columns:
                continue

            lag_corrs = {}
            for lag in range(-self.max_lag, self.max_lag + 1):
                if lag < 0:
                    x = recent[hub].iloc[:lag].values
                    y = recent[neighbor].iloc[-lag:].values
                elif lag > 0:
                    x = recent[hub].iloc[lag:].values
                    y = recent[neighbor].iloc[:-lag].values
                else:
                    x = recent[hub].values
                    y = recent[neighbor].values

                if len(x) > 10:
                    corr = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(corr):
                        lag_corrs[lag] = corr

            if lag_corrs:
                best_lag = max(lag_corrs.keys(), key=lambda k: abs(lag_corrs[k]))
                results[neighbor] = LeadLagResult(
                    lag_correlations=lag_corrs,
                    best_lag=best_lag,
                    best_corr=lag_corrs[best_lag],
                    contemporaneous=lag_corrs.get(0, 0),
                    hub_leads=best_lag < 0,
                )

        return results

    def compute_conditional_response(
        self,
        returns: pd.DataFrame,
        hub: str,
        neighbors: List[str],
        std_threshold: float = 1.0
    ) -> Dict:
        """
        Analyze response when hub moves > threshold std.

        Args:
            returns: Returns DataFrame
            hub: Hub asset name
            neighbors: List of neighbor asset names
            std_threshold: Standard deviation threshold for "big move"

        Returns:
            Dict with hub info and neighbor responses
        """
        recent = returns.iloc[-self.window:]

        if hub not in recent.columns:
            return {}

        hub_std = recent[hub].std()
        big_move_mask = recent[hub].abs() > (hub_std * std_threshold)
        big_move_dates = recent[big_move_mask].index

        results = {
            'hub': hub,
            'n_events': len(big_move_dates),
            'std_threshold': std_threshold
        }

        for neighbor in neighbors:
            if neighbor not in recent.columns:
                continue

            up_responses = []
            down_responses = []

            for event_date in big_move_dates:
                idx = recent.index.get_loc(event_date)
                hub_ret = recent[hub].iloc[idx]

                response = {
                    'date': event_date,
                    'hub_ret': hub_ret,
                    'same_day': recent[neighbor].iloc[idx],
                }

                if idx >= 5:
                    response['past_5d'] = recent[neighbor].iloc[idx-5:idx].mean()
                else:
                    response['past_5d'] = np.nan

                if idx + 5 < len(recent):
                    response['future_5d'] = recent[neighbor].iloc[idx+1:idx+6].mean()
                else:
                    response['future_5d'] = np.nan

                if hub_ret > 0:
                    up_responses.append(response)
                else:
                    down_responses.append(response)

            neighbor_result = {}

            if up_responses:
                up_df = pd.DataFrame(up_responses)
                neighbor_result['up'] = ConditionalResponse(
                    n_events=len(up_df),
                    past_5d_avg=up_df['past_5d'].mean(),
                    same_day_avg=up_df['same_day'].mean(),
                    future_5d_avg=up_df['future_5d'].mean(),
                )

            if down_responses:
                down_df = pd.DataFrame(down_responses)
                neighbor_result['down'] = ConditionalResponse(
                    n_events=len(down_df),
                    past_5d_avg=down_df['past_5d'].mean(),
                    same_day_avg=down_df['same_day'].mean(),
                    future_5d_avg=down_df['future_5d'].mean(),
                )

            results[neighbor] = neighbor_result

        return results
