"""
Tail dependence analysis module.

Implements copula-free empirical tail dependence estimation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import (
    TAIL_CORR_COEFFICIENT,
    TAIL_CRISIS_THRESHOLD,
    TAIL_ELEVATED_THRESHOLD,
    TAIL_MILD_THRESHOLD,
    TAIL_DIVERSIFIED_THRESHOLD,
    TAIL_DEFAULT_Q,
)

logger = logging.getLogger(__name__)


@dataclass
class TailResult:
    """Tail dependence result for a pair."""
    lower_10: float          # Lower tail dependence at 10%
    upper_10: float          # Upper tail dependence at 10%
    correlation: float       # Linear correlation
    expected: float          # Expected tail given correlation
    excess_lower: float      # Excess over expected (key metric)
    excess_upper: float      # Excess over expected
    asymmetry: float         # Lower - Upper
    co_crash_prob: float     # Same as lower_10
    interpretation: str      # Text interpretation


class TailDependenceCalculator:
    """
    Copula-free empirical tail dependence estimator.

    Key insight: Raw tail dependence is misleading.
    What matters is EXCESS tail dependence = Tail - Expected(given correlation)

    If two assets have corr=0.5, we expect tail dependence ~27%.
    If actual tail is 45%, the EXCESS of +18% indicates crisis contagion.
    """

    def __init__(self, config: Config):
        """
        Initialize tail dependence calculator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.key_pairs = config.key_pairs or []
        self.thresholds = [0.05, 0.10]  # 5% and 10% tails

    def _compute_tail_dependence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        q: float = TAIL_DEFAULT_Q
    ) -> dict:
        """
        Compute empirical tail dependence coefficients.

        Args:
            x: First asset returns
            y: Second asset returns
            q: Tail quantile (default: 0.10)

        Returns:
            Dict with lower, upper, asymmetry
        """
        # Lower tail (co-crash)
        x_threshold_low = np.percentile(x, q * 100)
        y_threshold_low = np.percentile(y, q * 100)

        x_in_lower = x < x_threshold_low
        if x_in_lower.sum() > 0:
            lower_td = (y[x_in_lower] < y_threshold_low).mean()
        else:
            lower_td = 0.0

        # Upper tail (co-rally)
        x_threshold_high = np.percentile(x, (1 - q) * 100)
        y_threshold_high = np.percentile(y, (1 - q) * 100)

        x_in_upper = x > x_threshold_high
        if x_in_upper.sum() > 0:
            upper_td = (y[x_in_upper] > y_threshold_high).mean()
        else:
            upper_td = 0.0

        return {
            'lower': lower_td,
            'upper': upper_td,
            'asymmetry': lower_td - upper_td
        }

    def _expected_tail_given_corr(self, corr: float, q: float = TAIL_DEFAULT_Q) -> float:
        """
        Approximate expected tail dependence given correlation.

        For bivariate normal, tail dependence = 0 for |rho| < 1.
        But empirically, higher correlation → higher tail dependence.
        Using linear approximation: expected = q + TAIL_CORR_COEFFICIENT * |corr|

        Args:
            corr: Correlation coefficient
            q: Tail quantile

        Returns:
            Expected tail dependence
        """
        return q + TAIL_CORR_COEFFICIENT * abs(corr)

    def _interpret(self, raw_tail: float, excess: float) -> str:
        """
        Interpret tail dependence based on excess.

        Args:
            raw_tail: Raw tail dependence
            excess: Excess over expected

        Returns:
            Interpretation string
        """
        if excess > TAIL_CRISIS_THRESHOLD:
            return "CRISIS CONTAGION"
        elif excess > TAIL_ELEVATED_THRESHOLD:
            return "Elevated contagion"
        elif excess > TAIL_MILD_THRESHOLD:
            return "Mild excess"
        elif excess < TAIL_DIVERSIFIED_THRESHOLD:
            return "Tail diversified"
        else:
            return "Normal"

    def compute_for_pairs(
        self,
        returns: pd.DataFrame,
        pairs: Optional[List[List[str]]] = None
    ) -> Dict[str, TailResult]:
        """
        Compute tail dependence for specified pairs.

        Args:
            returns: Returns DataFrame
            pairs: List of [asset_a, asset_b] pairs (default: config.key_pairs)

        Returns:
            Dict of 'A/B' -> TailResult
        """
        if pairs is None:
            pairs = self.key_pairs

        if not pairs:
            logger.warning("No pairs specified for tail dependence")
            return {}

        # Correlation matrix
        corr_matrix = returns.corr()

        results = {}
        for pair in pairs:
            a, b = pair[0], pair[1]
            if a not in returns.columns or b not in returns.columns:
                logger.warning(f"Pair {a}/{b} not in data")
                continue

            x = returns[a].dropna().values
            y = returns[b].dropna().values

            # Align lengths
            min_len = min(len(x), len(y))
            x, y = x[-min_len:], y[-min_len:]

            # Get correlation
            corr = corr_matrix.loc[a, b] if a in corr_matrix.index and b in corr_matrix.columns else 0

            # Compute tail dependence at 10%
            td = self._compute_tail_dependence(x, y, q=TAIL_DEFAULT_Q)

            # Expected given correlation
            expected = self._expected_tail_given_corr(corr, q=TAIL_DEFAULT_Q)

            # Excess over correlation
            excess_lower = td['lower'] - expected
            excess_upper = td['upper'] - expected

            results[f'{a}/{b}'] = TailResult(
                lower_10=td['lower'],
                upper_10=td['upper'],
                correlation=corr,
                expected=expected,
                excess_lower=excess_lower,
                excess_upper=excess_upper,
                asymmetry=td['asymmetry'],
                co_crash_prob=td['lower'],
                interpretation=self._interpret(td['lower'], excess_lower),
            )

        return results

    def get_crisis_pairs(
        self,
        tail_results: Dict[str, TailResult],
        threshold: float = TAIL_ELEVATED_THRESHOLD
    ) -> List[str]:
        """
        Get pairs with elevated crisis contagion risk.

        Args:
            tail_results: Tail analysis results
            threshold: Excess threshold

        Returns:
            List of pair keys with excess > threshold
        """
        return [k for k, v in tail_results.items() if v.excess_lower > threshold]
