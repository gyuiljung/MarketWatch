"""
V8 Signal analysis module.

Computes trading signals from V8DB factors using z-score based methodology.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from ..data.v8db_loader import V8DBLoader, V8DBData
from ..core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass
class FactorSignal:
    """Individual factor signal."""
    name: str
    current_value: float
    z_score: float
    contribution: float
    direction: str  # 'momentum' or 'contrarian'
    trigger_zone: str  # 'high_z', 'low_z', 'both_z'
    t_plus: int  # T+1 or T+2


@dataclass
class V8SignalResult:
    """V8 signal computation result."""
    date: pd.Timestamp
    bm: str  # 'kospi' or '3ybm'
    total_signal: float  # -1 to 1 (clipped)
    raw_signal: float  # Before clipping
    t1_contribution: float  # T+1 (Korean factors)
    t2_contribution: float  # T+2 (Overseas factors)
    active_factors: List[FactorSignal]
    top_contributors: List[FactorSignal]
    interpretation: str  # 'long', 'short', 'neutral'

    def __repr__(self) -> str:
        return (
            f"V8SignalResult({self.bm}: {self.total_signal:.2f} [{self.interpretation}], "
            f"T+1={self.t1_contribution:+.2f}, T+2={self.t2_contribution:+.2f})"
        )


class V8SignalAnalyzer:
    """
    V8 Signal analyzer using z-score based factor contributions.

    Methodology:
    1. Compute z-scores for each factor (rolling window)
    2. Apply factor weights (from config or learned)
    3. Aggregate contributions by T+1/T+2
    4. Clip total signal to [-1, 1]
    """

    # Factor categorization
    KOREAN_KEYWORDS = ['한국', 'KRW', 'KOSPI', '코스피', 'KTB', '국고', 'NDF', 'TP ', '원화', 'KR']
    OVERSEAS_KEYWORDS = ['미국', 'US', 'JP', 'EU', 'CN', '달러', 'VIX', 'S&P', '일본', '중국', '유럽', '독일']

    # Z-score thresholds
    Z_THRESHOLD = 1.5  # Activation threshold
    Z_EXTREME = 2.5  # Extreme threshold

    def __init__(
        self,
        v8db_path: Optional[str] = None,
        z_window: int = 252,
        contribution_threshold: float = 0.1
    ):
        """
        Initialize V8 signal analyzer.

        Args:
            v8db_path: Path to V8DB_daily.xlsx
            z_window: Rolling window for z-score calculation
            contribution_threshold: Minimum contribution to be considered active
        """
        self.loader = V8DBLoader(v8db_path)
        self.z_window = z_window
        self.contribution_threshold = contribution_threshold

    def compute_signal(
        self,
        bm: str = 'kospi',
        target_date: Optional[str] = None,
        lookback: int = 252
    ) -> V8SignalResult:
        """
        Compute V8 signal for specified BM.

        Args:
            bm: 'kospi' or '3ybm'
            target_date: Target date (default: latest)
            lookback: Lookback period for z-score calculation

        Returns:
            V8SignalResult object
        """
        # Load data
        data = self.loader.load(bm, lookback=lookback + self.z_window)

        if target_date:
            target = pd.Timestamp(target_date)
            if target not in data.returns.index:
                # Find closest date
                idx = data.returns.index.get_indexer([target], method='ffill')[0]
                if idx >= 0:
                    target = data.returns.index[idx]
                else:
                    target = data.returns.index[-1]
        else:
            target = data.returns.index[-1]

        logger.info(f"Computing V8 signal for {bm} at {target.date()}")

        # Compute z-scores
        z_scores = self._compute_z_scores(data.prices, target)

        # Compute contributions
        contributions = self._compute_contributions(z_scores, data.factors)

        # Aggregate by T+N
        t1_contrib = 0.0
        t2_contrib = 0.0
        active_factors = []

        for factor, contrib in contributions.items():
            if abs(contrib) < self.contribution_threshold:
                continue

            z = z_scores.get(factor, 0.0)
            t_plus = self._classify_t_plus(factor)
            direction = self._classify_direction(factor)
            trigger = self._classify_trigger_zone(z)

            factor_signal = FactorSignal(
                name=factor,
                current_value=float(data.prices[factor].iloc[-1]) if factor in data.prices.columns else 0.0,
                z_score=z,
                contribution=contrib,
                direction=direction,
                trigger_zone=trigger,
                t_plus=t_plus,
            )
            active_factors.append(factor_signal)

            if t_plus == 1:
                t1_contrib += contrib
            else:
                t2_contrib += contrib

        # Total signal
        raw_signal = t1_contrib + t2_contrib
        total_signal = np.clip(raw_signal, -1.0, 1.0)

        # Interpretation
        if total_signal > 0.5:
            interpretation = 'long'
        elif total_signal < -0.5:
            interpretation = 'short'
        else:
            interpretation = 'neutral'

        # Top contributors
        active_factors.sort(key=lambda x: abs(x.contribution), reverse=True)
        top_contributors = active_factors[:10]

        return V8SignalResult(
            date=target,
            bm=bm,
            total_signal=float(total_signal),
            raw_signal=float(raw_signal),
            t1_contribution=t1_contrib,
            t2_contribution=t2_contrib,
            active_factors=active_factors,
            top_contributors=top_contributors,
            interpretation=interpretation,
        )

    def _compute_z_scores(
        self,
        prices: pd.DataFrame,
        target_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Compute z-scores for all factors at target date."""
        z_scores = {}

        # Get data up to target date
        mask = prices.index <= target_date
        data = prices[mask]

        if len(data) < self.z_window:
            logger.warning(f"Insufficient data for z-score: {len(data)} < {self.z_window}")
            return z_scores

        for col in data.columns:
            try:
                series = data[col].dropna()
                if len(series) < self.z_window:
                    continue

                # Rolling z-score
                rolling_mean = series.rolling(self.z_window).mean()
                rolling_std = series.rolling(self.z_window).std()

                if rolling_std.iloc[-1] > 0:
                    z = (series.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
                    z_scores[col] = float(z)
            except Exception as e:
                logger.debug(f"Z-score failed for {col}: {e}")
                continue

        return z_scores

    def _compute_contributions(
        self,
        z_scores: Dict[str, float],
        factors: List[str]
    ) -> Dict[str, float]:
        """
        Compute factor contributions from z-scores.

        Simple model: contribution = sign(z) * min(|z|/Z_EXTREME, 1) * weight
        """
        contributions = {}

        for factor in factors:
            if factor not in z_scores:
                continue

            z = z_scores[factor]
            if abs(z) < self.Z_THRESHOLD:
                continue

            # Simple contribution: scaled z-score
            # Positive z -> typically short (contrarian) for most factors
            # This is simplified; real V8 uses learned weights
            direction_mult = -1.0  # Default contrarian

            # Adjust for momentum factors
            if any(kw in factor for kw in ['VIX', '변동성', 'volatility']):
                direction_mult = -1.0  # High VIX -> short
            elif any(kw in factor for kw in ['S&P', 'KOSPI', 'NKY', '지수']):
                direction_mult = 1.0  # High index -> long (momentum)

            contribution = direction_mult * np.clip(z / self.Z_EXTREME, -1.0, 1.0) * 0.5
            contributions[factor] = contribution

        return contributions

    def _classify_t_plus(self, factor: str) -> int:
        """Classify factor as T+1 (Korean) or T+2 (Overseas)."""
        if any(kw in factor for kw in self.KOREAN_KEYWORDS):
            return 1
        return 2

    def _classify_direction(self, factor: str) -> str:
        """Classify factor direction as momentum or contrarian."""
        momentum_keywords = ['추세', 'momentum', 'trend']
        if any(kw in factor.lower() for kw in momentum_keywords):
            return 'momentum'
        return 'contrarian'

    def _classify_trigger_zone(self, z: float) -> str:
        """Classify z-score trigger zone."""
        if z > self.Z_THRESHOLD:
            return 'high_z'
        elif z < -self.Z_THRESHOLD:
            return 'low_z'
        else:
            return 'neutral'

    def compute_all_signals(
        self,
        target_date: Optional[str] = None
    ) -> Dict[str, V8SignalResult]:
        """
        Compute signals for all BMs.

        Args:
            target_date: Target date (default: latest)

        Returns:
            Dict of BM -> V8SignalResult
        """
        results = {}

        for bm in ['kospi', '3ybm']:
            try:
                results[bm] = self.compute_signal(bm, target_date)
            except Exception as e:
                logger.error(f"Failed to compute signal for {bm}: {e}")

        return results

    def get_signal_summary(
        self,
        target_date: Optional[str] = None
    ) -> str:
        """
        Get formatted signal summary.

        Args:
            target_date: Target date (default: latest)

        Returns:
            Formatted summary string
        """
        signals = self.compute_all_signals(target_date)

        lines = []
        lines.append("=" * 60)
        lines.append("V8 Signal Summary")
        lines.append("=" * 60)

        for bm, result in signals.items():
            lines.append(f"\n{bm.upper()}")
            lines.append("-" * 40)
            lines.append(f"Date: {result.date.date()}")
            lines.append(f"Signal: {result.total_signal:.2f} ({result.interpretation.upper()})")
            lines.append(f"T+1 (Korean): {result.t1_contribution:+.2f}")
            lines.append(f"T+2 (Overseas): {result.t2_contribution:+.2f}")

            if result.top_contributors:
                lines.append("\nTop Contributors:")
                for f in result.top_contributors[:5]:
                    lines.append(f"  {f.name[:30]:30s} Z={f.z_score:+.2f} C={f.contribution:+.3f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
