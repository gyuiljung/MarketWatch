"""
L1: Turbulence Monitor
======================
개별 자산 RV percentile 기반 시장 변동 상태 측정.
market-watch/src/regime/turbulence.py 기반.

예측 아님. "지금 얼마나 출렁이는가?"
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TurbulenceState:
    asset: str
    rv_5d: float
    rv_20d: float
    percentile_5d: float
    percentile_20d: float
    status: str  # 'calm', 'normal', 'elevated', 'extreme'


class TurbulenceMonitor:
    """RV percentile 기반 시장 변동 모니터"""

    def __init__(self, config: dict):
        analysis = config.get('analysis', {})
        self.short_window = analysis.get('rv_window', 5)
        self.long_window = 20
        self.lookback = analysis.get('lookback', 252)
        self.thresholds = config.get('thresholds', {})

    def compute_rv_percentile(self, series: pd.Series, window: int = None) -> pd.Series:
        """Rolling RV percentile"""
        if window is None:
            window = self.short_window
        rv = series.rolling(window).std() * np.sqrt(252) * 100
        return rv.rolling(self.lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
        )

    def compute_all_rv(self, returns: pd.DataFrame) -> pd.DataFrame:
        """모든 자산의 RV percentile (5d)"""
        result = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            result[col] = self.compute_rv_percentile(returns[col])
        return result

    def get_current_states(self, returns: pd.DataFrame) -> List[TurbulenceState]:
        """현재 자산별 변동 상태"""
        states = []
        for col in returns.columns:
            r = returns[col].dropna()
            rv5 = r.rolling(self.short_window).std().iloc[-1] * np.sqrt(252) * 100
            rv20 = r.rolling(self.long_window).std().iloc[-1] * np.sqrt(252) * 100
            pct5 = self.compute_rv_percentile(r, self.short_window).iloc[-1]
            pct20 = self.compute_rv_percentile(r, self.long_window).iloc[-1]

            if np.isnan(pct5):
                continue

            status = self._classify(pct5)
            states.append(TurbulenceState(
                asset=col, rv_5d=rv5, rv_20d=rv20,
                percentile_5d=pct5, percentile_20d=pct20, status=status
            ))

        return sorted(states, key=lambda s: s.percentile_5d, reverse=True)

    def get_overall_status(self, states: List[TurbulenceState]) -> Tuple[float, str]:
        """전체 시장 상태"""
        if not states:
            return 50.0, 'normal'
        avg_pct = np.mean([s.percentile_5d for s in states])
        return avg_pct, self._classify(avg_pct)

    def get_extreme_assets(self, states: List[TurbulenceState]) -> List[str]:
        """Extreme 상태 자산"""
        threshold = self.thresholds.get('rv_extreme', 90)
        return [s.asset for s in states if s.percentile_5d > threshold]

    @staticmethod
    def _classify(pct: float) -> str:
        if pct < 25:
            return 'calm'
        elif pct < 75:
            return 'normal'
        elif pct < 90:
            return 'elevated'
        return 'extreme'
