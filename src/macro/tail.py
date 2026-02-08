"""
L3: Tail Dependence
===================
꼬리의존성 excess 분석 (market_monitor 기반).

핵심: Raw tail dependence가 아니라 EXCESS (= Tail - Expected given correlation).
상관 대비 초과 꼬리 의존이 위기 전염을 나타냄.
"""
import numpy as np
import pandas as pd


class TailDependenceCalculator:
    """Copula-free empirical tail dependence estimation"""

    def __init__(self, config: dict):
        self.config = config
        self.thresholds = [0.05, 0.10]

    def _compute_tail_dependence(self, x: np.ndarray, y: np.ndarray, q: float = 0.10) -> dict:
        """Compute empirical tail dependence coefficients"""
        # Lower tail (동반 폭락)
        x_threshold_low = np.percentile(x, q * 100)
        y_threshold_low = np.percentile(y, q * 100)

        x_in_lower = x < x_threshold_low
        lower_td = (y[x_in_lower] < y_threshold_low).mean() if x_in_lower.sum() > 0 else 0.0

        # Upper tail (동반 급등)
        x_threshold_high = np.percentile(x, (1 - q) * 100)
        y_threshold_high = np.percentile(y, (1 - q) * 100)

        x_in_upper = x > x_threshold_high
        upper_td = (y[x_in_upper] > y_threshold_high).mean() if x_in_upper.sum() > 0 else 0.0

        return {
            'lower': lower_td,
            'upper': upper_td,
            'asymmetry': lower_td - upper_td,
        }

    @staticmethod
    def _expected_tail_given_corr(corr: float, q: float = 0.10) -> float:
        """Expected tail dependence given correlation (linear approx)"""
        return q + 0.35 * abs(corr)

    def compute_for_pairs(self, returns: pd.DataFrame, pairs: list = None) -> dict:
        """Compute tail dependence with excess over correlation"""
        if pairs is None:
            pairs = self.config.get('key_pairs', [])

        corr_matrix = returns.corr()
        results = {}

        for pair in pairs:
            a, b = pair[0], pair[1]
            if a not in returns.columns or b not in returns.columns:
                continue

            x = returns[a].dropna().values
            y = returns[b].dropna().values

            min_len = min(len(x), len(y))
            x, y = x[-min_len:], y[-min_len:]

            corr = corr_matrix.loc[a, b] if a in corr_matrix.index and b in corr_matrix.columns else 0

            td = self._compute_tail_dependence(x, y, q=0.10)
            expected = self._expected_tail_given_corr(corr, q=0.10)

            excess_lower = td['lower'] - expected
            excess_upper = td['upper'] - expected

            results[f'{a}/{b}'] = {
                'lower_10': td['lower'],
                'upper_10': td['upper'],
                'correlation': corr,
                'expected': expected,
                'excess_lower': excess_lower,
                'excess_upper': excess_upper,
                'asymmetry': td['asymmetry'],
                'co_crash_prob': td['lower'],
                'interpretation': self._interpret(td['lower'], excess_lower),
            }

        return results

    @staticmethod
    def _interpret(raw_tail: float, excess: float) -> str:
        if excess > 0.20:
            return "CRISIS CONTAGION"
        elif excess > 0.15:
            return "Elevated contagion"
        elif excess > 0.10:
            return "Mild excess"
        elif excess < -0.10:
            return "Tail diversified"
        return "Normal"
