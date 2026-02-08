"""
L2: Multi-scale Synchronization
================================
네트워크 동기화 수준을 여러 윈도우에서 측정.
단기/중기/장기 동기화 변화로 레짐 전환 감지.
"""
import numpy as np
import pandas as pd


class SyncAnalyzer:
    """Multi-scale 동기화 분석"""

    def __init__(self, config: dict):
        self.windows = config.get('analysis', {}).get('windows', [5, 20, 60])
        self.thresholds = config.get('thresholds', {})

    @staticmethod
    def calc_sync(corr: pd.DataFrame) -> float:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr.values[mask].mean()

    def compute_current(self, returns: pd.DataFrame, network_assets: list) -> dict:
        """현재 multi-scale sync 계산"""
        cols = [c for c in network_assets if c in returns.columns]
        result = {}
        for w in self.windows:
            if len(returns) >= w:
                recent = returns[cols].iloc[-w:]
                corr = recent.corr()
                result[f'sync_{w}d'] = self.calc_sync(corr)
        return result

    def compute_divergence(self, sync_values: dict) -> dict:
        """단기/장기 동기화 괴리 → 레짐 전환 감지"""
        if len(self.windows) < 2:
            return {}

        short_key = f'sync_{self.windows[0]}d'
        long_key = f'sync_{self.windows[-1]}d'

        short_sync = sync_values.get(short_key, 0)
        long_sync = sync_values.get(long_key, 0)
        divergence = short_sync - long_sync

        regime_shift = divergence > 0.08

        return {
            'short_sync': short_sync,
            'long_sync': long_sync,
            'divergence': divergence,
            'regime_shift': regime_shift,
            'interpretation': self._interpret(divergence, short_sync),
        }

    def _interpret(self, divergence: float, short_sync: float) -> str:
        sync_warning = self.thresholds.get('sync_warning', 0.20)

        if divergence > 0.08:
            return "REGIME SHIFT: 단기 동기화 급등 (위험 증가)"
        elif short_sync > sync_warning:
            return "HIGH SYNC: 분산 효과 저하 구간"
        elif short_sync < 0.05:
            return "LOW SYNC: 자산간 탈동조화"
        return "NORMAL"
