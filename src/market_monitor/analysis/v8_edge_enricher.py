"""
V8 Edge Enricher module.

Enriches network MST edges with V8 signal attributes.
Applies T+1 (Korean) / T+2 (Overseas) time lag to signal calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import networkx as nx

from .v8_signal import V8SignalAnalyzer, V8SignalResult

logger = logging.getLogger(__name__)


# Asset to V8 factor mapping
# Market Watch asset -> V8DB factor keywords
ASSET_FACTOR_MAP = {
    # FX
    'USDKRW': ['KRW', 'NDF', '원'],
    'USDJPY': ['JPY', '엔'],
    'DXY': ['달러', 'Dollar'],
    'EURUSD': ['EUR', '유로'],

    # Equity
    'SPX': ['S&P', 'SPX'],
    'NKY': ['니케이', 'Nikkei', 'NKY'],
    'HSI': ['항셍', 'HSI', 'Hang'],
    'DAX': ['DAX', '독일'],
    'KOSPI': ['KOSPI', '코스피'],

    # Volatility
    'VIX': ['VIX', '변동성'],

    # Commodities
    'GOLD': ['금', 'Gold', 'XAU'],
    'WTI': ['WTI', '원유', 'Oil'],

    # Rates
    'KTB_2Y': ['KTB', '국고'],
    'KTB_10Y': ['KTB', '국고'],
    'KTB_30Y': ['KTB', '국고'],
    'UST_2Y': ['UST', 'Treasury'],
    'UST_10Y': ['UST', 'Treasury'],
    'UST_30Y': ['UST', 'Treasury'],
    'JGB_2Y': ['JGB', '일본국채'],
    'JGB_10Y': ['JGB', '일본국채'],
    'JGB_30Y': ['JGB', '일본국채'],
}

# T+1 assets (Korean market, same-day signal)
T1_ASSETS = {'USDKRW', 'KOSPI', 'KTB_2Y', 'KTB_10Y', 'KTB_30Y'}

# T+2 assets (Overseas, previous-day signal)
T2_ASSETS = {'SPX', 'NKY', 'HSI', 'DAX', 'VIX', 'GOLD', 'WTI', 'USDJPY', 'DXY', 'EURUSD',
             'UST_2Y', 'UST_10Y', 'UST_30Y', 'JGB_2Y', 'JGB_10Y', 'JGB_30Y', 'Bund_10Y'}


@dataclass
class EdgeV8Info:
    """V8 signal information for an edge."""
    asset1: str
    asset2: str
    signal: float  # Combined signal strength (-1 to 1)
    direction: str  # 'bullish', 'bearish', 'neutral'
    asset1_contrib: float
    asset2_contrib: float
    lag_info: str  # e.g., "T+1/T+2"


class V8EdgeEnricher:
    """
    Enriches network edges with V8 signal attributes.

    Usage:
        enricher = V8EdgeEnricher(v8db_path)
        enriched_mst = enricher.enrich_edges(mst, target_date)
    """

    def __init__(self, v8db_path: Optional[str] = None, bm: str = 'kospi'):
        """
        Initialize V8 edge enricher.

        Args:
            v8db_path: Path to V8DB_daily.xlsx
            bm: Benchmark to use ('kospi' or '3ybm')
        """
        self.analyzer = V8SignalAnalyzer(v8db_path)
        self.bm = bm
        self._signal_cache: Dict[str, V8SignalResult] = {}

    def enrich_edges(
        self,
        mst: nx.Graph,
        target_date: Optional[str] = None
    ) -> nx.Graph:
        """
        Enrich MST edges with V8 signal attributes.

        Args:
            mst: NetworkX MST graph
            target_date: Target date for signal computation

        Returns:
            MST with v8_signal attributes on edges
        """
        # Get V8 signal
        try:
            signal = self._get_signal(target_date)
        except Exception as e:
            logger.warning(f"Failed to compute V8 signal: {e}")
            return mst

        # Build factor contribution map
        factor_contrib = self._build_factor_contribution_map(signal)

        # Enrich each edge
        for u, v in mst.edges():
            edge_info = self._compute_edge_signal(u, v, factor_contrib)

            # Add attributes to edge
            mst[u][v]['v8_signal'] = edge_info.signal
            mst[u][v]['v8_direction'] = edge_info.direction
            mst[u][v]['v8_asset1_contrib'] = edge_info.asset1_contrib
            mst[u][v]['v8_asset2_contrib'] = edge_info.asset2_contrib
            mst[u][v]['v8_lag_info'] = edge_info.lag_info

        logger.info(f"Enriched {len(mst.edges())} edges with V8 signals")
        return mst

    def _get_signal(self, target_date: Optional[str] = None) -> V8SignalResult:
        """Get V8 signal, using cache if available."""
        cache_key = f"{self.bm}_{target_date or 'latest'}"

        if cache_key not in self._signal_cache:
            self._signal_cache[cache_key] = self.analyzer.compute_signal(
                self.bm, target_date
            )

        return self._signal_cache[cache_key]

    def _build_factor_contribution_map(
        self,
        signal: V8SignalResult
    ) -> Dict[str, float]:
        """
        Build map of factor keywords to contributions.

        Returns:
            Dict of factor_keyword -> contribution
        """
        contrib_map = {}

        for factor in signal.active_factors:
            contrib_map[factor.name] = factor.contribution

        return contrib_map

    def _compute_edge_signal(
        self,
        asset1: str,
        asset2: str,
        factor_contrib: Dict[str, float]
    ) -> EdgeV8Info:
        """
        Compute V8 signal for an edge.

        Args:
            asset1: First asset
            asset2: Second asset
            factor_contrib: Factor contribution map

        Returns:
            EdgeV8Info with signal details
        """
        # Get contributions for each asset
        contrib1 = self._get_asset_contribution(asset1, factor_contrib)
        contrib2 = self._get_asset_contribution(asset2, factor_contrib)

        # Combined signal (average of both contributions)
        combined = (contrib1 + contrib2) / 2 if (contrib1 != 0 or contrib2 != 0) else 0

        # Direction
        if combined > 0.2:
            direction = 'bullish'
        elif combined < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Lag info
        lag1 = 'T+1' if asset1 in T1_ASSETS else 'T+2'
        lag2 = 'T+1' if asset2 in T1_ASSETS else 'T+2'
        lag_info = f"{lag1}/{lag2}"

        return EdgeV8Info(
            asset1=asset1,
            asset2=asset2,
            signal=combined,
            direction=direction,
            asset1_contrib=contrib1,
            asset2_contrib=contrib2,
            lag_info=lag_info,
        )

    def _get_asset_contribution(
        self,
        asset: str,
        factor_contrib: Dict[str, float]
    ) -> float:
        """
        Get V8 contribution for a Market Watch asset.

        Args:
            asset: Market Watch asset name
            factor_contrib: Factor contribution map

        Returns:
            Total contribution from matching factors
        """
        keywords = ASSET_FACTOR_MAP.get(asset, [])

        if not keywords:
            return 0.0

        total = 0.0
        count = 0

        for factor_name, contrib in factor_contrib.items():
            for kw in keywords:
                if kw in factor_name:
                    total += contrib
                    count += 1
                    break

        return total

    def get_edge_summary(self, mst: nx.Graph) -> str:
        """
        Get summary of V8-enriched edges.

        Args:
            mst: Enriched MST

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("V8 Signal by Edge")
        lines.append("-" * 60)

        # Sort by signal strength
        edges = []
        for u, v, data in mst.edges(data=True):
            signal = data.get('v8_signal', 0)
            edges.append((u, v, signal, data.get('v8_direction', 'N/A')))

        edges.sort(key=lambda x: abs(x[2]), reverse=True)

        for u, v, signal, direction in edges[:10]:
            arrow = "+" if signal > 0 else "-" if signal < 0 else " "
            lines.append(f"  {u:<10} -- {v:<10}  {arrow}{abs(signal):.3f}  ({direction})")

        return "\n".join(lines)

    def get_node_v8_summary(self, mst: nx.Graph) -> Dict[str, float]:
        """
        Get average V8 signal by node (based on connected edges).

        Args:
            mst: Enriched MST

        Returns:
            Dict of node -> average v8 signal
        """
        node_signals = {}

        for node in mst.nodes():
            signals = []
            for neighbor in mst.neighbors(node):
                edge_signal = mst[node][neighbor].get('v8_signal', 0)
                signals.append(edge_signal)

            node_signals[node] = np.mean(signals) if signals else 0.0

        return node_signals
