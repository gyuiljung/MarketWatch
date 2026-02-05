"""
Network topology analysis module.

Provides MST construction, centrality analysis, and multi-scale synchronization.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np
import networkx as nx

from ..core.config import Config
from ..core.constants import MST_MIN_NODES, CORRELATION_THRESHOLD, DEFAULT_SYNC_WINDOWS
from ..core.exceptions import InsufficientDataError, NetworkConstructionError

logger = logging.getLogger(__name__)


@dataclass
class NetworkSnapshot:
    """Container for network analysis results at a point in time."""
    corr: pd.DataFrame
    mst: nx.Graph
    betweenness: Dict[str, float]
    eigenvector: Dict[str, float]
    top_hub_bt: str
    top_hub_ev: str
    top3_bt: List[Tuple[str, float]]
    top3_ev: List[Tuple[str, float]]
    neighbors: List[str]
    hub_avg_corr: float
    hub_influence: float
    network_sync: float
    cat_betweenness: Optional[Dict[str, float]] = None


class NetworkAnalyzer:
    """
    Network topology analyzer using MST and centrality measures.

    Implements:
    - Minimum Spanning Tree (MST) construction
    - Betweenness centrality (topology-based)
    - Eigenvector centrality (importance-based)
    - Multi-scale network synchronization
    """

    def __init__(self, config: Config):
        """
        Initialize network analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.windows = config.analysis.windows or DEFAULT_SYNC_WINDOWS
        self.step = config.analysis.step
        self.lookback = config.analysis.lookback

    @staticmethod
    def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """
        Convert correlation matrix to Mantegna distance matrix.

        Distance = sqrt(2 * (1 - correlation))

        Args:
            corr: Correlation matrix

        Returns:
            Distance matrix
        """
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)

    @staticmethod
    def build_mst(corr: pd.DataFrame) -> nx.Graph:
        """
        Build Minimum Spanning Tree from correlation matrix.

        Args:
            corr: Correlation matrix

        Returns:
            NetworkX Graph representing MST
        """
        dist = NetworkAnalyzer.corr_to_distance(corr)

        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j:
                    G.add_edge(
                        a1, a2,
                        weight=dist.iloc[i, j],
                        corr=corr.iloc[i, j]
                    )

        return nx.minimum_spanning_tree(G)

    @staticmethod
    def calc_network_sync(corr: pd.DataFrame) -> float:
        """
        Calculate network synchronization (average absolute correlation).

        Args:
            corr: Correlation matrix

        Returns:
            Average correlation (excluding diagonal)
        """
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return float(corr.values[mask].mean())

    def get_category(self, asset: str) -> str:
        """
        Get category for an asset.

        Args:
            asset: Asset name

        Returns:
            Category name or 'OTHER'
        """
        for cat, assets in self.config.categories.items():
            if asset in assets:
                return cat

        # Check clusters
        if self.config.clusters:
            for cluster_name, cluster in self.config.clusters.items():
                if asset in cluster.assets:
                    return cluster_name

        return 'OTHER'

    def compute_snapshot(
        self,
        returns: pd.DataFrame,
        network_assets: Optional[List[str]] = None,
        window: Optional[int] = None
    ) -> NetworkSnapshot:
        """
        Compute network snapshot for current state.

        Args:
            returns: Returns DataFrame
            network_assets: Assets to include (default: all)
            window: Lookback window (default: from config)

        Returns:
            NetworkSnapshot with all metrics
        """
        if window is None:
            window = self.windows[-1]

        # Filter to network assets
        if network_assets:
            cols = [c for c in network_assets if c in returns.columns]
        else:
            cols = list(returns.columns)

        if len(cols) < MST_MIN_NODES:
            raise InsufficientDataError(MST_MIN_NODES, len(cols), "network construction")

        # Get recent data
        recent = returns[cols].iloc[-window:] if len(returns) >= window else returns[cols]
        corr = recent.corr()

        # Build MST
        try:
            mst = self.build_mst(corr)
        except Exception as e:
            raise NetworkConstructionError(str(e), len(cols)) from e

        # Calculate centralities
        bt = nx.betweenness_centrality(mst)
        ev = self._compute_eigenvector(corr)

        # Sort by centrality
        sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
        sorted_ev = sorted(ev.items(), key=lambda x: x[1], reverse=True)

        # Top hubs
        top_hub_bt = sorted_bt[0][0] if sorted_bt else None
        top_hub_ev = sorted_ev[0][0] if sorted_ev else None

        # Hub neighbors and correlations
        neighbors = list(mst.neighbors(top_hub_bt)) if top_hub_bt else []
        hub_corrs = [abs(corr.loc[top_hub_bt, n]) for n in neighbors if n in corr.columns]
        hub_avg_corr = float(np.mean(hub_corrs)) if hub_corrs else 0.0

        # Hub influence = betweenness * avg correlation
        hub_influence = sorted_bt[0][1] * hub_avg_corr if sorted_bt else 0.0

        # Category betweenness
        cat_bt = {}
        for cat in self.config.categories.keys():
            cat_assets = self.config.categories[cat]
            cat_bt[cat] = sum(bt.get(a, 0) for a in cat_assets)

        return NetworkSnapshot(
            corr=corr,
            mst=mst,
            betweenness=bt,
            eigenvector=ev,
            top_hub_bt=top_hub_bt,
            top_hub_ev=top_hub_ev,
            top3_bt=sorted_bt[:3],
            top3_ev=sorted_ev[:3],
            neighbors=neighbors,
            hub_avg_corr=hub_avg_corr,
            hub_influence=hub_influence,
            network_sync=self.calc_network_sync(corr),
            cat_betweenness=cat_bt,
        )

    def _compute_eigenvector(self, corr: pd.DataFrame) -> Dict[str, float]:
        """
        Compute eigenvector centrality on full correlation graph.

        Args:
            corr: Correlation matrix

        Returns:
            Dict of asset -> eigenvector centrality
        """
        # Build graph with edges for significant correlations
        G_full = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i, j]) > CORRELATION_THRESHOLD:
                    G_full.add_edge(a1, a2, weight=abs(corr.iloc[i, j]))

        try:
            return nx.eigenvector_centrality_numpy(G_full, weight='weight')
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence) as e:
            logger.warning(f"Eigenvector centrality failed: {e}")
            return {n: 0.0 for n in corr.columns}

    def compute_multi_scale_sync(
        self,
        returns: pd.DataFrame,
        network_assets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute synchronization at multiple time scales.

        Args:
            returns: Returns DataFrame
            network_assets: Assets to include

        Returns:
            Dict of 'sync_{window}d' -> sync value
        """
        if network_assets:
            cols = [c for c in network_assets if c in returns.columns]
        else:
            cols = list(returns.columns)

        result = {}
        for w in self.windows:
            if len(returns) >= w:
                recent = returns[cols].iloc[-w:]
                corr = recent.corr()
                result[f'sync_{w}d'] = self.calc_network_sync(corr)

        return result

    def compute_timeseries(
        self,
        returns: pd.DataFrame,
        network_assets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute network metrics over time.

        Args:
            returns: Returns DataFrame
            network_assets: Assets to include

        Returns:
            DataFrame with time series of network metrics
        """
        if network_assets:
            cols = [c for c in network_assets if c in returns.columns]
        else:
            cols = list(returns.columns)

        window = max(self.windows)
        step = self.step

        records = []
        total = (len(returns) - window) // step

        for idx, i in enumerate(range(window, len(returns), step)):
            if idx % 20 == 0:
                print(f"\r  Computing network... {idx}/{total}", end='')

            date = returns.index[i]

            try:
                record = {'date': date}

                # Multi-scale sync
                for w in self.windows:
                    subset = returns[cols].iloc[i-w:i]
                    corr = subset.corr()
                    record[f'sync_{w}d'] = self.calc_network_sync(corr)

                # Hub metrics (using largest window)
                subset = returns[cols].iloc[i-window:i]
                snap = self.compute_snapshot(subset, cols, window)
                record['top_hub_bt'] = snap.top_hub_bt
                record['top_hub_ev'] = snap.top_hub_ev
                record['hub_influence'] = snap.hub_influence
                record['hub_avg_corr'] = snap.hub_avg_corr

                # Category betweenness
                if snap.cat_betweenness:
                    for cat, val in snap.cat_betweenness.items():
                        record[f'bt_{cat}'] = val

                records.append(record)

            except Exception as e:
                logger.debug(f"Skipping {date}: {e}")
                continue

        print(f"\r  Computing network... Done ({total} observations)")

        df = pd.DataFrame(records).set_index('date')

        # Add percentiles
        lookback_periods = self.lookback // step
        for col in ['hub_influence'] + [f'sync_{w}d' for w in self.windows]:
            if col in df.columns:
                df[f'{col}_pct'] = df[col].rolling(lookback_periods, min_periods=20).apply(
                    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
                )

        return df
