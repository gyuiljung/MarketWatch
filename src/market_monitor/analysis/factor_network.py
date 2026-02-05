"""
Factor Network Analysis module.

Analyzes V8DB factors as a network to identify hub factors
and their influence on signal generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import networkx as nx

from ..data.v8db_loader import V8DBLoader

logger = logging.getLogger(__name__)


@dataclass
class FactorNetworkSnapshot:
    """Container for factor network analysis results."""
    corr: pd.DataFrame
    mst: nx.Graph
    betweenness: Dict[str, float]
    eigenvector: Dict[str, float]
    top_hub: str
    top3_hubs: List[Tuple[str, float]]
    hub_neighbors: List[str]
    hub_avg_corr: float
    network_sync: float
    clusters: Dict[str, List[str]]  # cluster_name -> factors
    factor_count: int
    period: Tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class FactorInfluence:
    """Factor influence on V8 signal."""
    name: str
    betweenness: float
    eigenvector: float
    z_score: float
    signal_contribution: float
    neighbor_count: int
    neighbor_avg_z: float
    is_hub: bool


class FactorNetworkAnalyzer:
    """
    Analyzes V8DB factors as a correlation network.

    Identifies:
    - Hub factors (high betweenness centrality)
    - Factor clusters (correlated groups)
    - Hub influence on signal generation
    """

    # Factor categories for clustering
    FACTOR_CATEGORIES = {
        'KR_RATE': ['국고', 'KTB', 'IRS', 'CRS', '금리', '스왑', 'NDF'],
        'KR_EQUITY': ['코스피', 'KOSPI', 'KRX'],
        'KR_MACRO': ['한국', 'CPI', 'BSI', '수지', '건설'],
        'US': ['미국', 'S&P', 'DOW', 'Treasury', 'Fed'],
        'JP': ['일본', 'JGB', '니케이', 'Nikkei'],
        'EU': ['유럽', '독일', 'EUR', 'Bund'],
        'CN': ['중국', 'SHFE', 'CNY'],
        'COMMODITY': ['금', 'Gold', 'WTI', '원유', '구리', '알루미늄'],
        'FX': ['달러', 'Dollar', 'KRW', 'JPY', 'EUR'],
        'VOLATILITY': ['VIX', '변동성'],
    }

    def __init__(self, v8db_path: Optional[str] = None):
        """
        Initialize factor network analyzer.

        Args:
            v8db_path: Path to V8DB_daily.xlsx
        """
        self.loader = V8DBLoader(v8db_path)

    def analyze(
        self,
        bm: str = 'kospi',
        window: int = 60,
        lookback: int = 252
    ) -> FactorNetworkSnapshot:
        """
        Analyze factor network.

        Args:
            bm: Benchmark ('kospi' or '3ybm')
            window: Correlation window (days)
            lookback: Data lookback period

        Returns:
            FactorNetworkSnapshot with analysis results
        """
        # Load V8DB data
        data = self.loader.load(bm, lookback=lookback)

        # Use returns for correlation
        returns = data.returns.iloc[-window:]

        # Drop columns with too many NaNs
        valid_cols = returns.columns[returns.notna().sum() > window * 0.8]
        returns = returns[valid_cols]

        # Calculate correlation matrix
        corr = returns.corr()

        # Build MST
        mst = self._build_mst(corr)

        # Calculate centralities
        bt = nx.betweenness_centrality(mst)
        ev = self._compute_eigenvector(corr)

        # Sort by betweenness
        sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
        top_hub = sorted_bt[0][0] if sorted_bt else None

        # Hub neighbors
        neighbors = list(mst.neighbors(top_hub)) if top_hub else []
        hub_corrs = [abs(corr.loc[top_hub, n]) for n in neighbors if n in corr.columns]
        hub_avg_corr = float(np.mean(hub_corrs)) if hub_corrs else 0.0

        # Network sync (handle NaN)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_values = corr.values[mask]
        network_sync = float(np.nanmean(np.abs(corr_values)))

        # Identify clusters
        clusters = self._identify_clusters(list(corr.columns))

        logger.info(f"Factor network: {len(corr.columns)} factors, hub={top_hub}")

        return FactorNetworkSnapshot(
            corr=corr,
            mst=mst,
            betweenness=bt,
            eigenvector=ev,
            top_hub=top_hub,
            top3_hubs=sorted_bt[:3],
            hub_neighbors=neighbors,
            hub_avg_corr=hub_avg_corr,
            network_sync=network_sync,
            clusters=clusters,
            factor_count=len(corr.columns),
            period=(data.period[0], data.period[1]),
        )

    def analyze_hub_influence(
        self,
        snapshot: FactorNetworkSnapshot,
        z_scores: Dict[str, float],
        contributions: Dict[str, float]
    ) -> List[FactorInfluence]:
        """
        Analyze hub factors' influence on signal.

        Args:
            snapshot: Factor network snapshot
            z_scores: Factor Z-scores
            contributions: Factor contributions to signal

        Returns:
            List of FactorInfluence objects sorted by influence
        """
        influences = []

        # Top 10 by betweenness
        top_factors = sorted(
            snapshot.betweenness.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for factor, bt in top_factors:
            # Get neighbors
            neighbors = list(snapshot.mst.neighbors(factor))
            neighbor_zs = [z_scores.get(n, 0) for n in neighbors]

            influences.append(FactorInfluence(
                name=factor,
                betweenness=bt,
                eigenvector=snapshot.eigenvector.get(factor, 0),
                z_score=z_scores.get(factor, 0),
                signal_contribution=contributions.get(factor, 0),
                neighbor_count=len(neighbors),
                neighbor_avg_z=float(np.mean(neighbor_zs)) if neighbor_zs else 0,
                is_hub=(bt == snapshot.betweenness.get(snapshot.top_hub, 0)),
            ))

        return influences

    def get_signal_reliability(
        self,
        snapshot: FactorNetworkSnapshot,
        z_scores: Dict[str, float]
    ) -> Tuple[str, float, str]:
        """
        Assess signal reliability based on hub factor state.

        Args:
            snapshot: Factor network snapshot
            z_scores: Factor Z-scores

        Returns:
            (reliability_level, score, explanation)
        """
        hub = snapshot.top_hub
        hub_z = z_scores.get(hub, 0)

        # Check hub neighbors agreement
        neighbors = snapshot.hub_neighbors
        neighbor_zs = [z_scores.get(n, 0) for n in neighbors]

        # Agreement: same direction
        if neighbor_zs:
            agreement_ratio = sum(
                1 for z in neighbor_zs if np.sign(z) == np.sign(hub_z)
            ) / len(neighbor_zs)
        else:
            agreement_ratio = 0.5

        # Hub extremity
        hub_extreme = abs(hub_z) > 2.0

        # Calculate reliability score
        score = 0.0
        reasons = []

        if hub_extreme:
            score += 0.4
            reasons.append(f"Hub({hub}) extreme Z={hub_z:.2f}")
        else:
            reasons.append(f"Hub({hub}) moderate Z={hub_z:.2f}")

        if agreement_ratio > 0.7:
            score += 0.3
            reasons.append(f"Neighbors agree ({agreement_ratio:.0%})")
        elif agreement_ratio < 0.3:
            score -= 0.2
            reasons.append(f"Neighbors disagree ({agreement_ratio:.0%})")

        if snapshot.network_sync > 0.3:
            score += 0.2
            reasons.append(f"High sync ({snapshot.network_sync:.2f})")

        # Determine level
        if score >= 0.6:
            level = "HIGH"
        elif score >= 0.3:
            level = "MEDIUM"
        else:
            level = "LOW"

        explanation = "; ".join(reasons)

        return level, score, explanation

    @staticmethod
    def _build_mst(corr: pd.DataFrame) -> nx.Graph:
        """Build MST from correlation matrix."""
        # Convert correlation to distance
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)

        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and not np.isnan(dist[i, j]):
                    G.add_edge(a1, a2, weight=dist[i, j], corr=corr.iloc[i, j])

        return nx.minimum_spanning_tree(G)

    def _compute_eigenvector(self, corr: pd.DataFrame) -> Dict[str, float]:
        """Compute eigenvector centrality."""
        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i, j]) > 0.3:
                    G.add_edge(a1, a2, weight=abs(corr.iloc[i, j]))

        try:
            return nx.eigenvector_centrality_numpy(G, weight='weight')
        except:
            return {n: 0.0 for n in corr.columns}

    def _identify_clusters(self, factors: List[str]) -> Dict[str, List[str]]:
        """Identify factor clusters by category keywords."""
        clusters = {cat: [] for cat in self.FACTOR_CATEGORIES}
        clusters['OTHER'] = []

        for factor in factors:
            matched = False
            for cat, keywords in self.FACTOR_CATEGORIES.items():
                if any(kw in factor for kw in keywords):
                    clusters[cat].append(factor)
                    matched = True
                    break
            if not matched:
                clusters['OTHER'].append(factor)

        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}

    def get_summary(
        self,
        snapshot: FactorNetworkSnapshot,
        influences: Optional[List[FactorInfluence]] = None
    ) -> str:
        """Generate text summary of factor network analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("FACTOR NETWORK ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nFactors: {snapshot.factor_count}")
        lines.append(f"Period: {snapshot.period[0].date()} ~ {snapshot.period[1].date()}")
        lines.append(f"Network Sync: {snapshot.network_sync:.4f}")

        lines.append(f"\n--- TOP HUB FACTORS ---")
        for i, (factor, bt) in enumerate(snapshot.top3_hubs, 1):
            ev = snapshot.eigenvector.get(factor, 0)
            lines.append(f"#{i} {factor[:40]:40s} Bt={bt:.4f} Ev={ev:.4f}")

        lines.append(f"\n--- HUB NEIGHBORS ({snapshot.top_hub}) ---")
        for n in snapshot.hub_neighbors[:8]:
            corr_val = snapshot.corr.loc[snapshot.top_hub, n] if n in snapshot.corr.columns else 0
            lines.append(f"  {n[:35]:35s} corr={corr_val:+.3f}")

        lines.append(f"\n--- FACTOR CLUSTERS ---")
        for cluster, factors in sorted(snapshot.clusters.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {cluster:12s}: {len(factors):2d} factors")

        if influences:
            lines.append(f"\n--- HUB INFLUENCE ON SIGNAL ---")
            for inf in influences[:5]:
                hub_mark = "*" if inf.is_hub else " "
                lines.append(
                    f"{hub_mark} {inf.name[:30]:30s} Z={inf.z_score:+5.2f} "
                    f"Contrib={inf.signal_contribution:+.3f} "
                    f"Neighbors={inf.neighbor_count}"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
