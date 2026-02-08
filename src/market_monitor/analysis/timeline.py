"""
Timeline tracking module.

Tracks weekly changes in hub centrality and TE flows.
"""

from typing import List, Optional
import logging

import pandas as pd
import numpy as np
import networkx as nx

from ..core.config import Config

logger = logging.getLogger(__name__)


class TimelineTracker:
    """
    Timeline tracker for hub and TE changes.

    Tracks weekly snapshots of:
    - Top hub changes
    - TE flow changes
    """

    def __init__(self, config: Config):
        """
        Initialize timeline tracker.

        Args:
            config: Configuration object
        """
        self.config = config
        analysis = config.analysis
        self.hub_window = analysis.hub_window
        self.te_window = analysis.te_window_long
        self.weekly_step = 5  # Trading days per week

    def compute_hub_timeline(
        self,
        returns: pd.DataFrame,
        network_assets: List[str],
        n_weeks: int = 8
    ) -> pd.DataFrame:
        """
        Track top hub changes over past n weeks.

        Args:
            returns: Returns DataFrame
            network_assets: Assets for network analysis
            n_weeks: Number of weeks to track

        Returns:
            DataFrame with weekly hub snapshots
        """
        records = []

        for week in range(n_weeks):
            end_idx = len(returns) - week * self.weekly_step
            start_idx = end_idx - self.hub_window

            if start_idx < 0:
                break

            subset = returns.iloc[start_idx:end_idx]
            cols = [c for c in network_assets if c in subset.columns]
            subset = subset[cols].dropna(axis=1, how='all')

            if len(subset.columns) < 3:
                continue

            corr = subset.corr()

            # Build MST
            dist = np.sqrt(2 * (1 - corr.values))
            np.fill_diagonal(dist, 0)

            G = nx.Graph()
            for i, a1 in enumerate(corr.columns):
                for j, a2 in enumerate(corr.columns):
                    if i < j:
                        G.add_edge(a1, a2, weight=dist[i, j])

            mst = nx.minimum_spanning_tree(G)
            bt = nx.betweenness_centrality(mst)

            # Top hub
            top_hub = max(bt.keys(), key=lambda k: bt[k])
            top_bt = bt[top_hub]

            # Top 3 hubs
            sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)[:3]

            # Network sync
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sync = corr.values[mask].mean()

            records.append({
                'week_ago': week,
                'end_date': returns.index[end_idx - 1],
                'top_hub': top_hub,
                'top_bt': top_bt,
                'top3': sorted_bt,
                'network_sync': sync,
            })

        return pd.DataFrame(records)

    def compute_te_timeline(
        self,
        returns: pd.DataFrame,
        te_calc,
        n_weeks: int = 4,
        top_n: int = 5
    ) -> list:
        """
        Track top TE flows over past n weeks.

        Args:
            returns: Returns DataFrame
            te_calc: TransferEntropyCalculator instance
            n_weeks: Number of weeks to track
            top_n: Top flows to track per week

        Returns:
            List of weekly TE snapshots
        """
        records = []

        for week in range(n_weeks):
            end_idx = len(returns) - week * self.weekly_step
            start_idx = end_idx - self.te_window

            if start_idx < 0:
                break

            subset = returns.iloc[start_idx:end_idx]

            try:
                te_all = te_calc.compute_all_pairs(subset)
                net_flow = te_calc.compute_net_flow_all(subset)

                top_flows = sorted(
                    net_flow.items(),
                    key=lambda x: x[1].flow_strength,
                    reverse=True
                )[:top_n]

                sig_count = sum(1 for v in te_all.values() if v.significant)

                records.append({
                    'week_ago': week,
                    'end_date': returns.index[end_idx - 1],
                    'top_flows': top_flows,
                    'sig_count': sig_count,
                    'total_pairs': len(te_all),
                })
            except Exception as e:
                logger.warning(f"TE timeline week {week} failed: {e}")
                continue

        return records

    def detect_hub_changes(self, hub_timeline: pd.DataFrame) -> List[dict]:
        """
        Detect significant hub changes.

        Args:
            hub_timeline: Hub timeline DataFrame

        Returns:
            List of change events
        """
        changes = []

        if len(hub_timeline) < 2:
            return changes

        for i in range(len(hub_timeline) - 1):
            current = hub_timeline.iloc[i]
            prev = hub_timeline.iloc[i + 1]

            if current['top_hub'] != prev['top_hub']:
                changes.append({
                    'date': current['end_date'],
                    'from': prev['top_hub'],
                    'to': current['top_hub'],
                    'sync_change': current['network_sync'] - prev['network_sync'],
                })

        return changes
