"""
L2: Network Analyzer
====================
MST 구조 + Hub centrality 분석.
market_monitor ClusteredNetworkAnalyzer 기반.
"""
import numpy as np
import pandas as pd
import networkx as nx


class NetworkAnalyzer:
    """MST 기반 네트워크 분석"""

    def __init__(self, config: dict):
        self.config = config
        self.analysis = config.get('analysis', {'windows': [5, 20, 60]})
        self.windows = self.analysis.get('windows', [5, 20, 60])

    @staticmethod
    def corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        return dist

    @staticmethod
    def build_mst(corr: pd.DataFrame) -> nx.Graph:
        dist = NetworkAnalyzer.corr_to_distance(corr)
        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j:
                    G.add_edge(a1, a2, weight=dist[i, j], corr=corr.iloc[i, j])
        return nx.minimum_spanning_tree(G)

    @staticmethod
    def calc_network_sync(corr: pd.DataFrame) -> float:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr.values[mask].mean()

    def get_asset_color(self, asset: str) -> str:
        for cat, assets in self.config.get('core_categories', {}).items():
            if asset in assets:
                return self.config.get('core_colors', {}).get(cat, '#8b949e')
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            if asset in cluster_data.get('assets', {}) or asset == cluster_data.get('representative'):
                return cluster_data.get('color', '#8b949e')
        return '#8b949e'

    def compute_snapshot(self, returns: pd.DataFrame, network_assets: list, window: int = None) -> dict:
        if window is None:
            window = self.windows[-1]

        cols = [c for c in network_assets if c in returns.columns]
        recent = returns[cols].iloc[-window:] if len(returns) >= window else returns[cols]
        corr = recent.corr()
        mst = self.build_mst(corr)

        bt = nx.betweenness_centrality(mst)

        # Eigenvector on full correlation graph
        G_full = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i, j]) > 0.1:
                    G_full.add_edge(a1, a2, weight=abs(corr.iloc[i, j]))

        try:
            ev = nx.eigenvector_centrality_numpy(G_full, weight='weight')
        except Exception:
            ev = {n: 0 for n in corr.columns}

        sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
        sorted_ev = sorted(ev.items(), key=lambda x: x[1], reverse=True)

        top_hub_bt = sorted_bt[0][0] if sorted_bt else None
        top_hub_ev = sorted_ev[0][0] if sorted_ev else None

        neighbors = list(mst.neighbors(top_hub_bt)) if top_hub_bt else []
        hub_corrs = [abs(corr.loc[top_hub_bt, n]) for n in neighbors if n in corr.columns]
        hub_avg_corr = np.mean(hub_corrs) if hub_corrs else 0

        return {
            'corr': corr,
            'mst': mst,
            'betweenness': bt,
            'eigenvector': ev,
            'top_hub_bt': top_hub_bt,
            'top_hub_ev': top_hub_ev,
            'top3_bt': sorted_bt[:3],
            'top3_ev': sorted_ev[:3],
            'neighbors': neighbors,
            'hub_avg_corr': hub_avg_corr,
            'hub_influence': sorted_bt[0][1] * hub_avg_corr if sorted_bt else 0,
            'network_sync': self.calc_network_sync(corr),
        }

    def compute_multi_scale_sync(self, returns: pd.DataFrame, network_assets: list) -> dict:
        cols = [c for c in network_assets if c in returns.columns]
        result = {}
        for w in self.windows:
            if len(returns) >= w:
                recent = returns[cols].iloc[-w:]
                corr = recent.corr()
                result[f'sync_{w}d'] = self.calc_network_sync(corr)
        return result

    def compute_timeseries(self, returns: pd.DataFrame, network_assets: list) -> pd.DataFrame:
        step = self.analysis.get('step', 5)
        window = max(self.windows)
        cols = [c for c in network_assets if c in returns.columns]

        records = []
        total = (len(returns) - window) // step

        for idx, i in enumerate(range(window, len(returns), step)):
            if idx % 20 == 0:
                print(f"  Computing network... {idx}/{total}", end='\r')

            date = returns.index[i]
            try:
                record = {'date': date}
                for w in self.windows:
                    subset = returns[cols].iloc[i-w:i]
                    corr = subset.corr()
                    record[f'sync_{w}d'] = self.calc_network_sync(corr)

                    if w == self.windows[-1]:
                        mst = self.build_mst(corr)
                        bt = nx.betweenness_centrality(mst)
                        sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
                        record['top_hub_bt'] = sorted_bt[0][0] if sorted_bt else None
                        record['hub_influence'] = sorted_bt[0][1] if sorted_bt else 0

                records.append(record)
            except Exception:
                pass

        print(f"  Computing network... Done ({total} observations)")

        df = pd.DataFrame(records).set_index('date')

        lookback = self.analysis.get('lookback', 252) // step
        for col in ['hub_influence'] + [f'sync_{w}d' for w in self.windows]:
            if col in df.columns:
                df[f'{col}_pct'] = df[col].rolling(lookback, min_periods=20).apply(
                    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
                )

        return df
