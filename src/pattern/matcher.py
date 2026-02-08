"""
Pattern Matcher
===============
상태벡터 + 코사인 유사도 기반 유사 구간 매칭.
market_monitor/pattern_matcher.py 기반 이식.

방법론:
1. Hub 필터: 같은 국가 + 같은 카테고리의 hub였던 시점만
2. State Vector: [sync, hub_bt, rv_avg, rv_vix, rv_jgb, rv_ktb, corr_carry, corr_krjp]
3. Weighted cosine similarity
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple

ASSET_META = {
    'USDKRW': {'country': 'KR', 'category': 'FX'},
    'USDJPY': {'country': 'JP', 'category': 'FX'},
    'DXY': {'country': 'US', 'category': 'FX'},
    'EURUSD': {'country': 'EU', 'category': 'FX'},
    'SPX': {'country': 'US', 'category': 'EQ'},
    'NKY': {'country': 'JP', 'category': 'EQ'},
    'KOSPI': {'country': 'KR', 'category': 'EQ'},
    'HSI': {'country': 'CN', 'category': 'EQ'},
    'DAX': {'country': 'EU', 'category': 'EQ'},
    'VIX': {'country': 'US', 'category': 'VOL'},
    'GOLD': {'country': 'GLOBAL', 'category': 'CMD'},
    'WTI': {'country': 'GLOBAL', 'category': 'CMD'},
    'KTB_2Y': {'country': 'KR', 'category': 'IR'},
    'KTB_10Y': {'country': 'KR', 'category': 'IR'},
    'KTB_30Y': {'country': 'KR', 'category': 'IR'},
    'JGB_2Y': {'country': 'JP', 'category': 'IR'},
    'JGB_10Y': {'country': 'JP', 'category': 'IR'},
    'JGB_30Y': {'country': 'JP', 'category': 'IR'},
    'UST_2Y': {'country': 'US', 'category': 'IR'},
    'UST_10Y': {'country': 'US', 'category': 'IR'},
    'UST_30Y': {'country': 'US', 'category': 'IR'},
    'Bund_10Y': {'country': 'EU', 'category': 'IR'},
}


def get_asset_group(asset: str) -> Tuple[str, str]:
    meta = ASSET_META.get(asset, {'country': 'OTHER', 'category': 'OTHER'})
    return meta['country'], meta['category']


def calc_rv_percentile(series: pd.Series, window: int = 5, lookback: int = 252) -> pd.Series:
    rv = series.rolling(window).std() * np.sqrt(252) * 100
    return rv.rolling(lookback, min_periods=60).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
    )


class HistoricalPatternMatcher:
    """과거 유사 패턴 찾기"""

    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame,
                 window: int = 60, step: int = 5):
        self.returns = returns
        self.prices = prices
        self.window = window
        self.step = step
        self.rv_pct = self._compute_all_rv()
        self.network_states = self._compute_network_states()

    def _compute_all_rv(self) -> pd.DataFrame:
        result = pd.DataFrame(index=self.returns.index)
        for col in self.returns.columns:
            result[col] = calc_rv_percentile(self.returns[col])
        return result

    def _compute_network_states(self) -> pd.DataFrame:
        records = []
        total = (len(self.returns) - self.window) // self.step
        print(f"  Computing network states... (n={total})")

        for idx, i in enumerate(range(self.window, len(self.returns), self.step)):
            if idx % 50 == 0:
                print(f"    Progress: {idx}/{total}", end='\r')

            date = self.returns.index[i]
            subset = self.returns.iloc[i-self.window:i]

            try:
                corr = subset.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                sync = corr.values[mask].mean()

                # MST + hub
                dist = np.sqrt(2 * (1 - corr.values))
                np.fill_diagonal(dist, 0)
                G = nx.Graph()
                for ci, a1 in enumerate(corr.columns):
                    for cj, a2 in enumerate(corr.columns):
                        if ci < cj:
                            G.add_edge(a1, a2, weight=dist[ci, cj])
                mst = nx.minimum_spanning_tree(G)
                bt = nx.betweenness_centrality(mst)
                top_hub = max(bt.keys(), key=lambda k: bt[k])
                top_bt = bt[top_hub]

                corr_carry = corr.loc['USDJPY', 'NKY'] if 'USDJPY' in corr.index and 'NKY' in corr.columns else 0
                corr_krjp = corr.loc['USDKRW', 'USDJPY'] if 'USDKRW' in corr.index and 'USDJPY' in corr.columns else 0

                hub_country, hub_category = get_asset_group(top_hub)

                records.append({
                    'date': date, 'sync': sync, 'top_hub': top_hub, 'top_bt': top_bt,
                    'hub_country': hub_country, 'hub_category': hub_category,
                    'corr_carry': corr_carry, 'corr_krjp': corr_krjp,
                })
            except Exception:
                pass

        print(f"    Progress: Done ({len(records)} observations)")
        return pd.DataFrame(records).set_index('date')

    def _build_state_vector(self, date: pd.Timestamp) -> Optional[np.ndarray]:
        if date not in self.network_states.index:
            return None

        ns = self.network_states.loc[date]
        rv_date = self.rv_pct.index[self.rv_pct.index.get_indexer([date], method='nearest')[0]]
        rv = self.rv_pct.loc[rv_date]

        components = [
            min(ns['sync'] * 100 / 0.5, 100),
            ns['top_bt'] * 100,
            rv.mean() if not rv.isna().all() else 50,
            rv.get('VIX', 50),
            rv.get('JGB_10Y', 50),
            rv.get('KTB_10Y', 50),
            (ns['corr_carry'] + 1) * 50,
            (ns['corr_krjp'] + 1) * 50,
        ]

        return np.array([50 if pd.isna(v) else v for v in components])

    def _weighted_cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        weights = np.array([1.5, 1.0, 1.0, 1.5, 1.2, 1.0, 1.2, 0.8])
        if v1[3] > 80 or v2[3] > 80:
            weights[3] = 2.5
            weights[0] = 2.0

        w1 = v1 * weights
        w2 = v2 * weights
        dot = np.dot(w1, w2)
        norm = np.linalg.norm(w1) * np.linalg.norm(w2)
        return dot / norm if norm > 0 else 0

    def find_similar_periods(self, target_date: pd.Timestamp = None,
                             top_k: int = 5, min_gap_days: int = 30) -> pd.DataFrame:
        if target_date is None:
            target_date = self.network_states.index[-1]

        print(f"\n[PATTERN MATCHER] Target: {target_date.strftime('%Y-%m-%d')}")

        target_state = self._build_state_vector(target_date)
        if target_state is None:
            print("  ERROR: Cannot build state vector")
            return pd.DataFrame()

        target_ns = self.network_states.loc[target_date]
        target_country = target_ns['hub_country']
        target_category = target_ns['hub_category']

        print(f"  Target Hub: {target_ns['top_hub']} ({target_country}/{target_category})")

        # Filter by hub country + category
        filtered = self.network_states[
            (self.network_states['hub_country'] == target_country) &
            (self.network_states['hub_category'] == target_category) &
            (self.network_states.index < target_date - pd.Timedelta(days=min_gap_days))
        ].index

        print(f"  Filtered periods: {len(filtered)}")
        if len(filtered) == 0:
            return pd.DataFrame()

        # Compute similarities
        similarities = []
        for date in filtered:
            state = self._build_state_vector(date)
            if state is not None:
                sim = self._weighted_cosine(target_state, state)
                similarities.append({'date': date, 'similarity': sim,
                                     'hub': self.network_states.loc[date, 'top_hub'],
                                     'sync': self.network_states.loc[date, 'sync'],
                                     'hub_bt': self.network_states.loc[date, 'top_bt']})

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Select with min gap
        selected = []
        for s in similarities:
            too_close = any(abs((s['date'] - prev['date']).days) < min_gap_days for prev in selected)
            if not too_close:
                selected.append(s)
                if len(selected) >= top_k:
                    break

        if selected:
            avg_sim = np.mean([s['similarity'] for s in selected])
            print(f"  Avg similarity: {avg_sim:.3f}")

        # Add forward returns
        records = []
        for s in selected:
            record = {k: v for k, v in s.items()}
            idx = self.returns.index.get_loc(s['date'])

            for h in [5, 20, 60]:
                if idx + h < len(self.returns):
                    future_date = self.returns.index[idx + h]
                    for asset in ['SPX', 'NKY', 'KOSPI', 'USDJPY', 'USDKRW', 'VIX']:
                        if asset in self.prices.columns:
                            start_p = self.prices.loc[s['date'], asset]
                            end_p = self.prices.loc[future_date, asset]
                            if pd.notna(start_p) and pd.notna(end_p) and start_p != 0:
                                record[f'{h}d_{asset}'] = (end_p / start_p - 1) * 100

            records.append(record)

        return pd.DataFrame(records)
