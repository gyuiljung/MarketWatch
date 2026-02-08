#!/usr/bin/env python3
"""
=============================================================================
MARKET NETWORK MONITOR v2.3 - Dual Window Analysis
=============================================================================
- Network Sync: Multi-scale (5/20/60d)
- Transfer Entropy: Dual window (60d short-term + 252d structural)
- Tail Dependence: Dual window (60d + 252d)
- Surrogate-based significance testing
=============================================================================
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import yaml
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter


# =============================================================================
# DATA LOADER (Clustered)
# =============================================================================
class ClusteredDataLoader:
    def __init__(self, filepath: str, config: dict):
        self.filepath = filepath
        self.config = config
        self.prices = None
        self.returns = None
        self.core_assets = []
        self.cluster_reps = []
        self.all_assets = []
        
    def load(self) -> 'ClusteredDataLoader':
        print(f"  Loading {self.filepath}...")
        
        df = pd.read_excel(self.filepath, header=None)
        data = df.iloc[4:, :].copy()
        asset_names = df.iloc[2, :].tolist()
        data.columns = asset_names
        data = data.rename(columns={asset_names[0]: 'Date'})
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        selected = pd.DataFrame()
        
        # Load Core assets
        core_cfg = self.config.get('core_assets', {})
        for short_name, full_name in core_cfg.items():
            if full_name in data.columns:
                selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                self.core_assets.append(short_name)
                self.all_assets.append(short_name)
        
        # Load Cluster assets
        clusters_cfg = self.config.get('clusters', {})
        for cluster_name, cluster_data in clusters_cfg.items():
            assets = cluster_data.get('assets', {})
            rep = cluster_data.get('representative')
            
            for short_name, full_name in assets.items():
                if full_name in data.columns:
                    selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                    self.all_assets.append(short_name)
                    if short_name == rep:
                        self.cluster_reps.append(short_name)
        
        self.prices = selected
        
        # Calculate returns
        returns = np.log(selected / selected.shift(1))
        for col in self.config.get('rate_assets', []):
            if col in returns.columns:
                returns[col] = selected[col].diff()
        
        # Filter working days
        if 'USDKRW' in returns.columns:
            working = returns['USDKRW'][returns['USDKRW'] != 0].index
            returns = returns.loc[working]
            self.prices = self.prices.loc[working]
        
        self.returns = returns.dropna()
        self.prices = self.prices.loc[self.returns.index]
        
        print(f"  Core assets ({len(self.core_assets)}): {', '.join(self.core_assets[:8])}...")
        print(f"  Cluster reps ({len(self.cluster_reps)}): {', '.join(self.cluster_reps)}")
        print(f"  Period: {self.returns.index[0].date()} ~ {self.returns.index[-1].date()}")
        print(f"  Working days: {len(self.returns)}")
        
        return self
    
    def get_network_assets(self) -> list:
        """Core + Cluster representatives for network"""
        return self.core_assets + self.cluster_reps


# =============================================================================
# TRANSFER ENTROPY (v2.3 - All-Pairs, Optimized)
# =============================================================================
class TransferEntropyCalculator:
    """
    All-pairs Transfer Entropy with surrogate testing.
    
    Optimizations:
    1. Pre-discretize all assets once
    2. Batch matrix computation
    3. Reduced surrogates (30) - sufficient for p-value estimation
    4. Report top N significant pairs
    """
    def __init__(self, config: dict):
        te_cfg = config.get('transfer_entropy', {})
        self.bins = te_cfg.get('bins', 6)
        self.lag = te_cfg.get('lag', 1)
        self.n_surrogates = te_cfg.get('n_surrogates', 30)  # Reduced for efficiency
        self.alpha = te_cfg.get('alpha', 0.05)
        self.top_n = te_cfg.get('top_n', 15)  # Report top N pairs
    
    def _discretize_all(self, returns: pd.DataFrame) -> np.ndarray:
        """Pre-discretize all assets at once"""
        n_assets = len(returns.columns)
        T = len(returns)
        result = np.zeros((n_assets, T), dtype=int)
        
        for i, col in enumerate(returns.columns):
            series = returns[col].values
            valid = ~np.isnan(series)
            ranks = np.argsort(np.argsort(series[valid]))
            n_valid = np.sum(valid)
            result[i, valid] = np.minimum(ranks * self.bins // n_valid, self.bins - 1)
        
        return result
    
    def _compute_te_matrix(self, discretized: np.ndarray) -> np.ndarray:
        """Compute all pairwise TE in one pass"""
        n_assets, T = discretized.shape
        n = T - self.lag
        results = np.zeros((n_assets, n_assets))
        
        for tgt in range(n_assets):
            tgt_d = discretized[tgt]
            tgt_future = tgt_d[self.lag:]
            tgt_past = tgt_d[:-self.lag]
            
            # P(Y_t, Y_{t-1}) - same for all sources
            joint_yy = np.zeros((self.bins, self.bins))
            for t in range(n):
                joint_yy[tgt_future[t], tgt_past[t]] += 1
            joint_yy /= n
            
            # P(Y_{t-1})
            marginal_y = np.bincount(tgt_past, minlength=self.bins) / n
            
            for src in range(n_assets):
                if src == tgt:
                    continue
                
                src_past = discretized[src, :-self.lag]
                
                # P(Y_t, Y_{t-1}, X_{t-1})
                joint_yyx = np.zeros((self.bins, self.bins, self.bins))
                for t in range(n):
                    joint_yyx[tgt_future[t], tgt_past[t], src_past[t]] += 1
                joint_yyx /= n
                
                # P(Y_{t-1}, X_{t-1})
                marginal_yx = np.zeros((self.bins, self.bins))
                for t in range(n):
                    marginal_yx[tgt_past[t], src_past[t]] += 1
                marginal_yx /= n
                
                # Compute TE
                te = 0.0
                for yt in range(self.bins):
                    for yp in range(self.bins):
                        for xp in range(self.bins):
                            p_joint = joint_yyx[yt, yp, xp]
                            if p_joint > 1e-10:
                                p_yy = joint_yy[yt, yp]
                                p_yx = marginal_yx[yp, xp]
                                p_y = marginal_y[yp]
                                if p_yy > 1e-10 and p_yx > 1e-10 and p_y > 1e-10:
                                    te += p_joint * np.log2((p_joint * p_y) / (p_yx * p_yy))
                
                results[src, tgt] = max(0, te)
        
        return results
    
    def compute_all_pairs(self, returns: pd.DataFrame) -> dict:
        """Compute TE for ALL pairs with surrogate testing"""
        assets = list(returns.columns)
        n_assets = len(assets)
        
        # Pre-discretize
        discretized = self._discretize_all(returns)
        
        # Raw TE matrix
        te_raw = self._compute_te_matrix(discretized)
        
        # Surrogate distribution
        te_surrogates = np.zeros((self.n_surrogates, n_assets, n_assets))
        for s in range(self.n_surrogates):
            perm = np.random.permutation(len(returns))
            shuffled = discretized[:, perm]
            te_surrogates[s] = self._compute_te_matrix(shuffled)
        
        # Compute Z-scores and p-values
        te_mean = np.mean(te_surrogates, axis=0)
        te_std = np.std(te_surrogates, axis=0) + 1e-10
        te_z = (te_raw - te_mean) / te_std
        p_values = np.mean(te_surrogates >= te_raw[np.newaxis, :, :], axis=0)
        
        # Build results dict
        results = {}
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i != j:
                    results[f'{src}â†’{tgt}'] = {
                        'te_raw': te_raw[i, j],
                        'te_z': te_z[i, j],
                        'p_value': p_values[i, j],
                        'significant': p_values[i, j] < self.alpha,
                        'surrogate_mean': te_mean[i, j],
                        'surrogate_std': te_std[i, j]
                    }
        
        return results
    
    def compute_net_flow_all(self, returns: pd.DataFrame) -> dict:
        """Compute Net Information Flow for all unique pairs"""
        all_te = self.compute_all_pairs(returns)
        assets = list(returns.columns)
        
        results = {}
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i < j:  # Unique pairs only
                    key_ab = f'{a}â†’{b}'
                    key_ba = f'{b}â†’{a}'
                    
                    if key_ab in all_te and key_ba in all_te:
                        te_ab = all_te[key_ab]
                        te_ba = all_te[key_ba]
                        
                        net_flow = te_ab['te_z'] - te_ba['te_z']
                        
                        results[f'{a}â‡„{b}'] = {
                            'te_ab': te_ab,
                            'te_ba': te_ba,
                            'net_flow_z': net_flow,
                            'dominant': a if net_flow > 0 else b,
                            'dominant_direction': f'{a}â†’{b}' if net_flow > 0 else f'{b}â†’{a}',
                            'flow_strength': abs(net_flow),
                            'both_significant': te_ab['significant'] and te_ba['significant'],
                            'any_significant': te_ab['significant'] or te_ba['significant']
                        }
        
        return results
    
    def get_top_significant(self, all_te: dict, n: int = None) -> list:
        """Get top N significant TE pairs by Z-score"""
        if n is None:
            n = self.top_n
        
        sig_pairs = [(k, v) for k, v in all_te.items() if v['significant']]
        sig_pairs.sort(key=lambda x: x[1]['te_z'], reverse=True)
        return sig_pairs[:n]
    
    def get_top_net_flows(self, net_flow: dict, n: int = None) -> list:
        """Get top N net flows by strength"""
        if n is None:
            n = self.top_n
        
        flows = list(net_flow.items())
        flows.sort(key=lambda x: x[1]['flow_strength'], reverse=True)
        return flows[:n]
    
    # Legacy methods for backward compatibility
    def compute_for_pairs(self, returns: pd.DataFrame, pairs: list = None) -> dict:
        """Legacy: compute for specific pairs"""
        all_te = self.compute_all_pairs(returns)
        if pairs is None:
            return all_te
        
        result = {}
        for pair in pairs:
            key = f'{pair[0]}â†’{pair[1]}'
            if key in all_te:
                result[key] = all_te[key]['te_raw']
        return result
    
    def compute_with_significance(self, returns: pd.DataFrame, pairs: list = None) -> dict:
        """Legacy: compute with significance for specific pairs"""
        all_te = self.compute_all_pairs(returns)
        if pairs is None:
            return all_te
        
        result = {}
        for pair in pairs:
            key = f'{pair[0]}â†’{pair[1]}'
            if key in all_te:
                result[key] = all_te[key]
        return result
    
    def compute_net_flow(self, returns: pd.DataFrame, pairs: list = None) -> dict:
        """Legacy: compute net flow (now computes all pairs)"""
        return self.compute_net_flow_all(returns)


# =============================================================================
# NETWORK ANALYZER (Clustered)
# =============================================================================
class ClusteredNetworkAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.analysis = config.get('analysis', {'windows': [5, 20, 60]})
        self.windows = self.analysis.get('windows', [5, 20, 60])
        
    @staticmethod
    def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    
    @staticmethod
    def build_mst(corr: pd.DataFrame) -> nx.Graph:
        dist = ClusteredNetworkAnalyzer.corr_to_distance(corr)
        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j:
                    G.add_edge(a1, a2, weight=dist.iloc[i, j], corr=corr.iloc[i, j])
        return nx.minimum_spanning_tree(G)
    
    @staticmethod
    def calc_network_sync(corr: pd.DataFrame) -> float:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr.values[mask].mean()
    
    def get_core_category(self, asset: str) -> str:
        for cat, assets in self.config.get('core_categories', {}).items():
            if asset in assets:
                return cat
        # Check clusters
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            if asset in cluster_data.get('assets', {}):
                return cluster_name
            if asset == cluster_data.get('representative'):
                return cluster_name
        return 'OTHER'
    
    def get_asset_color(self, asset: str) -> str:
        # Core colors
        for cat, assets in self.config.get('core_categories', {}).items():
            if asset in assets:
                return self.config.get('core_colors', {}).get(cat, '#8b949e')
        # Cluster colors
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            if asset in cluster_data.get('assets', {}) or asset == cluster_data.get('representative'):
                return cluster_data.get('color', '#8b949e')
        return '#8b949e'
    
    def compute_snapshot(self, returns: pd.DataFrame, network_assets: list, window: int = None) -> dict:
        if window is None:
            window = self.windows[-1]
        
        # Filter to network assets only
        cols = [c for c in network_assets if c in returns.columns]
        recent = returns[cols].iloc[-window:] if len(returns) >= window else returns[cols]
        corr = recent.corr()
        mst = self.build_mst(corr)
        
        # Centralities
        bt = nx.betweenness_centrality(mst)
        
        # Eigenvector on full correlation graph
        G_full = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i, j]) > 0.1:
                    G_full.add_edge(a1, a2, weight=abs(corr.iloc[i, j]))
        
        try:
            ev = nx.eigenvector_centrality_numpy(G_full, weight='weight')
        except:
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
                    snap = self.compute_snapshot(subset, cols, w)
                    record[f'sync_{w}d'] = snap['network_sync']
                    if w == self.windows[-1]:
                        record['top_hub_bt'] = snap['top_hub_bt']
                        record['top_hub_ev'] = snap['top_hub_ev']
                        record['hub_influence'] = snap['hub_influence']
                
                records.append(record)
            except:
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


# =============================================================================
# KEY PAIRS ANALYZER
# =============================================================================
class KeyPairsAnalyzer:
    def __init__(self, config: dict):
        self.key_pairs = config.get('key_pairs', [])
    
    def compute_current(self, returns: pd.DataFrame, window: int = 60) -> dict:
        recent = returns.iloc[-window:] if len(returns) >= window else returns
        corr = recent.corr()
        
        result = {}
        for pair in self.key_pairs:
            a, b = pair[0], pair[1]
            if a in corr.columns and b in corr.columns:
                result[f'{a}/{b}'] = corr.loc[a, b]
        return result
    
    def compute_change(self, returns: pd.DataFrame, window: int = 60, lookback: int = 20) -> dict:
        if len(returns) < window + lookback:
            return {}
        
        recent = returns.iloc[-window:]
        past = returns.iloc[-(window + lookback):-lookback]
        
        corr_now = recent.corr()
        corr_past = past.corr()
        
        result = {}
        for pair in self.key_pairs:
            a, b = pair[0], pair[1]
            if a in corr_now.columns and b in corr_now.columns:
                now = corr_now.loc[a, b]
                past_val = corr_past.loc[a, b]
                result[f'{a}/{b}'] = {'now': now, 'past': past_val, 'change': now - past_val}
        return result
    
    def compute_timeseries(self, returns: pd.DataFrame, step: int = 5) -> pd.DataFrame:
        window = 60
        records = []
        
        for i in range(window, len(returns), step):
            date = returns.index[i]
            subset = returns.iloc[i-window:i]
            corr = subset.corr()
            
            record = {'date': date}
            for pair in self.key_pairs:
                a, b = pair[0], pair[1]
                if a in corr.columns and b in corr.columns:
                    record[f'{a}/{b}'] = corr.loc[a, b]
            records.append(record)
        
        return pd.DataFrame(records).set_index('date')


# =============================================================================
# VOLATILITY ANALYZER
# =============================================================================
class VolatilityAnalyzer:
    def __init__(self, config: dict):
        self.analysis = config.get('analysis', {})
    
    def compute_rv_percentile(self, series: pd.Series) -> pd.Series:
        rv_window = self.analysis.get('rv_window', 5)
        lookback = self.analysis.get('lookback', 252)
        
        rv = series.rolling(rv_window).std() * np.sqrt(252) * 100
        return rv.rolling(lookback, min_periods=60).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
        )
    
    def compute_all(self, returns: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            result[col] = self.compute_rv_percentile(returns[col])
        return result


# =============================================================================
# TAIL DEPENDENCE CALCULATOR
# =============================================================================
class TailDependenceCalculator:
    """
    Copula-free empirical tail dependence estimation
    
    Key insight: Raw tail dependence is misleading.
    What matters is EXCESS tail dependence = Tail - Expected(given correlation)
    
    If two assets have corr=0.5, we expect tail dependence ~27%.
    If actual tail is 45%, the EXCESS of +18% indicates crisis contagion.
    """
    def __init__(self, config: dict):
        self.config = config
        self.thresholds = [0.05, 0.10]  # 5% and 10% tails
    
    def _compute_tail_dependence(self, x: np.ndarray, y: np.ndarray, q: float = 0.10) -> dict:
        """Compute empirical tail dependence coefficients"""
        # Lower tail (ë™ë°˜ í­ë½)
        x_threshold_low = np.percentile(x, q * 100)
        y_threshold_low = np.percentile(y, q * 100)
        
        x_in_lower = x < x_threshold_low
        if x_in_lower.sum() > 0:
            lower_td = (y[x_in_lower] < y_threshold_low).mean()
        else:
            lower_td = 0.0
        
        # Upper tail (ë™ë°˜ ê¸‰ë“±)
        x_threshold_high = np.percentile(x, (1 - q) * 100)
        y_threshold_high = np.percentile(y, (1 - q) * 100)
        
        x_in_upper = x > x_threshold_high
        if x_in_upper.sum() > 0:
            upper_td = (y[x_in_upper] > y_threshold_high).mean()
        else:
            upper_td = 0.0
        
        return {
            'lower': lower_td,
            'upper': upper_td,
            'asymmetry': lower_td - upper_td
        }
    
    def _expected_tail_given_corr(self, corr: float, q: float = 0.10) -> float:
        """
        Rough approximation of expected tail dependence given correlation.
        For bivariate normal, tail dependence = 0 for |rho| < 1.
        But empirically, higher correlation â†’ higher tail dependence.
        Using linear approximation: expected = q + 0.35 * |corr|
        """
        return q + 0.35 * abs(corr)
    
    def compute_for_pairs(self, returns: pd.DataFrame, pairs: list = None) -> dict:
        """Compute tail dependence with excess over correlation"""
        if pairs is None:
            pairs = self.config.get('key_pairs', [])
        
        # Compute correlation matrix
        corr_matrix = returns.corr()
        
        results = {}
        for pair in pairs:
            a, b = pair[0], pair[1]
            if a in returns.columns and b in returns.columns:
                x = returns[a].dropna().values
                y = returns[b].dropna().values
                
                # Align
                min_len = min(len(x), len(y))
                x, y = x[-min_len:], y[-min_len:]
                
                # Get correlation
                corr = corr_matrix.loc[a, b] if a in corr_matrix.index and b in corr_matrix.columns else 0
                
                # Compute tail dependence at 10%
                td = self._compute_tail_dependence(x, y, q=0.10)
                
                # Expected given correlation
                expected = self._expected_tail_given_corr(corr, q=0.10)
                
                # Excess over correlation (the meaningful metric!)
                excess_lower = td['lower'] - expected
                excess_upper = td['upper'] - expected
                
                results[f'{a}/{b}'] = {
                    'lower_10': td['lower'],
                    'upper_10': td['upper'],
                    'correlation': corr,
                    'expected': expected,
                    'excess_lower': excess_lower,  # KEY METRIC
                    'excess_upper': excess_upper,
                    'asymmetry': td['asymmetry'],
                    'co_crash_prob': td['lower'],
                    'interpretation': self._interpret(td['lower'], excess_lower)
                }
        
        return results
    
    def _interpret(self, raw_tail: float, excess: float) -> str:
        """
        Interpretation based on EXCESS over correlation, not raw tail.
        excess > 0.15: Significant crisis contagion
        excess < -0.10: Tail diversification (good)
        """
        if excess > 0.20:
            return "âš ï¸ CRISIS CONTAGION"
        elif excess > 0.15:
            return "âš ï¸ Elevated contagion"
        elif excess > 0.10:
            return "Mild excess"
        elif excess < -0.10:
            return "Tail diversified"
        else:
            return "Normal"


# =============================================================================
# IMPULSE RESPONSE ANALYZER
# =============================================================================
class ImpulseResponseAnalyzer:
    """
    Analyze how hub asset movements propagate to connected assets.
    - Lead/Lag cross-correlation
    - Conditional response (when hub moves > 1 std)
    """
    def __init__(self, config: dict):
        self.config = config
        analysis = config.get('analysis', {})
        self.window = analysis.get('impulse_window', 60)
        self.max_lag = analysis.get('impulse_max_lag', 5)
    
    def compute_lead_lag(self, returns: pd.DataFrame, hub: str, neighbors: list) -> dict:
        """Compute lead-lag cross-correlations"""
        recent = returns.iloc[-self.window:]
        results = {}
        
        for neighbor in neighbors:
            if neighbor not in recent.columns or hub not in recent.columns:
                continue
            
            lag_corrs = {}
            for lag in range(-self.max_lag, self.max_lag + 1):
                if lag < 0:
                    x = recent[hub].iloc[:lag].values
                    y = recent[neighbor].iloc[-lag:].values
                elif lag > 0:
                    x = recent[hub].iloc[lag:].values
                    y = recent[neighbor].iloc[:-lag].values
                else:
                    x = recent[hub].values
                    y = recent[neighbor].values
                
                if len(x) > 10:
                    corr = np.corrcoef(x, y)[0, 1]
                    lag_corrs[lag] = corr
            
            if lag_corrs:
                best_lag = max(lag_corrs.keys(), key=lambda k: abs(lag_corrs[k]))
                results[neighbor] = {
                    'lag_correlations': lag_corrs,
                    'best_lag': best_lag,
                    'best_corr': lag_corrs[best_lag],
                    'contemporaneous': lag_corrs.get(0, 0),
                    'hub_leads': best_lag < 0,
                }
        
        return results
    
    def compute_conditional_response(self, returns: pd.DataFrame, hub: str, 
                                     neighbors: list, std_threshold: float = 1.0) -> dict:
        """When hub moves > threshold std, what happens to neighbors?"""
        recent = returns.iloc[-self.window:]
        
        if hub not in recent.columns:
            return {}
        
        hub_std = recent[hub].std()
        big_move_mask = recent[hub].abs() > (hub_std * std_threshold)
        big_move_dates = recent[big_move_mask].index
        
        results = {'hub': hub, 'n_events': len(big_move_dates), 'std_threshold': std_threshold}
        
        for neighbor in neighbors:
            if neighbor not in recent.columns:
                continue
            
            up_responses = []
            down_responses = []
            
            for event_date in big_move_dates:
                idx = recent.index.get_loc(event_date)
                hub_ret = recent[hub].iloc[idx]
                
                response = {
                    'date': event_date,
                    'hub_ret': hub_ret,
                    'same_day': recent[neighbor].iloc[idx],
                }
                
                if idx >= 5:
                    response['past_5d'] = recent[neighbor].iloc[idx-5:idx].mean()
                else:
                    response['past_5d'] = np.nan
                
                if idx + 5 < len(recent):
                    response['future_5d'] = recent[neighbor].iloc[idx+1:idx+6].mean()
                else:
                    response['future_5d'] = np.nan
                
                if hub_ret > 0:
                    up_responses.append(response)
                else:
                    down_responses.append(response)
            
            neighbor_result = {}
            
            if up_responses:
                up_df = pd.DataFrame(up_responses)
                neighbor_result['up'] = {
                    'n': len(up_df),
                    'past_5d_avg': up_df['past_5d'].mean(),
                    'same_day_avg': up_df['same_day'].mean(),
                    'future_5d_avg': up_df['future_5d'].mean(),
                }
            
            if down_responses:
                down_df = pd.DataFrame(down_responses)
                neighbor_result['down'] = {
                    'n': len(down_df),
                    'past_5d_avg': down_df['past_5d'].mean(),
                    'same_day_avg': down_df['same_day'].mean(),
                    'future_5d_avg': down_df['future_5d'].mean(),
                }
            
            results[neighbor] = neighbor_result
        
        return results


# =============================================================================
# TIMELINE TRACKER (Weekly snapshots)
# =============================================================================
class TimelineTracker:
    """Track weekly changes in Hub centrality and TE flows."""
    def __init__(self, config: dict):
        self.config = config
        analysis = config.get('analysis', {})
        self.hub_window = analysis.get('hub_window', 60)
        self.te_window = analysis.get('te_window_long', 252)
        self.weekly_step = 5
    
    def compute_hub_timeline(self, returns: pd.DataFrame, network_assets: list, 
                            n_weeks: int = 8) -> pd.DataFrame:
        """Track top hub changes over past n weeks"""
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
            dist = np.sqrt(2 * (1 - corr.values))
            np.fill_diagonal(dist, 0)
            
            G = nx.Graph()
            for i, a1 in enumerate(corr.columns):
                for j, a2 in enumerate(corr.columns):
                    if i < j:
                        G.add_edge(a1, a2, weight=dist[i, j])
            
            mst = nx.minimum_spanning_tree(G)
            bt = nx.betweenness_centrality(mst)
            
            top_hub = max(bt.keys(), key=lambda k: bt[k])
            top_bt = bt[top_hub]
            
            # Top 3 hubs
            sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)[:3]
            
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
    
    def compute_te_timeline(self, returns: pd.DataFrame, te_calc, 
                           n_weeks: int = 4, top_n: int = 5) -> list:
        """Track top TE flows over past n weeks"""
        records = []
        
        for week in range(n_weeks):
            end_idx = len(returns) - week * self.weekly_step
            start_idx = end_idx - self.te_window
            
            if start_idx < 0:
                break
            
            subset = returns.iloc[start_idx:end_idx]
            
            te_all = te_calc.compute_all_pairs(subset)
            net_flow = te_calc.compute_net_flow_all(subset)
            
            top_flows = sorted(net_flow.items(), 
                             key=lambda x: x[1]['flow_strength'], reverse=True)[:top_n]
            
            sig_count = sum(1 for v in te_all.values() if v['significant'])
            
            records.append({
                'week_ago': week,
                'end_date': returns.index[end_idx - 1],
                'top_flows': top_flows,
                'sig_count': sig_count,
                'total_pairs': len(te_all),
            })
        
        return records


# =============================================================================
# REPORT GENERATOR
# =============================================================================
class ReportGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        # Get window settings
        analysis = config.get('analysis', {})
        self.te_short = analysis.get('te_window_short', 60)
        self.te_long = analysis.get('te_window_long', 252)
        self.tail_window = analysis.get('tail_window', 252)
        self.corr_window = analysis.get('corr_window', 60)
    
    def generate(self, snapshot: dict, indicators: pd.DataFrame, rv_pct: pd.DataFrame,
                te_results: dict, te_sig_short: dict, te_net_flow_short: dict,
                te_sig_long: dict, te_net_flow_long: dict,
                key_pairs: dict, pair_changes: dict, 
                tail_short: dict, tail_long: dict,
                core_assets: list, cluster_reps: list, date: datetime,
                hub_timeline: pd.DataFrame = None, te_timeline: list = None,
                lead_lag: dict = None, cond_response: dict = None) -> str:
        
        current = indicators.iloc[-1] if len(indicators) > 0 else {}
        rv_current = rv_pct.iloc[-1] if len(rv_pct) > 0 else {}
        
        windows = self.config.get('analysis', {}).get('windows', [5, 20, 60])
        
        report = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘                   NETWORK MONITOR v2.3 (DUAL WINDOW) REPORT                      â•‘
â•‘                         {date.strftime('%Y-%m-%d %H:%M')}                                       â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
NETWORK STRUCTURE
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  Core Assets ({len(core_assets)}):    {', '.join(core_assets[:10])}
  Cluster Reps ({len(cluster_reps)}):  {', '.join(cluster_reps)}
  Total Network Nodes: {len(core_assets) + len(cluster_reps)}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
MULTI-SCALE SYNCHRONIZATION
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

"""
        for w in windows:
            sync_key = f'sync_{w}d'
            pct_key = f'{sync_key}_pct'
            sync_val = current.get(sync_key, 0)
            pct_val = current.get(pct_key, 50)
            status = "âš ï¸ HIGH" if sync_val > self.thresholds.get('sync_warning', 0.20) else ""
            report += f"  {w:3d}d Sync:  {sync_val:+.4f}  ({pct_val:.0f}%ile) {status}\n"
        
        if len(windows) >= 2:
            short_sync = current.get(f'sync_{windows[0]}d', 0)
            long_sync = current.get(f'sync_{windows[-1]}d', 0)
            divergence = short_sync - long_sync
            report += f"\n  Divergence ({windows[0]}d - {windows[-1]}d): {divergence:+.4f}"
            if divergence > 0.08:
                report += " âš ï¸ REGIME SHIFT"
            report += "\n"
        
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
TOP HUBS
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  BETWEENNESS (topology)          EIGENVECTOR (importance)
"""
        for i, ((bt_node, bt_val), (ev_node, ev_val)) in enumerate(
            zip(snapshot['top3_bt'], snapshot['top3_ev']), 1):
            bt_type = "CORE" if bt_node in core_assets else "CLUS"
            ev_type = "CORE" if ev_node in core_assets else "CLUS"
            report += f"  #{i}  {bt_node:<12} [{bt_type}] {bt_val:.4f}   #{i}  {ev_node:<12} [{ev_type}] {ev_val:.4f}\n"
        
        # DUAL WINDOW: Transfer Entropy (ALL PAIRS, show top N)
        n_pairs_short = len(te_net_flow_short)
        n_pairs_long = len(te_net_flow_long)
        
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
TRANSFER ENTROPY - ALL PAIRS (Top 10 by strength)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  [{self.te_long}d STRUCTURAL] ({n_pairs_long} unique pairs analyzed)
  Rank  Pair              Direction             Net Z    Sig
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
        top_flows_long = sorted(te_net_flow_long.items(), key=lambda x: x[1]['flow_strength'], reverse=True)[:10]
        for rank, (pair, data) in enumerate(top_flows_long, 1):
            sig_str = "âœ“âœ“" if data['both_significant'] else ("âœ“" if data['any_significant'] else "")
            report += f"  {rank:>2}.  {pair:<16} {data['dominant_direction']:<20} {data['net_flow_z']:+5.2f}   {sig_str}\n"
        
        sig_long = sum(1 for v in te_sig_long.values() if v['significant'])
        report += f"\n  Total significant directional pairs: {sig_long}/{len(te_sig_long)}\n"
        
        # Highlight significant TE from long window
        sig_pairs_long = [(k, v) for k, v in te_sig_long.items() if v['significant']]
        if sig_pairs_long:
            report += f"\n  âš ï¸ SIGNIFICANT CAUSALITY [{self.te_long}d]:\n"
            for pair, data in sorted(sig_pairs_long, key=lambda x: x[1]['te_z'], reverse=True)[:10]:
                report += f"    {pair}: Z={data['te_z']:+.2f}, p={data['p_value']:.3f}\n"
        
        # Short vs Long comparison - Z-score DELTA for regime change
        report += f"""
  [REGIME CHANGE] Z-score Delta ({self.te_short}d - {self.te_long}d)
  Positive = strengthening recently, Negative = weakening
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
        # Compute Z-score delta for same pairs
        z_deltas = []
        for pair, data_short in te_net_flow_short.items():
            if pair in te_net_flow_long:
                data_long = te_net_flow_long[pair]
                z_short = data_short['net_flow_z']
                z_long = data_long['net_flow_z']
                delta = z_short - z_long
                z_deltas.append({
                    'pair': pair,
                    'direction_short': data_short['dominant_direction'],
                    'direction_long': data_long['dominant_direction'],
                    'z_short': z_short,
                    'z_long': z_long,
                    'delta': delta,
                    'abs_delta': abs(delta),
                    'direction_change': data_short['dominant_direction'] != data_long['dominant_direction']
                })
        
        # Sort by absolute delta (biggest changes)
        z_deltas.sort(key=lambda x: x['abs_delta'], reverse=True)
        
        for item in z_deltas[:7]:
            dir_flag = "âš ï¸ FLIP" if item['direction_change'] else ""
            sign = "â†‘" if item['delta'] > 0 else "â†“"
            report += f"  {item['pair']:<18} Î”={item['delta']:+5.2f} {sign}  ({self.te_short}d:{item['z_short']:+.1f}, {self.te_long}d:{item['z_long']:+.1f}) {dir_flag}\n"
        
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
KEY CORRELATION PAIRS
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

"""
        for pair_name, val in key_pairs.items():
            change_data = pair_changes.get(pair_name, {})
            change = change_data.get('change', 0)
            report += f"  {pair_name:<20} {val:+.3f}  (Î”20d: {change:+.3f})\n"
        
        # Tail Dependence (EXCESS-based, long window only)
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
TAIL DEPENDENCE - EXCESS ANALYSIS ({self.tail_window}d, 10% tails)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  Excess = Tail Dependence - Expected(given correlation)
  Positive excess = Crisis contagion (tail coupling stronger than normal)

  Pair                Corr    Tail    Expected  Excess   Status
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
        for pair, data in sorted(tail_long.items(), key=lambda x: x[1].get('excess_lower', 0), reverse=True):
            corr = data.get('correlation', 0)
            tail = data.get('lower_10', 0)
            expected = data.get('expected', 0)
            excess = data.get('excess_lower', 0)
            status = data.get('interpretation', '')
            report += f"  {pair:<18} {corr:+.2f}   {tail:5.1%}   {expected:5.1%}   {excess:+5.1%}   {status}\n"
        
        # Highlight significant excess (crisis contagion)
        high_excess = [(k, v) for k, v in tail_long.items() if v.get('excess_lower', 0) > 0.15]
        if high_excess:
            report += f"\n  âš ï¸ CRISIS CONTAGION DETECTED [{self.tail_window}d]:\n"
            for pair, data in sorted(high_excess, key=lambda x: x[1].get('excess_lower', 0), reverse=True):
                report += f"    {pair}: +{data['excess_lower']:.0%} excess (tail {data['lower_10']:.0%} vs expected {data['expected']:.0%})\n"
        
        # Alerts (based on long window analysis)
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
ALERTS (based on {self.te_long}d structural analysis)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        alerts = []
        
        if len(windows) >= 2:
            short_sync = current.get(f'sync_{windows[0]}d', 0)
            long_sync = current.get(f'sync_{windows[-1]}d', 0)
            if short_sync - long_sync > 0.08:
                alerts.append(f"âš ï¸ REGIME SHIFT: Short-term sync diverging")
        
        # Alert on SIGNIFICANT TE from long window (more reliable)
        for pair, data in te_sig_long.items():
            if data['significant'] and data['te_z'] > 2.0:
                alerts.append(f"âš ï¸ STRONG TE [{self.te_long}d]: {pair} (Z={data['te_z']:.2f}, p={data['p_value']:.3f})")
        
        # Net flow alerts from long window
        for pair, data in te_net_flow_long.items():
            if data['any_significant'] and abs(data['net_flow_z']) > 1.5:
                alerts.append(f"âš ï¸ DIRECTIONAL FLOW [{self.te_long}d]: {data['dominant_direction']} (Z={data['net_flow_z']:+.2f})")
        
        # Tail dependence alerts based on EXCESS (not raw tail)
        for pair, data in tail_long.items():
            excess = data.get('excess_lower', 0)
            if excess > 0.20:
                alerts.append(f"âš ï¸ CRISIS CONTAGION [{self.tail_window}d]: {pair} excess={excess:+.0%}")
            elif excess > 0.15:
                alerts.append(f"â–³ Elevated contagion [{self.tail_window}d]: {pair} excess={excess:+.0%}")
        
        if snapshot['top_hub_bt'] != snapshot['top_hub_ev']:
            alerts.append(f"âš ï¸ HUB DIVERGENCE: BT={snapshot['top_hub_bt']}, EV={snapshot['top_hub_ev']}")
        
        extreme = [a for a, v in rv_current.items() if v > self.thresholds.get('rv_extreme', 90)]
        if extreme:
            alerts.append(f"âš ï¸ EXTREME VOL: {', '.join(extreme[:5])}")
        
        if alerts:
            for a in alerts[:10]:
                report += f"{a}\n"
        else:
            report += "âœ“ No critical alerts\n"
        
        # Volatility
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
VOLATILITY (Top 10)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        rv_sorted = rv_current.sort_values(ascending=False)
        for asset, pct in rv_sorted.head(10).items():
            status = "âš ï¸ EXTREME" if pct > 90 else ("â–³ ELEVATED" if pct > 75 else "")
            report += f"  {asset:<12}: {pct:5.1f}%  {status}\n"
        
        # Hub Timeline (weekly changes)
        if hub_timeline is not None and len(hub_timeline) > 0:
            report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
HUB TIMELINE (Weekly, 60d rolling)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  Week      Date        Top Hub (BT)    BT Value   Sync
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
            for _, row in hub_timeline.iterrows():
                week_label = "Current" if row['week_ago'] == 0 else f"-{row['week_ago']}w"
                date_str = row['end_date'].strftime('%m/%d')
                report += f"  {week_label:<8}  {date_str:<10}  {row['top_hub']:<14}  {row['top_bt']:.4f}    {row['network_sync']:+.3f}\n"
            
            # Detect hub changes
            hub_changes = []
            for i in range(len(hub_timeline) - 1):
                curr = hub_timeline.iloc[i]['top_hub']
                prev = hub_timeline.iloc[i + 1]['top_hub']
                if curr != prev:
                    hub_changes.append(f"-{hub_timeline.iloc[i]['week_ago']}w: {prev} â†’ {curr}")
            
            if hub_changes:
                report += f"\n  âš ï¸ Hub Changes: {', '.join(hub_changes)}\n"
            else:
                report += f"\n  âœ“ Hub stable over {len(hub_timeline)} weeks\n"
        
        # Impulse Response
        if lead_lag is not None and cond_response is not None and len(lead_lag) > 0:
            hub = cond_response.get('hub', 'N/A')
            n_events = cond_response.get('n_events', 0)
            
            report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
IMPULSE RESPONSE: {hub} â†’ Neighbors (60d)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

  [LEAD-LAG STRUCTURE]
  Neighbor        Contemp.   Best Lag   Best Corr   Direction
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
            for neighbor, data in lead_lag.items():
                contemp = data['contemporaneous']
                best_lag = data['best_lag']
                best_corr = data['best_corr']
                direction = "Hub leads" if data['hub_leads'] else "Hub lags"
                if best_lag == 0:
                    direction = "Contemp."
                report += f"  {neighbor:<14} {contemp:+.3f}     {best_lag:+2d}d       {best_corr:+.3f}      {direction}\n"
            
            report += f"""
  [CONDITIONAL RESPONSE] When {hub} moves > 1Ïƒ (n={n_events} events)
  Neighbor         â”‚ Hub UP                        â”‚ Hub DOWN
                   â”‚ Past5d   Same    Future5d     â”‚ Past5d   Same    Future5d
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""
            for neighbor, data in cond_response.items():
                if neighbor in ['hub', 'n_events', 'std_threshold']:
                    continue
                
                up = data.get('up', {})
                down = data.get('down', {})
                
                up_past = up.get('past_5d_avg', 0) * 100 if up else 0
                up_same = up.get('same_day_avg', 0) * 100 if up else 0
                up_future = up.get('future_5d_avg', 0) * 100 if up else 0
                
                down_past = down.get('past_5d_avg', 0) * 100 if down else 0
                down_same = down.get('same_day_avg', 0) * 100 if down else 0
                down_future = down.get('future_5d_avg', 0) * 100 if down else 0
                
                report += f"  {neighbor:<16} â”‚ {up_past:+5.2f}%  {up_same:+5.2f}%  {up_future:+5.2f}%    â”‚ {down_past:+5.2f}%  {down_same:+5.2f}%  {down_future:+5.2f}%\n"
        
        report += "\n" + "â•" * 82 + "\n"
        
        return report


# =============================================================================
# VISUALIZER (Clustered)
# =============================================================================
class ClusteredVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.viz = config.get('visualization', {})
        self.colors = self.viz.get('colors', {
            'bg': '#0d1117', 'panel': '#161b22', 'text': '#e6edf3',
            'grid': '#30363d', 'danger': '#f85149', 'warning': '#d29922',
            'safe': '#3fb950', 'accent': '#58a6ff',
        })
    
    def get_node_color(self, asset: str) -> str:
        # Core categories
        for cat, assets in self.config.get('core_categories', {}).items():
            if asset in assets:
                return self.config.get('core_colors', {}).get(cat, '#8b949e')
        # Clusters
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            if asset in cluster_data.get('assets', {}) or asset == cluster_data.get('representative'):
                return cluster_data.get('color', '#8b949e')
        return '#8b949e'
    
    def create_dashboard(self, snapshot: dict, indicators: pd.DataFrame,
                        rv_pct: pd.DataFrame, pair_ts: pd.DataFrame,
                        te_results: dict, core_assets: list, cluster_reps: list,
                        output_path: str):
        
        fig = plt.figure(figsize=self.viz.get('figsize', [28, 24]), facecolor=self.colors['bg'])
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1.4, 1, 1, 0.8], hspace=0.3, wspace=0.25)
        
        # Row 1: Network + Metrics
        ax1 = fig.add_subplot(gs[0, :2])
        self._draw_network(ax1, snapshot, core_assets, cluster_reps)
        
        ax2 = fig.add_subplot(gs[0, 2])
        self._draw_metrics(ax2, snapshot, indicators, te_results, core_assets, cluster_reps)
        
        # Row 2: Multi-scale sync + TE bars
        ax3 = fig.add_subplot(gs[1, :2])
        self._draw_multi_scale_sync(ax3, indicators)
        
        ax4 = fig.add_subplot(gs[1, 2])
        self._draw_te_bars(ax4, te_results)
        
        # Row 3: Key pairs + Correlation matrix
        ax5 = fig.add_subplot(gs[2, :2])
        self._draw_key_pairs(ax5, pair_ts)
        
        ax6 = fig.add_subplot(gs[2, 2])
        self._draw_corr_matrix(ax6, snapshot)
        
        # Row 4: RV Heatmap
        ax7 = fig.add_subplot(gs[3, :])
        self._draw_rv_heatmap(ax7, rv_pct, core_assets + cluster_reps)
        
        # Title
        date = indicators.index[-1] if len(indicators) > 0 else datetime.now()
        fig.suptitle(f'NETWORK MONITOR v2.3 (DUAL WINDOW) - {date.strftime("%Y-%m-%d")}',
                    fontsize=20, fontweight='bold', color='white', y=0.995)
        
        plt.savefig(output_path, dpi=self.viz.get('dpi', 150),
                   facecolor=self.colors['bg'], bbox_inches='tight')
        plt.close()
    
    def _draw_network(self, ax, snapshot, core_assets, cluster_reps):
        ax.set_facecolor(self.colors['panel'])
        
        mst = snapshot['mst']
        bt = snapshot['betweenness']
        
        # Node sizes: Core = big, Cluster = small
        core_size = self.viz.get('core_node_size', 800)
        cluster_size = self.viz.get('cluster_node_size', 300)
        
        node_colors = [self.get_node_color(n) for n in mst.nodes()]
        node_sizes = [core_size + bt.get(n, 0) * 3000 if n in core_assets 
                     else cluster_size + bt.get(n, 0) * 1500 for n in mst.nodes()]
        
        pos = nx.kamada_kawai_layout(mst)
        
        edge_colors = []
        edge_widths = []
        for u, v in mst.edges():
            c = mst[u][v]['corr']
            edge_widths.append(1 + abs(c) * 4)
            edge_colors.append(self.colors['safe'] if c > 0.6 else
                              (self.colors['warning'] if c > 0.3 else self.colors['grid']))
        
        nx.draw_networkx_edges(mst, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(mst, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                              alpha=0.9, edgecolors='white', linewidths=2)
        
        # Labels: Core = bold, Cluster = normal
        labels_core = {n: n for n in mst.nodes() if n in core_assets}
        labels_cluster = {n: n for n in mst.nodes() if n in cluster_reps}
        
        nx.draw_networkx_labels(mst, pos, labels=labels_core, ax=ax, 
                               font_size=10, font_color='white', font_weight='bold')
        nx.draw_networkx_labels(mst, pos, labels=labels_cluster, ax=ax, 
                               font_size=8, font_color='#c9d1d9', font_weight='normal')
        
        # Mark top hubs
        for rank, (node, _) in enumerate(snapshot['top3_bt'], 1):
            x, y = pos[node]
            ax.annotate(f'B{rank}', xy=(x, y), xytext=(x+0.08, y+0.08),
                       fontsize=12, color=self.colors['danger'], fontweight='bold')
        
        ev_top = snapshot['top_hub_ev']
        if ev_top and ev_top != snapshot['top_hub_bt']:
            x, y = pos.get(ev_top, (0, 0))
            ax.annotate(f'E1', xy=(x, y), xytext=(x-0.12, y+0.08),
                       fontsize=12, color=self.colors['accent'], fontweight='bold')
        
        ax.set_title(f"CLUSTERED NETWORK (Core={len(core_assets)}, Cluster={len(cluster_reps)})",
                    fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.axis('off')
        
        # Legend
        legend_items = []
        for cat, color in self.config.get('core_colors', {}).items():
            legend_items.append(mpatches.Patch(facecolor=color, label=cat))
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            legend_items.append(mpatches.Patch(facecolor=cluster_data.get('color', '#8b949e'), 
                                              label=cluster_name, alpha=0.5))
        
        ax.legend(handles=legend_items[:12], loc='lower left', fontsize=7, ncol=2,
                 facecolor=self.colors['panel'], labelcolor=self.colors['text'])
    
    def _draw_metrics(self, ax, snapshot, indicators, te_results, core_assets, cluster_reps):
        ax.set_facecolor(self.colors['panel'])
        ax.axis('off')
        
        current = indicators.iloc[-1] if len(indicators) > 0 else {}
        windows = self.config.get('analysis', {}).get('windows', [5, 20, 60])
        
        text = f"""NETWORK STRUCTURE
{'â”€' * 35}
Core Assets:    {len(core_assets)}
Cluster Reps:   {len(cluster_reps)}
Total Nodes:    {len(core_assets) + len(cluster_reps)}

TOP 3 HUBS
{'â”€' * 35}
BETWEENNESS        EIGENVECTOR
"""
        for i, ((bt_n, bt_v), (ev_n, ev_v)) in enumerate(
            zip(snapshot['top3_bt'], snapshot['top3_ev']), 1):
            text += f"#{i} {bt_n:<10} {bt_v:.3f}  {ev_n:<10} {ev_v:.3f}\n"
        
        text += f"""
MULTI-SCALE SYNC
{'â”€' * 35}
"""
        for w in windows:
            sync_val = current.get(f'sync_{w}d', 0)
            text += f"{w:3d}d: {sync_val:+.4f}\n"
        
        text += f"""
TOP TE FLOWS
{'â”€' * 35}
"""
        for pair, val in sorted(te_results.items(), key=lambda x: x[1], reverse=True)[:5]:
            text += f"{pair}: {val:.4f}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
               color=self.colors['text'], va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.5))
    
    def _draw_multi_scale_sync(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        
        windows = self.config.get('analysis', {}).get('windows', [5, 20, 60])
        colors_list = [self.colors['danger'], self.colors['warning'], self.colors['safe']]
        
        for i, w in enumerate(windows):
            col = f'sync_{w}d'
            if col in indicators.columns:
                ax.plot(indicators.index, indicators[col], 
                       color=colors_list[i % len(colors_list)], lw=1.5, label=f'{w}d')
        
        ax.axhline(0.20, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)
        ax.legend(loc='upper left', fontsize=8, facecolor=self.colors['panel'],
                 labelcolor=self.colors['text'])
        
        ax.set_ylabel('Network Sync', color=self.colors['text'])
        ax.set_title('MULTI-SCALE SYNCHRONIZATION', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_te_bars(self, ax, te_results):
        ax.set_facecolor(self.colors['panel'])
        
        # Top 10 TE pairs
        sorted_te = sorted(te_results.items(), key=lambda x: x[1], reverse=True)[:10]
        pairs = [p[0] for p in sorted_te]
        vals = [p[1] for p in sorted_te]
        
        colors = [self.colors['danger'] if v > 0.15 else self.colors['accent'] for v in vals]
        
        y_pos = range(len(pairs))
        ax.barh(y_pos, vals, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=7, color=self.colors['text'])
        ax.axvline(0.15, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)
        
        ax.set_xlabel('Transfer Entropy', color=self.colors['text'])
        ax.set_title('INFORMATION FLOW', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_key_pairs(self, ax, pair_ts):
        ax.set_facecolor(self.colors['panel'])
        
        key_pairs_to_show = ['USDJPY/NKY', 'USDKRW/USDJPY', 'JGB_10Y/KTB_10Y', 
                            'UST_10Y/JGB_10Y', 'Bund_10Y/UST_10Y']
        colors_list = [self.colors['danger'], self.colors['accent'], self.colors['warning'],
                      self.colors['safe'], '#8b5cf6']
        
        for i, pair in enumerate(key_pairs_to_show):
            if pair in pair_ts.columns:
                ax.plot(pair_ts.index, pair_ts[pair], 
                       color=colors_list[i % len(colors_list)], lw=1.5, label=pair)
        
        ax.axhline(0, color=self.colors['grid'], ls='-', lw=0.5)
        ax.legend(loc='upper left', fontsize=7, facecolor=self.colors['panel'],
                 labelcolor=self.colors['text'], ncol=2)
        
        ax.set_ylabel('Correlation', color=self.colors['text'])
        ax.set_title('KEY PAIR CORRELATIONS', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_corr_matrix(self, ax, snapshot):
        ax.set_facecolor(self.colors['panel'])
        
        corr = snapshot['corr']
        
        cmap = LinearSegmentedColormap.from_list('corr', 
            ['#f85149', '#161b22', '#3fb950'], N=100)
        
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=6, color=self.colors['text'])
        ax.set_yticklabels(corr.columns, fontsize=6, color=self.colors['text'])
        
        ax.set_title('CORRELATION MATRIX', fontsize=11, fontweight='bold', color=self.colors['text'])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.colors['text'])
    
    def _draw_rv_heatmap(self, ax, rv_pct, network_assets):
        ax.set_facecolor(self.colors['panel'])
        
        days = self.viz.get('heatmap_days', 90)
        
        # Filter to network assets
        cols = [c for c in network_assets if c in rv_pct.columns]
        rv_recent = rv_pct[cols].iloc[-days:]
        
        cmap = LinearSegmentedColormap.from_list('stress',
            ['#238636', '#3fb950', '#d29922', '#f85149', '#da3633'], N=100)
        
        im = ax.imshow(rv_recent.T.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)
        
        ax.set_yticks(range(len(rv_recent.columns)))
        ax.set_yticklabels(rv_recent.columns, color=self.colors['text'], fontsize=8)
        
        n = len(rv_recent)
        ticks = [0, n//4, n//2, 3*n//4, n-1]
        labels = [rv_recent.index[i].strftime('%m/%d') for i in ticks if i < len(rv_recent)]
        ax.set_xticks(ticks[:len(labels)])
        ax.set_xticklabels(labels, color=self.colors['text'])
        
        ax.set_title(f'RV PERCENTILE ({days}d)', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.15)
        cbar.set_label('Percentile', color=self.colors['text'])
        cbar.ax.xaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=self.colors['text'])


# =============================================================================
# MAIN
# =============================================================================
def load_config(config_path: str) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config not found: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Market Network Monitor v2.1 (Clustered)')
    parser.add_argument('-d', '--data', required=True, help='Path to MARKET_WATCH.xlsx')
    parser.add_argument('-c', '--config', required=True, help='Path to clustered config.yaml')
    parser.add_argument('-o', '--output', default='./output_clustered', help='Output directory')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MARKET NETWORK MONITOR v2.3 (DUAL WINDOW)")
    print("=" * 70)
    
    # Load config
    print("\n[1/7] Loading configuration...")
    config = load_config(args.config)
    print(f"  Loaded from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[2/7] Loading market data...")
    loader = ClusteredDataLoader(args.data, config)
    loader.load()
    
    network_assets = loader.get_network_assets()
    
    # Compute network metrics
    print("\n[3/7] Computing network metrics...")
    network = ClusteredNetworkAnalyzer(config)
    indicators = network.compute_timeseries(loader.returns, network_assets)
    
    # Get window settings from config
    analysis_cfg = config.get('analysis', {})
    hub_window = analysis_cfg.get('hub_window', 60)
    te_window_short = analysis_cfg.get('te_window_short', 60)
    te_window_long = analysis_cfg.get('te_window_long', 252)
    tail_window = analysis_cfg.get('tail_window', 252)
    corr_window = analysis_cfg.get('corr_window', 60)
    corr_change_lookback = analysis_cfg.get('corr_change_lookback', 20)
    
    snapshot = network.compute_snapshot(loader.returns, network_assets, hub_window)
    print(f"  Top Hub (BT): {snapshot['top_hub_bt']}, (EV): {snapshot['top_hub_ev']}")
    
    # Compute Transfer Entropy (ALL PAIRS with surrogate testing)
    # Dual window analysis: short-term (regime) + long-term (structural)
    print(f"\n[4/7] Computing Transfer Entropy (all pairs, {te_window_short}d + {te_window_long}d)...")
    te_calc = TransferEntropyCalculator(config)
    
    # Short-term - for regime change detection
    recent_short = loader.returns.iloc[-te_window_short:]
    print(f"  [{te_window_short}d] Computing all pairs...")
    te_all_60 = te_calc.compute_all_pairs(recent_short)
    te_net_flow_60 = te_calc.compute_net_flow_all(recent_short)
    
    sig_count_60 = sum(1 for v in te_all_60.values() if v['significant'])
    n_pairs_60 = len(te_all_60)
    print(f"  [{te_window_short}d] Significant: {sig_count_60}/{n_pairs_60} directional pairs")
    
    # Long-term - for structural causality (more reliable)
    recent_long = loader.returns.iloc[-te_window_long:] if len(loader.returns) >= te_window_long else loader.returns
    print(f"  [{te_window_long}d] Computing all pairs...")
    te_all_252 = te_calc.compute_all_pairs(recent_long)
    te_net_flow_252 = te_calc.compute_net_flow_all(recent_long)
    
    sig_count_252 = sum(1 for v in te_all_252.values() if v['significant'])
    n_pairs_252 = len(te_all_252)
    print(f"  [{te_window_long}d] Significant: {sig_count_252}/{n_pairs_252} directional pairs")
    
    # Show top significant
    top_sig = te_calc.get_top_significant(te_all_252, n=3)
    if top_sig:
        print(f"  Top Significant [{te_window_long}d]:")
        for pair, data in top_sig:
            print(f"    {pair}: Z={data['te_z']:+.2f}, p={data['p_value']:.3f}")
    
    # For backward compatibility
    te_results = {k: v['te_raw'] for k, v in te_all_60.items()}
    te_sig_60 = te_all_60
    te_sig_252 = te_all_252
    
    # Compute Key Pairs
    print(f"\n[5/7] Computing key correlation pairs ({corr_window}d window)...")
    pairs_analyzer = KeyPairsAnalyzer(config)
    key_pairs = pairs_analyzer.compute_current(loader.returns, corr_window)
    pair_changes = pairs_analyzer.compute_change(loader.returns, corr_window, lookback=corr_change_lookback)
    pair_ts = pairs_analyzer.compute_timeseries(loader.returns, step=config['analysis']['step'])
    
    # Compute volatility
    print("\n[6/7] Computing volatility metrics...")
    vol = VolatilityAnalyzer(config)
    rv_pct = vol.compute_all(loader.returns)
    
    # Compute Tail Dependence (long window only for statistical validity)
    print(f"  Computing tail dependence ({tail_window}d)...")
    tail_calc = TailDependenceCalculator(config)
    
    # Short-term - for comparison (noisy)
    tail_results_60 = tail_calc.compute_for_pairs(recent_short, config.get('key_pairs', []))
    
    # Long-term - structural tail dependence (primary)
    recent_tail = loader.returns.iloc[-tail_window:] if len(loader.returns) >= tail_window else loader.returns
    tail_results_252 = tail_calc.compute_for_pairs(recent_tail, config.get('key_pairs', []))
    
    # Report high EXCESS (crisis contagion) from 252d
    high_excess = [(k, v) for k, v in tail_results_252.items() if v.get('excess_lower', 0) > 0.15]
    if high_excess:
        print(f"  âš ï¸ Crisis contagion [252d] ({len(high_excess)}):")
        for pair, data in sorted(high_excess, key=lambda x: x[1].get('excess_lower', 0), reverse=True)[:3]:
            print(f"    {pair}: excess={data['excess_lower']:+.0%} (tail {data['lower_10']:.0%})")
    else:
        print("  âœ“ No elevated contagion [252d]")
    print("  Done")
    
    # Compute Timeline (Hub/TE weekly changes)
    print("\n[7/8] Computing timeline analysis...")
    timeline = TimelineTracker(config)
    hub_timeline = timeline.compute_hub_timeline(loader.returns, network_assets, n_weeks=8)
    print(f"  Hub timeline: {len(hub_timeline)} weeks")
    
    # Show hub changes
    if len(hub_timeline) > 1:
        current_hub = hub_timeline.iloc[0]['top_hub']
        prev_hub = hub_timeline.iloc[1]['top_hub'] if len(hub_timeline) > 1 else current_hub
        if current_hub != prev_hub:
            print(f"  âš ï¸ Hub changed: {prev_hub} â†’ {current_hub}")
        else:
            print(f"  Hub stable: {current_hub}")
    
    # TE timeline (expensive, only 4 weeks)
    print("  Computing TE timeline (4 weeks)...")
    te_timeline = timeline.compute_te_timeline(loader.returns, te_calc, n_weeks=4, top_n=5)
    print(f"  TE timeline: {len(te_timeline)} weeks")
    
    # Compute Impulse Response
    print("\n[8/8] Computing impulse response...")
    impulse = ImpulseResponseAnalyzer(config)
    
    # Get top hub and its MST neighbors
    top_hub = snapshot['top_hub_bt']
    mst_neighbors = list(snapshot['mst'].neighbors(top_hub)) if top_hub in snapshot['mst'] else []
    
    lead_lag = impulse.compute_lead_lag(loader.returns, top_hub, mst_neighbors)
    cond_response = impulse.compute_conditional_response(loader.returns, top_hub, mst_neighbors)
    
    print(f"  Hub: {top_hub}, Neighbors: {len(mst_neighbors)}")
    print(f"  Big move events: {cond_response.get('n_events', 0)}")
    
    # Generate outputs
    print("\n[9/9] Generating outputs...")
    
    date = loader.returns.index[-1]
    date_str = date.strftime('%Y%m%d')
    
    output_cfg = config.get('output', {})
    
    # Report
    if output_cfg.get('save_report', True):
        report_gen = ReportGenerator(config)
        report = report_gen.generate(snapshot, indicators, rv_pct, 
                                    te_results, te_sig_60, te_net_flow_60,
                                    te_sig_252, te_net_flow_252,
                                    key_pairs, pair_changes, 
                                    tail_results_60, tail_results_252,
                                    loader.core_assets, loader.cluster_reps, date,
                                    hub_timeline=hub_timeline, te_timeline=te_timeline,
                                    lead_lag=lead_lag, cond_response=cond_response)
        
        report_path = output_dir / f"{output_cfg.get('report_prefix', 'report')}_{date_str}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Report: {report_path}")
        
        if not args.quiet:
            print("\n" + "=" * 70)
            print(report)
    
    # Dashboard
    if output_cfg.get('save_dashboard', True):
        viz = ClusteredVisualizer(config)
        dashboard_path = output_dir / f"{output_cfg.get('dashboard_prefix', 'dashboard')}_{date_str}.png"
        viz.create_dashboard(snapshot, indicators, rv_pct, pair_ts, te_results,
                            loader.core_assets, loader.cluster_reps, str(dashboard_path))
        print(f"  Dashboard: {dashboard_path}")
    
    # Indicators
    if output_cfg.get('save_indicators', True):
        indicators_path = output_dir / 'indicators_clustered.pkl'
        indicators.to_pickle(indicators_path)
        print(f"  Indicators: {indicators_path}")
    
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
