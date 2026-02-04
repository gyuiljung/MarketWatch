#!/usr/bin/env python3
"""
=============================================================================
MARKET NETWORK MONITOR v1.0
=============================================================================

ì‹œìž¥ ë„¤íŠ¸ì›Œí¬ í† í´ë¡œì§€ ê¸°ë°˜ ìŠ¤íŠ¸ë ˆìŠ¤ ëª¨ë‹ˆí„°ë§ ë„êµ¬

Usage:
    # ê¸°ë³¸ ì‹¤í–‰ (ì˜¤ëŠ˜ ê¸°ì¤€)
    python run_monitor.py -d /path/to/MARKET_WATCH.xlsx
    
    # ì¶œë ¥ ë””ë ‰í† ë¦¬ ì§€ì •
    python run_monitor.py -d MARKET_WATCH.xlsx -o ./reports/
    
    # ë¦¬í¬íŠ¸ë§Œ ìƒì„± (ì´ë¯¸ì§€ ì—†ì´)
    python run_monitor.py -d MARKET_WATCH.xlsx --report-only
    
    # íŠ¹ì • ë‚ ì§œ ê¸°ì¤€ ë¶„ì„
    python run_monitor.py -d MARKET_WATCH.xlsx --date 2024-08-05
    
    # ì„¤ì • íŒŒì¼ ì§€ì •
    python run_monitor.py -d MARKET_WATCH.xlsx -c custom_config.yaml

Output:
    1. network_monitor_YYYYMMDD.png  - ëŒ€ì‹œë³´ë“œ ì´ë¯¸ì§€
    2. network_report_YYYYMMDD.txt   - í…ìŠ¤íŠ¸ ë¦¬í¬íŠ¸
    3. indicators.pkl                - ì§€í‘œ ì‹œê³„ì—´ (ìž¬ì‚¬ìš©ìš©)

=============================================================================
"""

import argparse
import os
import sys
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


# =============================================================================
# DEFAULT CONFIG (config.yaml ì—†ì„ ë•Œ ì‚¬ìš©)
# =============================================================================
DEFAULT_CONFIG = {
    'assets': {
        'USDKRW': 'ì¢…í•© USDKRW ìŠ¤íŒŸ (~15:30)',
        'USDJPY': 'ì´ì¢…í†µí™” ì¢…í•© JPY',
        'DXY': 'ë‹¬ëŸ¬ì¸ë±ìŠ¤ DOLLARS',
        'SPX': 'S&P 500',
        'NKY': 'ë‹ˆì¼€ì´ 225',
        'VIX': 'CBOE VIX VOLATILITY INDEX',
        'GOLD': 'ê¸ˆ 2026-2 (ì—°ê²°ì„ ë¬¼)',
        'KTB_2Y': 'ê¸ˆíˆ¬í˜‘ ìž¥ì™¸ê±°ëž˜ëŒ€í‘œìˆ˜ìµë¥  êµ­ê³ ì±„ê¶Œ(2ë…„)',
        'KTB_10Y': 'ê¸ˆíˆ¬í˜‘ ìž¥ì™¸ê±°ëž˜ëŒ€í‘œìˆ˜ìµë¥  êµ­ê³ ì±„ê¶Œ(10ë…„)',
        'KTB_30Y': 'ê¸ˆíˆ¬í˜‘ ìž¥ì™¸ê±°ëž˜ëŒ€í‘œìˆ˜ìµë¥  êµ­ê³ ì±„ê¶Œ(30ë…„)',
        'JGB_2Y': 'ì¼ë³¸ 2ë…„',
        'JGB_10Y': 'ì¼ë³¸ 10ë…„',
        'JGB_30Y': 'ì¼ë³¸ 30ë…„',
        'UST_2Y': 'ë¯¸êµ­(ì¢…í•©) 2ë…„',
        'UST_10Y': 'ë¯¸êµ­(ì¢…í•©) 10ë…„',
        'UST_30Y': 'ë¯¸êµ­(ì¢…í•©) 30ë…„',
    },
    'rate_assets': ['KTB_2Y', 'KTB_10Y', 'KTB_30Y', 'JGB_2Y', 'JGB_10Y', 'JGB_30Y',
                    'UST_2Y', 'UST_10Y', 'UST_30Y'],
    'categories': {
        'EQ': ['SPX', 'NKY'],
        'FX': ['USDKRW', 'USDJPY', 'DXY'],
        'IR_JP': ['JGB_2Y', 'JGB_10Y', 'JGB_30Y'],
        'IR_KR': ['KTB_2Y', 'KTB_10Y', 'KTB_30Y'],
        'IR_US': ['UST_2Y', 'UST_10Y', 'UST_30Y'],
        'VOL': ['VIX'],
        'CMD': ['GOLD'],
    },
    'category_colors': {
        'EQ': '#3fb950', 'FX': '#f85149', 'IR_JP': '#f78166',
        'IR_KR': '#58a6ff', 'IR_US': '#a5d6ff', 'VOL': '#8b949e', 'CMD': '#d29922',
    },
    'analysis': {'window': 60, 'step': 5, 'rv_window': 5, 'lookback': 252},
    'thresholds': {'rv_extreme': 90, 'rv_elevated': 75},
    'visualization': {
        'colors': {
            'bg': '#0d1117', 'panel': '#161b22', 'text': '#e6edf3',
            'grid': '#30363d', 'danger': '#f85149', 'warning': '#d29922',
            'safe': '#3fb950', 'accent': '#58a6ff',
        },
        'figsize': [24, 20], 'dpi': 150, 'heatmap_days': 90,
    },
    'output': {
        'report_prefix': 'network_report', 'dashboard_prefix': 'network_monitor',
        'save_indicators': True, 'save_report': True, 'save_dashboard': True,
    },
}


# =============================================================================
# DATA LOADER
# =============================================================================
class DataLoader:
    """MARKET_WATCH.xlsx ë°ì´í„° ë¡œë”"""
    
    def __init__(self, filepath: str, config: dict):
        self.filepath = filepath
        self.config = config
        self.prices = None
        self.returns = None
        
    def load(self) -> 'DataLoader':
        """ë°ì´í„° ë¡œë“œ ë° ì „ì²˜ë¦¬"""
        print(f"  Loading {self.filepath}...")
        
        df = pd.read_excel(self.filepath, header=None)
        data = df.iloc[4:, :].copy()
        asset_names = df.iloc[2, :].tolist()
        data.columns = asset_names
        data = data.rename(columns={asset_names[0]: 'Date'})
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Select assets
        selected = pd.DataFrame()
        found_assets = []
        for short_name, full_name in self.config['assets'].items():
            if full_name in data.columns:
                selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                found_assets.append(short_name)
        
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
        
        print(f"  Found {len(found_assets)} assets: {', '.join(found_assets[:10])}...")
        print(f"  Period: {self.returns.index[0].date()} ~ {self.returns.index[-1].date()}")
        print(f"  Working days: {len(self.returns)}")
        
        return self


# =============================================================================
# NETWORK ANALYZER
# =============================================================================
class NetworkAnalyzer:
    """ë„¤íŠ¸ì›Œí¬ í† í´ë¡œì§€ ë¶„ì„"""
    
    def __init__(self, config: dict):
        self.config = config
        self.analysis = config.get('analysis', DEFAULT_CONFIG['analysis'])
        
    @staticmethod
    def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Mantegna distance"""
        dist = np.sqrt(2 * (1 - corr.values))
        np.fill_diagonal(dist, 0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    
    @staticmethod
    def build_mst(corr: pd.DataFrame) -> nx.Graph:
        """MST êµ¬ì¶•"""
        dist = NetworkAnalyzer.corr_to_distance(corr)
        G = nx.Graph()
        for i, a1 in enumerate(corr.columns):
            for j, a2 in enumerate(corr.columns):
                if i < j:
                    G.add_edge(a1, a2, weight=dist.iloc[i, j], corr=corr.iloc[i, j])
        return nx.minimum_spanning_tree(G)
    
    @staticmethod
    def calc_network_sync(corr: pd.DataFrame) -> float:
        """í‰ê·  ìƒê´€ê³„ìˆ˜"""
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        return corr.values[mask].mean()
    
    def get_category(self, asset: str) -> str:
        """ìžì‚° ì¹´í…Œê³ ë¦¬"""
        for cat, assets in self.config.get('categories', {}).items():
            if asset in assets:
                return cat
        return 'OTHER'
    
    def compute_snapshot(self, returns: pd.DataFrame) -> dict:
        """í˜„ìž¬ ìŠ¤ëƒ…ìƒ·"""
        corr = returns.corr()
        mst = self.build_mst(corr)
        bt = nx.betweenness_centrality(mst)
        
        sorted_bt = sorted(bt.items(), key=lambda x: x[1], reverse=True)
        top_hub = sorted_bt[0][0]
        neighbors = list(mst.neighbors(top_hub))
        hub_corrs = [abs(corr.loc[top_hub, n]) for n in neighbors if n in corr.columns]
        hub_avg_corr = np.mean(hub_corrs) if hub_corrs else 0
        
        # Category betweenness
        cat_bt = {}
        for cat in self.config.get('categories', {}).keys():
            cat_assets = self.config['categories'][cat]
            cat_bt[cat] = sum(bt.get(a, 0) for a in cat_assets)
        
        return {
            'corr': corr,
            'mst': mst,
            'betweenness': bt,
            'top_hub': top_hub,
            'top_hub_bt': sorted_bt[0][1],
            'top3': sorted_bt[:3],
            'neighbors': neighbors,
            'hub_avg_corr': hub_avg_corr,
            'hub_influence': sorted_bt[0][1] * hub_avg_corr,
            'network_sync': self.calc_network_sync(corr),
            'cat_betweenness': cat_bt,
        }
    
    def compute_timeseries(self, returns: pd.DataFrame) -> pd.DataFrame:
        """ì‹œê³„ì—´ ì§€í‘œ"""
        window = self.analysis['window']
        step = self.analysis['step']
        
        records = []
        total = (len(returns) - window) // step
        
        for idx, i in enumerate(range(window, len(returns), step)):
            if idx % 20 == 0:
                print(f"  Computing... {idx}/{total}", end='\r')
            
            subset = returns.iloc[i-window:i]
            date = returns.index[i]
            
            try:
                snap = self.compute_snapshot(subset)
                record = {
                    'date': date,
                    'top_hub': snap['top_hub'],
                    'top_hub_bt': snap['top_hub_bt'],
                    'hub_influence': snap['hub_influence'],
                    'hub_avg_corr': snap['hub_avg_corr'],
                    'network_sync': snap['network_sync'],
                }
                for cat, val in snap['cat_betweenness'].items():
                    record[f'bt_{cat}'] = val
                records.append(record)
            except:
                pass
        
        print(f"  Computing... Done ({total} observations)")
        
        df = pd.DataFrame(records).set_index('date')
        
        # Percentiles
        lookback = self.analysis['lookback'] // step
        for col in ['hub_influence', 'network_sync']:
            df[f'{col}_pct'] = df[col].rolling(lookback, min_periods=20).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
            )
        
        return df


# =============================================================================
# VOLATILITY ANALYZER
# =============================================================================
class VolatilityAnalyzer:
    """ë³€ë™ì„± ì§€í‘œ"""
    
    def __init__(self, config: dict):
        self.config = config
        self.analysis = config.get('analysis', DEFAULT_CONFIG['analysis'])
    
    def compute_rv_percentile(self, series: pd.Series) -> pd.Series:
        """RV Percentile"""
        rv = series.rolling(self.analysis['rv_window']).std() * np.sqrt(252) * 100
        return rv.rolling(self.analysis['lookback'], min_periods=60).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
        )
    
    def compute_all(self, returns: pd.DataFrame) -> pd.DataFrame:
        """ëª¨ë“  ìžì‚° RV percentile"""
        result = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            result[col] = self.compute_rv_percentile(returns[col])
        return result


# =============================================================================
# REPORT GENERATOR
# =============================================================================
class ReportGenerator:
    """í…ìŠ¤íŠ¸ ë¦¬í¬íŠ¸"""
    
    def __init__(self, config: dict):
        self.config = config
        self.thresholds = config.get('thresholds', DEFAULT_CONFIG['thresholds'])
    
    def generate(self, snapshot: dict, indicators: pd.DataFrame,
                rv_pct: pd.DataFrame, date: datetime) -> str:
        """ë¦¬í¬íŠ¸ ìƒì„±"""
        
        current = indicators.iloc[-1] if len(indicators) > 0 else {}
        rv_current = rv_pct.iloc[-1] if len(rv_pct) > 0 else {}
        
        # Thresholds
        thresh_sync = indicators['network_sync'].quantile(0.90)
        thresh_hub = indicators['hub_influence'].quantile(0.90)
        
        # Extreme assets
        extreme = [a for a, v in rv_current.items() if v > self.thresholds.get('rv_extreme', 90)]
        elevated = [a for a, v in rv_current.items() 
                   if self.thresholds.get('rv_elevated', 75) < v <= self.thresholds.get('rv_extreme', 90)]
        
        report = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘                         NETWORK MONITOR REPORT                                   â•‘
â•‘                         {date.strftime('%Y-%m-%d %H:%M')}                                       â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
CURRENT NETWORK STATE
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

Top Hub:           {snapshot['top_hub']} (Betweenness: {snapshot['top_hub_bt']:.4f})
Hub Neighbors:     {', '.join(snapshot['neighbors'])}
Hub Avg |Ï|:       {snapshot['hub_avg_corr']:.4f}
Hub Influence:     {snapshot['hub_influence']:.4f} ({current.get('hub_influence_pct', 0):.0f}%ile)
Network Sync:      {snapshot['network_sync']:.4f} ({current.get('network_sync_pct', 0):.0f}%ile)

Top 3 Hubs:
  #1  {snapshot['top3'][0][0]:<12}  Bt = {snapshot['top3'][0][1]:.4f}
  #2  {snapshot['top3'][1][0]:<12}  Bt = {snapshot['top3'][1][1]:.4f}
  #3  {snapshot['top3'][2][0]:<12}  Bt = {snapshot['top3'][2][1]:.4f}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
CATEGORY BETWEENNESS
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        for cat, val in sorted(snapshot['cat_betweenness'].items(), key=lambda x: x[1], reverse=True):
            bar = 'â–ˆ' * int(val * 20)
            report += f"  {cat:<8}: {val:.4f}  {bar}\n"
        
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
THRESHOLDS (Data-Driven, 90%ile)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

Network Sync:      {thresh_sync:.4f}
Hub Influence:     {thresh_hub:.4f}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
VOLATILITY STATE
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

EXTREME (>90%ile): {', '.join(extreme) if extreme else 'None'}
Elevated (>75%ile): {', '.join(elevated) if elevated else 'None'}

"""
        # Top 10 by RV
        rv_sorted = rv_current.sort_values(ascending=False)
        report += "Top 10 by RV Percentile:\n"
        for asset, pct in rv_sorted.head(10).items():
            status = "âš  EXTREME" if pct > 90 else ("â–³ ELEVATED" if pct > 75 else "")
            report += f"  {asset:<12}: {pct:5.1f}%  {status}\n"
        
        # Alerts
        report += f"""
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
ALERTS
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        alerts = []
        
        if current.get('network_sync_pct', 0) > 90:
            alerts.append(f"âš  NETWORK SYNC > 90%ile - Diversification breakdown")
        if current.get('hub_influence_pct', 0) > 90:
            alerts.append(f"âš  HUB INFLUENCE > 90%ile - Hub strongly connected")
        if extreme:
            alerts.append(f"âš  EXTREME VOLATILITY: {', '.join(extreme[:5])}")
        
        # JGB dominance check
        ir_jp = snapshot['cat_betweenness'].get('IR_JP', 0)
        ir_kr = snapshot['cat_betweenness'].get('IR_KR', 0)
        if ir_jp > 0 and ir_kr > 0 and ir_jp / ir_kr > 2:
            alerts.append(f"âš  JGB DOMINANCE: Japan rates as major shock hub")
        
        if alerts:
            for a in alerts:
                report += f"{a}\n"
        else:
            report += "âœ“ No critical alerts\n"
        
        # Key correlations
        report += """
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
KEY CORRELATIONS
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""
        corr = snapshot['corr']
        pairs = [('USDJPY', 'NKY'), ('USDKRW', 'USDJPY'), ('SPX', 'VIX'),
                 (snapshot['top_hub'], 'SPX'), (snapshot['top_hub'], 'VIX')]
        
        for a, b in pairs:
            if a in corr.index and b in corr.columns:
                report += f"  Ï({a}, {b}): {corr.loc[a, b]:+.4f}\n"
        
        report += "â•" * 82 + "\n"
        
        return report


# =============================================================================
# VISUALIZER
# =============================================================================
class Visualizer:
    """ëŒ€ì‹œë³´ë“œ ì‹œê°í™”"""
    
    def __init__(self, config: dict):
        self.config = config
        self.viz = config.get('visualization', DEFAULT_CONFIG['visualization'])
        self.colors = self.viz['colors']
        self.cat_colors = config.get('category_colors', DEFAULT_CONFIG['category_colors'])
    
    def create_dashboard(self, snapshot: dict, indicators: pd.DataFrame,
                        rv_pct: pd.DataFrame, output_path: str):
        """ëŒ€ì‹œë³´ë“œ ìƒì„±"""
        
        fig = plt.figure(figsize=self.viz['figsize'], facecolor=self.colors['bg'])
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.3, 1, 1], hspace=0.3, wspace=0.25)
        
        # Panel 1: Network
        ax1 = fig.add_subplot(gs[0, :2])
        self._draw_network(ax1, snapshot)
        
        # Panel 2: Metrics
        ax2 = fig.add_subplot(gs[0, 2])
        self._draw_metrics(ax2, snapshot, indicators)
        
        # Panel 3: Hub History
        ax3 = fig.add_subplot(gs[1, 0])
        self._draw_hub_history(ax3, indicators)
        
        # Panel 4: Hub Influence
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_hub_influence(ax4, indicators)
        
        # Panel 5: Category Stack
        ax5 = fig.add_subplot(gs[1, 2])
        self._draw_category_stack(ax5, indicators)
        
        # Panel 6: Network Sync
        ax6 = fig.add_subplot(gs[2, 0])
        self._draw_network_sync(ax6, indicators)
        
        # Panel 7: RV Heatmap
        ax7 = fig.add_subplot(gs[2, 1:])
        self._draw_rv_heatmap(ax7, rv_pct)
        
        # Title
        date = indicators.index[-1] if len(indicators) > 0 else datetime.now()
        fig.suptitle(f'NETWORK MONITOR - {date.strftime("%Y-%m-%d")}',
                    fontsize=20, fontweight='bold', color='white', y=0.995)
        
        plt.savefig(output_path, dpi=self.viz['dpi'], 
                   facecolor=self.colors['bg'], bbox_inches='tight')
        plt.close()
    
    def _get_category(self, asset: str) -> str:
        for cat, assets in self.config.get('categories', {}).items():
            if asset in assets:
                return cat
        return 'OTHER'
    
    def _draw_network(self, ax, snapshot):
        ax.set_facecolor(self.colors['panel'])
        
        mst = snapshot['mst']
        bt = snapshot['betweenness']
        
        node_colors = [self.cat_colors.get(self._get_category(n), '#8b949e') for n in mst.nodes()]
        node_sizes = [400 + bt[n] * 6000 for n in mst.nodes()]
        
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
        nx.draw_networkx_labels(mst, pos, ax=ax, font_size=9, font_color='white', font_weight='bold')
        
        for rank, (node, _) in enumerate(snapshot['top3'], 1):
            x, y = pos[node]
            ax.annotate(f'#{rank}', xy=(x, y), xytext=(x+0.08, y+0.08),
                       fontsize=14, color=self.colors['danger'], fontweight='bold')
        
        ax.set_title(f"CURRENT NETWORK (Top Hub: {snapshot['top_hub']})",
                    fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.axis('off')
        
        legend = [mpatches.Patch(facecolor=c, label=cat) for cat, c in self.cat_colors.items()]
        ax.legend(handles=legend, loc='lower left', fontsize=8,
                 facecolor=self.colors['panel'], labelcolor=self.colors['text'])
    
    def _draw_metrics(self, ax, snapshot, indicators):
        ax.set_facecolor(self.colors['panel'])
        ax.axis('off')
        
        current = indicators.iloc[-1] if len(indicators) > 0 else {}
        
        text = f"""
TOP 3 HUBS
{'â”€' * 30}
#1  {snapshot['top3'][0][0]:<10} {snapshot['top3'][0][1]:.4f}
#2  {snapshot['top3'][1][0]:<10} {snapshot['top3'][1][1]:.4f}
#3  {snapshot['top3'][2][0]:<10} {snapshot['top3'][2][1]:.4f}

METRICS
{'â”€' * 30}
Hub Influence: {snapshot['hub_influence']:.4f}
               ({current.get('hub_influence_pct', 0):.0f}%ile)
Network Sync:  {snapshot['network_sync']:.4f}
               ({current.get('network_sync_pct', 0):.0f}%ile)

CATEGORY BETWEENNESS
{'â”€' * 30}
"""
        for cat, val in sorted(snapshot['cat_betweenness'].items(), key=lambda x: x[1], reverse=True):
            text += f"{cat:<6}: {val:.4f}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               color=self.colors['text'], va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.5))
    
    def _draw_hub_history(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        
        colors = [self.cat_colors.get(self._get_category(h), 'gray') for h in indicators['top_hub']]
        ax.scatter(indicators.index, indicators['top_hub_bt'], c=colors, s=40, alpha=0.8)
        ax.plot(indicators.index, indicators['top_hub_bt'], color=self.colors['text'], lw=0.5, alpha=0.3)
        
        ax.set_ylabel('Top Hub Betweenness', color=self.colors['text'])
        ax.set_title('TOP HUB CHANGES', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_hub_influence(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        
        ax.fill_between(indicators.index, 0, indicators['hub_influence'],
                       color=self.colors['accent'], alpha=0.3)
        ax.plot(indicators.index, indicators['hub_influence'], color=self.colors['accent'], lw=1.5)
        
        thresh = indicators['hub_influence'].quantile(0.90)
        ax.axhline(thresh, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)
        
        ax.set_ylabel('Hub Influence', color=self.colors['text'])
        ax.set_title(f'HUB INFLUENCE (90%ile: {thresh:.4f})', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_category_stack(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        
        cats = [c for c in self.config.get('categories', {}).keys() if f'bt_{c}' in indicators.columns]
        cat_data = np.array([indicators[f'bt_{c}'].values for c in cats])
        
        if cat_data.size > 0:
            cat_data_pct = cat_data / cat_data.sum(axis=0) * 100
            colors = [self.cat_colors.get(c, 'gray') for c in cats]
            ax.stackplot(indicators.index, cat_data_pct, labels=cats, colors=colors, alpha=0.8)
            ax.legend(loc='upper left', fontsize=7, facecolor=self.colors['panel'],
                     labelcolor=self.colors['text'], ncol=2)
        
        ax.set_ylabel('Share (%)', color=self.colors['text'])
        ax.set_title('CATEGORY DOMINANCE', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.set_ylim(0, 100)
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_network_sync(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        
        thresh = indicators['network_sync'].quantile(0.90)
        ax.fill_between(indicators.index, thresh, indicators['network_sync'].max() * 1.1,
                       where=indicators['network_sync'] > thresh, alpha=0.3, color=self.colors['danger'])
        ax.axhline(thresh, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)
        ax.plot(indicators.index, indicators['network_sync'], color=self.colors['safe'], lw=1.5)
        
        ax.set_ylabel('Avg Correlation', color=self.colors['text'])
        ax.set_title(f'NETWORK SYNC (90%ile: {thresh:.4f})', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
    
    def _draw_rv_heatmap(self, ax, rv_pct):
        ax.set_facecolor(self.colors['panel'])
        
        days = self.viz.get('heatmap_days', 90)
        rv_recent = rv_pct.iloc[-days:]
        
        cmap = LinearSegmentedColormap.from_list('stress', 
            ['#238636', '#3fb950', '#d29922', '#f85149', '#da3633'], N=100)
        
        im = ax.imshow(rv_recent.T.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)
        
        ax.set_yticks(range(len(rv_recent.columns)))
        ax.set_yticklabels(rv_recent.columns, color=self.colors['text'], fontsize=9)
        
        n = len(rv_recent)
        ticks = [0, n//4, n//2, 3*n//4, n-1]
        labels = [rv_recent.index[i].strftime('%m/%d') for i in ticks if i < len(rv_recent)]
        ax.set_xticks(ticks[:len(labels)])
        ax.set_xticklabels(labels, color=self.colors['text'])
        
        ax.set_title(f'RV PERCENTILE ({days}d)', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Percentile', color=self.colors['text'])
        cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.colors['text'])


# =============================================================================
# MAIN
# =============================================================================
def load_config(config_path: str = None) -> dict:
    """ì„¤ì • ë¡œë“œ"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"  Config loaded from {config_path}")
        return config
    
    # Try default location
    default_paths = ['config/config.yaml', './config.yaml', '../config/config.yaml']
    for p in default_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"  Config loaded from {p}")
            return config
    
    print("  Using default config")
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description='Market Network Monitor v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-d', '--data', required=True, help='Path to MARKET_WATCH.xlsx')
    parser.add_argument('-o', '--output', default='./output', help='Output directory')
    parser.add_argument('-c', '--config', help='Path to config.yaml')
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    parser.add_argument('--date', help='Analysis date (YYYY-MM-DD)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MARKET NETWORK MONITOR v1.0")
    print("=" * 70)
    
    # Load config
    print("\n[1/5] Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[2/5] Loading market data...")
    loader = DataLoader(args.data, config)
    loader.load()
    
    # Compute network metrics
    print("\n[3/5] Computing network metrics...")
    network = NetworkAnalyzer(config)
    indicators = network.compute_timeseries(loader.returns)
    
    window = config.get('analysis', {}).get('window', 60)
    recent = loader.returns.iloc[-window:]
    snapshot = network.compute_snapshot(recent)
    print(f"  Top Hub: {snapshot['top_hub']} (Bt={snapshot['top_hub_bt']:.4f})")
    
    # Compute volatility
    print("\n[4/5] Computing volatility metrics...")
    vol = VolatilityAnalyzer(config)
    rv_pct = vol.compute_all(loader.returns)
    print("  Done")
    
    # Generate outputs
    print("\n[5/5] Generating outputs...")
    
    date = loader.returns.index[-1]
    date_str = date.strftime('%Y%m%d')
    
    output_cfg = config.get('output', DEFAULT_CONFIG['output'])
    
    # Report
    if output_cfg.get('save_report', True):
        report_gen = ReportGenerator(config)
        report = report_gen.generate(snapshot, indicators, rv_pct, date)
        
        report_path = output_dir / f"{output_cfg['report_prefix']}_{date_str}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Report: {report_path}")
        
        if not args.quiet:
            print("\n" + "=" * 70)
            print(report)
    
    # Dashboard
    if output_cfg.get('save_dashboard', True) and not args.report_only:
        viz = Visualizer(config)
        dashboard_path = output_dir / f"{output_cfg['dashboard_prefix']}_{date_str}.png"
        viz.create_dashboard(snapshot, indicators, rv_pct, str(dashboard_path))
        print(f"  Dashboard: {dashboard_path}")
    
    # Indicators
    if output_cfg.get('save_indicators', True):
        indicators_path = output_dir / 'indicators.pkl'
        indicators.to_pickle(indicators_path)
        print(f"  Indicators: {indicators_path}")
    
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
