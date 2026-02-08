"""
Unified Data Loader
===================
MARKET_WATCH.xlsx 로더 (market_monitor ClusteredDataLoader 기반 통합)

- Core assets + Cluster assets 로딩
- 금리는 diff, 나머지는 log return
- USDKRW 기준 워킹데이 필터
"""
import numpy as np
import pandas as pd
from pathlib import Path


class MarketDataLoader:
    """MARKET_WATCH.xlsx 통합 로더"""

    def __init__(self, filepath: str, config: dict):
        self.filepath = filepath
        self.config = config
        self.prices = None
        self.returns = None
        self.core_assets = []
        self.cluster_reps = []
        self.all_assets = []

    def load(self) -> 'MarketDataLoader':
        print(f"  Loading {self.filepath}...")

        df = pd.read_excel(self.filepath, header=None)
        data = df.iloc[4:, :].copy()
        asset_names = df.iloc[2, :].tolist()
        data.columns = asset_names
        data = data.rename(columns={asset_names[0]: 'Date'})
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        selected = pd.DataFrame()

        # Core assets
        core_cfg = self.config.get('core_assets', {})
        for short_name, full_name in core_cfg.items():
            if full_name in data.columns:
                selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                self.core_assets.append(short_name)
                self.all_assets.append(short_name)

        # Cluster assets
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

        # Returns: log for prices, diff for rates
        returns = np.log(selected / selected.shift(1))
        for col in self.config.get('rate_assets', []):
            if col in returns.columns:
                returns[col] = selected[col].diff()

        # Working days (USDKRW 변동 있는 날)
        if 'USDKRW' in returns.columns:
            working = returns['USDKRW'][returns['USDKRW'] != 0].index
            returns = returns.loc[working]
            self.prices = self.prices.loc[working]

        self.returns = returns.dropna()
        self.prices = self.prices.loc[self.returns.index]

        print(f"  Core assets ({len(self.core_assets)}): {', '.join(self.core_assets[:8])}")
        print(f"  Cluster reps ({len(self.cluster_reps)}): {', '.join(self.cluster_reps)}")
        print(f"  Period: {self.returns.index[0].date()} ~ {self.returns.index[-1].date()}")
        print(f"  Working days: {len(self.returns)}")

        return self

    def get_network_assets(self) -> list:
        """Core + Cluster representatives for network analysis"""
        return self.core_assets + self.cluster_reps

    def get_recent(self, window: int) -> pd.DataFrame:
        """최근 N일 returns"""
        if len(self.returns) >= window:
            return self.returns.iloc[-window:]
        return self.returns
