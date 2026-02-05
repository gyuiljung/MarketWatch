"""
Data loading module for Market Monitor.

Handles loading and preprocessing of MARKET_WATCH.xlsx data files.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import EXCEL_HEADER_ROW, EXCEL_DATA_START_ROW, WORKING_DAY_REFERENCE
from ..core.exceptions import (
    DataLoadError,
    InvalidFormatError,
    MissingColumnError,
    InsufficientDataError,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadedData:
    """Container for loaded market data."""
    prices: pd.DataFrame
    returns: pd.DataFrame
    assets: List[str]
    period: Tuple[pd.Timestamp, pd.Timestamp]
    working_days: int

    def __repr__(self) -> str:
        return (
            f"LoadedData(assets={len(self.assets)}, "
            f"period={self.period[0].date()} ~ {self.period[1].date()}, "
            f"working_days={self.working_days})"
        )


class BaseDataLoader:
    """
    Base class for data loaders.

    Provides common functionality for loading MARKET_WATCH.xlsx files.
    """

    def __init__(self, filepath: Path, config: Config):
        """
        Initialize data loader.

        Args:
            filepath: Path to MARKET_WATCH.xlsx file
            config: Configuration object
        """
        self.filepath = Path(filepath)
        self.config = config
        self._data: Optional[LoadedData] = None

    @property
    def data(self) -> LoadedData:
        """Get loaded data, loading if necessary."""
        if self._data is None:
            self._data = self.load()
        return self._data

    def load(self) -> LoadedData:
        """Load and preprocess data. Override in subclasses."""
        raise NotImplementedError

    def _read_excel(self) -> pd.DataFrame:
        """
        Read Excel file with MARKET_WATCH format.

        Returns:
            Raw DataFrame from Excel

        Raises:
            DataLoadError: If file cannot be read
        """
        logger.info(f"Loading data from {self.filepath}")

        if not self.filepath.exists():
            raise DataLoadError(
                f"Data file not found: {self.filepath}",
                "Please provide a valid path to MARKET_WATCH.xlsx"
            )

        try:
            df = pd.read_excel(self.filepath, header=None)
            logger.debug(f"Read {len(df)} rows from Excel")
            return df
        except PermissionError:
            raise DataLoadError(
                f"Cannot read file: {self.filepath}",
                "File may be open in another application"
            )
        except Exception as e:
            raise DataLoadError(f"Failed to read Excel file: {e}") from e

    def _parse_market_watch_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse MARKET_WATCH.xlsx format.

        Format:
            - Row 3 (index 2): Asset names (OFFSET: A열 자산명 → B열 데이터)
            - Row 5+ (index 4+): Data
            - Column 0 (A열): Dates
            - Column 1+ (B열~): Asset data

        Note:
            MARKET_WATCH.xlsx has a special structure where asset names in Row 2
            are offset by one column. The asset name in column A actually refers
            to data in column B, and so on.

        Args:
            df: Raw DataFrame from Excel

        Returns:
            Parsed DataFrame with Date index and asset columns
        """
        # Extract asset names from header row (with offset handling)
        # MARKET_WATCH.xlsx 특수 구조:
        # - A열 Row 2: 자산명 (예: CNHKRW) → 실제로는 B열 데이터에 해당
        # - B열 Row 2: nan (비어있음)
        # - C열 Row 2: 자산명 (예: USDKRW) → C열 데이터에 해당
        # - A열 데이터: 날짜
        # - B열 데이터: A열 자산명의 값
        asset_names_raw = df.iloc[EXCEL_HEADER_ROW, :].tolist()

        # Build column names:
        # - Column 0 (A열): Date
        # - Column 1 (B열): A열 자산명 (offset 적용)
        # - Column 2+ (C열~): C열 이후 자산명 그대로
        asset_names = ['Date', asset_names_raw[0]] + asset_names_raw[2:]  # B열 자산명(nan) skip

        # Extract data rows
        data = df.iloc[EXCEL_DATA_START_ROW:, :].copy()

        # Ensure column count matches
        if len(data.columns) > len(asset_names):
            asset_names = asset_names + [f'_unnamed_{i}' for i in range(len(data.columns) - len(asset_names))]
        elif len(data.columns) < len(asset_names):
            asset_names = asset_names[:len(data.columns)]

        data.columns = asset_names

        # Parse dates
        try:
            data['Date'] = pd.to_datetime(data['Date'])
        except Exception as e:
            raise InvalidFormatError(
                str(self.filepath),
                "Date column should be parseable as datetime",
                f"Failed to parse dates: {e}"
            )

        data = data.set_index('Date')

        logger.debug(f"Parsed {len(data)} rows, {len(data.columns)} columns")
        return data

    def _select_assets(
        self,
        data: pd.DataFrame,
        asset_mapping: dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select and rename assets based on mapping.

        Args:
            data: Parsed DataFrame
            asset_mapping: Dict of short_name -> full_name

        Returns:
            Tuple of (selected DataFrame, list of found assets)
        """
        selected = pd.DataFrame(index=data.index)
        found_assets = []

        for short_name, full_name in asset_mapping.items():
            if full_name in data.columns:
                selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                found_assets.append(short_name)
            else:
                logger.warning(f"Asset not found in data: {short_name} ({full_name})")

        if not found_assets:
            raise MissingColumnError(
                "No assets found",
                list(data.columns)[:20]
            )

        logger.info(f"Selected {len(found_assets)} assets: {', '.join(found_assets[:8])}...")
        return selected, found_assets

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        rate_assets: List[str]
    ) -> pd.DataFrame:
        """
        Calculate returns (log returns for prices, diff for rates).

        Args:
            prices: Price DataFrame
            rate_assets: List of assets to use diff instead of log return

        Returns:
            Returns DataFrame
        """
        # Log returns for all
        returns = np.log(prices / prices.shift(1))

        # Diff for rate assets
        for col in rate_assets:
            if col in returns.columns:
                returns[col] = prices[col].diff()

        return returns

    def _filter_working_days(
        self,
        df: pd.DataFrame,
        reference: str = WORKING_DAY_REFERENCE
    ) -> pd.DataFrame:
        """
        Filter to working days based on reference asset.

        Args:
            df: DataFrame to filter
            reference: Reference asset for working days (default: USDKRW)

        Returns:
            Filtered DataFrame
        """
        if reference in df.columns:
            # Working days = days where reference has non-zero value
            mask = df[reference] != 0
            working_days = df[mask].index
            filtered = df.loc[working_days]
            logger.debug(f"Filtered to {len(filtered)} working days")
            return filtered
        else:
            logger.warning(f"Reference asset {reference} not found, skipping filter")
            return df


class DataLoader(BaseDataLoader):
    """
    Standard data loader for basic mode (v1.0 compatible).

    Loads all assets defined in config.assets.
    """

    def load(self) -> LoadedData:
        """
        Load and preprocess market data.

        Returns:
            LoadedData object with prices and returns
        """
        # Read Excel
        raw_df = self._read_excel()

        # Parse format
        data = self._parse_market_watch_format(raw_df)

        # Select assets
        prices, found_assets = self._select_assets(data, self.config.assets)

        # Calculate returns
        returns = self._calculate_returns(prices, self.config.rate_assets)

        # Filter working days
        returns = self._filter_working_days(returns)
        prices = prices.loc[returns.index]

        # Drop NaN rows
        returns = returns.dropna()
        prices = prices.loc[returns.index]

        if len(returns) < 60:
            raise InsufficientDataError(60, len(returns), "data loading")

        period = (returns.index[0], returns.index[-1])

        logger.info(f"Loaded data: {period[0].date()} ~ {period[1].date()} ({len(returns)} days)")

        self._data = LoadedData(
            prices=prices,
            returns=returns,
            assets=found_assets,
            period=period,
            working_days=len(returns),
        )
        return self._data


class ClusteredDataLoader(BaseDataLoader):
    """
    Clustered data loader for advanced mode (v2.3).

    Loads core assets and cluster assets separately.
    Provides methods to get network assets (core + representatives).
    """

    def __init__(self, filepath: Path, config: Config):
        super().__init__(filepath, config)
        self.core_assets: List[str] = []
        self.cluster_reps: List[str] = []
        self.all_assets: List[str] = []

    def load(self) -> LoadedData:
        """
        Load and preprocess market data in clustered mode.

        Returns:
            LoadedData object with prices and returns
        """
        # Read Excel
        raw_df = self._read_excel()

        # Parse format
        data = self._parse_market_watch_format(raw_df)

        # Build combined asset mapping
        selected = pd.DataFrame(index=data.index)

        # Load core assets
        core_mapping = self.config.core_assets or {}
        for short_name, full_name in core_mapping.items():
            if full_name in data.columns:
                selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                self.core_assets.append(short_name)
                self.all_assets.append(short_name)

        # Load cluster assets
        if self.config.clusters:
            for cluster_name, cluster_cfg in self.config.clusters.items():
                rep = cluster_cfg.representative

                for short_name, full_name in cluster_cfg.assets.items():
                    if full_name in data.columns:
                        selected[short_name] = pd.to_numeric(data[full_name], errors='coerce')
                        self.all_assets.append(short_name)

                        if short_name == rep and short_name not in self.cluster_reps:
                            self.cluster_reps.append(short_name)

        if not self.all_assets:
            raise MissingColumnError("No assets found", list(data.columns)[:20])

        prices = selected

        # Calculate returns
        returns = self._calculate_returns(prices, self.config.rate_assets)

        # Filter working days
        returns = self._filter_working_days(returns)
        prices = prices.loc[returns.index]

        # Drop NaN
        returns = returns.dropna()
        prices = prices.loc[returns.index]

        if len(returns) < 60:
            raise InsufficientDataError(60, len(returns), "data loading")

        period = (returns.index[0], returns.index[-1])

        logger.info(f"Core assets ({len(self.core_assets)}): {', '.join(self.core_assets[:8])}...")
        logger.info(f"Cluster reps ({len(self.cluster_reps)}): {', '.join(self.cluster_reps)}")
        logger.info(f"Period: {period[0].date()} ~ {period[1].date()} ({len(returns)} days)")

        self._data = LoadedData(
            prices=prices,
            returns=returns,
            assets=self.all_assets,
            period=period,
            working_days=len(returns),
        )
        return self._data

    def get_network_assets(self) -> List[str]:
        """
        Get assets for network analysis.

        Returns:
            List of core assets + cluster representatives
        """
        return self.core_assets + self.cluster_reps

    def get_cluster_assets(self, cluster_name: str) -> List[str]:
        """
        Get all assets in a specific cluster.

        Args:
            cluster_name: Name of the cluster

        Returns:
            List of asset names in the cluster
        """
        if self.config.clusters and cluster_name in self.config.clusters:
            return list(self.config.clusters[cluster_name].assets.keys())
        return []
