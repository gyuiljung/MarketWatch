"""
V8DB data loader module for Market Monitor.

Handles loading V8DB_daily.xlsx with factor data for signal generation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from ..core.exceptions import DataLoadError, InsufficientDataError

logger = logging.getLogger(__name__)

# V8DB Excel structure constants
V8DB_HEADER_ROW = 1  # Row 2 (0-indexed: 1) contains factor names
V8DB_DATA_START_ROW = 3  # Row 4 (0-indexed: 3) starts data


@dataclass
class V8DBData:
    """Container for V8DB data."""
    prices: pd.DataFrame
    returns: pd.DataFrame
    factors: List[str]
    bm_name: str
    bm_prices: pd.Series
    bm_returns: pd.Series
    period: Tuple[pd.Timestamp, pd.Timestamp]
    working_days: int

    def __repr__(self) -> str:
        return (
            f"V8DBData(bm={self.bm_name}, factors={len(self.factors)}, "
            f"period={self.period[0].date()} ~ {self.period[1].date()}, "
            f"days={self.working_days})"
        )


class V8DBLoader:
    """
    Loader for V8DB_daily.xlsx.

    V8DB contains factor data for KOSPI and 3Y bond signal generation.
    """

    # Default V8DB path
    DEFAULT_PATH = Path('C:/Users/infomax/Documents/V8DB_daily.xlsx')

    # Sheet mapping
    SHEET_MAP = {
        'kospi': 'KOSPI',
        'KOSPI': 'KOSPI',
        '3y': '3y',
        '3ybm': '3y',
        'bond': '3y',
    }

    # Rate assets (use diff instead of log return)
    RATE_KEYWORDS = ['수익률', '금리', 'yield', 'rate', 'IRS', 'CRS', 'NDF']

    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize V8DB loader.

        Args:
            filepath: Path to V8DB_daily.xlsx (default: Documents/V8DB_daily.xlsx)
        """
        self.filepath = Path(filepath) if filepath else self.DEFAULT_PATH
        self._cache: Dict[str, V8DBData] = {}

    def load(self, sheet: str = 'kospi', lookback: int = 252) -> V8DBData:
        """
        Load V8DB data for specified sheet.

        Args:
            sheet: 'kospi' or '3y' (or aliases)
            lookback: Number of days to load (default: 252)

        Returns:
            V8DBData object
        """
        cache_key = f"{sheet}_{lookback}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        sheet_name = self.SHEET_MAP.get(sheet.lower(), sheet)
        logger.info(f"Loading V8DB sheet: {sheet_name}")

        if not self.filepath.exists():
            raise DataLoadError(
                f"V8DB file not found: {self.filepath}",
                "Please provide valid path to V8DB_daily.xlsx"
            )

        try:
            df = pd.read_excel(self.filepath, sheet_name=sheet_name, header=None)
        except Exception as e:
            raise DataLoadError(f"Failed to read V8DB: {e}") from e

        # Parse V8DB format
        data = self._parse_v8db_format(df, sheet_name)

        # Apply lookback
        if lookback and len(data.returns) > lookback:
            data = V8DBData(
                prices=data.prices.iloc[-lookback:],
                returns=data.returns.iloc[-lookback:],
                factors=data.factors,
                bm_name=data.bm_name,
                bm_prices=data.bm_prices.iloc[-lookback:],
                bm_returns=data.bm_returns.iloc[-lookback:],
                period=(data.returns.index[-lookback], data.returns.index[-1]),
                working_days=lookback,
            )

        self._cache[cache_key] = data
        return data

    def _parse_v8db_format(self, df: pd.DataFrame, sheet_name: str) -> V8DBData:
        """
        Parse V8DB Excel format.

        Args:
            df: Raw DataFrame from Excel
            sheet_name: Sheet name for BM identification

        Returns:
            V8DBData object
        """
        # Extract factor names from header row
        factor_names = df.iloc[V8DB_HEADER_ROW, :].tolist()

        # Clean factor names
        factors = []
        for i, name in enumerate(factor_names):
            if pd.notna(name) and str(name).strip() and str(name) != '"':
                factors.append(str(name).strip())
            else:
                factors.append(f'_col_{i}')

        # Extract data
        data = df.iloc[V8DB_DATA_START_ROW:, :].copy()
        data.columns = ['Date'] + factors[1:]  # First column is date

        # Parse dates
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
        data = data.set_index('Date').sort_index()

        # Convert to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Identify BM column (last column typically)
        bm_col = data.columns[-1]
        bm_name = bm_col

        # Calculate returns
        returns = self._calculate_returns(data, factors[1:])

        # Filter working days (non-zero BM)
        mask = data[bm_col] != 0
        data = data[mask]
        returns = returns.loc[data.index]

        # Drop NaN
        returns = returns.dropna()
        data = data.loc[returns.index]

        if len(returns) < 60:
            raise InsufficientDataError(60, len(returns), "V8DB loading")

        # BM series
        bm_prices = data[bm_col]
        bm_returns = returns[bm_col] if bm_col in returns.columns else bm_prices.pct_change()

        logger.info(f"Loaded V8DB: {len(data)} rows, {len(data.columns)} factors")
        logger.info(f"BM: {bm_name}")
        logger.info(f"Period: {data.index[0].date()} ~ {data.index[-1].date()}")

        return V8DBData(
            prices=data,
            returns=returns,
            factors=list(data.columns),
            bm_name=bm_name,
            bm_prices=bm_prices,
            bm_returns=bm_returns,
            period=(data.index[0], data.index[-1]),
            working_days=len(data),
        )

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        factor_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate returns (log returns or diff based on factor type).

        Args:
            prices: Price DataFrame
            factor_names: List of factor names for type detection

        Returns:
            Returns DataFrame
        """
        returns = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            # Check if rate asset
            is_rate = any(kw in str(col) for kw in self.RATE_KEYWORDS)

            if is_rate:
                returns[col] = prices[col].diff()
            else:
                # Log return with zero handling
                with np.errstate(divide='ignore', invalid='ignore'):
                    ret = np.log(prices[col] / prices[col].shift(1))
                    ret = ret.replace([np.inf, -np.inf], np.nan)
                returns[col] = ret

        return returns

    def get_latest_prices(self, sheet: str = 'kospi') -> pd.Series:
        """
        Get latest prices for all factors.

        Args:
            sheet: 'kospi' or '3y'

        Returns:
            Series of latest prices
        """
        data = self.load(sheet, lookback=5)
        return data.prices.iloc[-1]

    def get_bm_return(self, date: str, sheet: str = 'kospi') -> Optional[float]:
        """
        Get BM return for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)
            sheet: 'kospi' or '3y'

        Returns:
            BM return (%) or None
        """
        try:
            data = self.load(sheet)
            target = pd.Timestamp(date)

            if target in data.bm_returns.index:
                ret = data.bm_returns.loc[target]
                if pd.notna(ret):
                    return round(float(ret) * 100, 4)

            return None
        except Exception as e:
            logger.warning(f"Failed to get BM return: {e}")
            return None
