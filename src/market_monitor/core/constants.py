"""
Constants for Market Monitor.

All magic numbers and hardcoded values should be defined here.
This makes the codebase more maintainable and configurable.
"""

from typing import Dict, List

# =============================================================================
# Version Info
# =============================================================================

VERSION = "2.4.0"
VERSION_NAME = "Market Network Monitor"

# =============================================================================
# Transfer Entropy Constants
# =============================================================================

TE_DEFAULT_BINS = 6
TE_DEFAULT_LAG = 1
TE_DEFAULT_SURROGATES = 30
TE_DEFAULT_ALPHA = 0.05
TE_MIN_SAMPLES = 60
TE_DEFAULT_TOP_N = 15

# =============================================================================
# Tail Dependence Constants
# =============================================================================

# Expected tail dependence approximation: expected = q + TAIL_CORR_COEFFICIENT * |corr|
# This is a linear approximation for bivariate normal tail behavior
TAIL_CORR_COEFFICIENT = 0.35

# Interpretation thresholds for excess tail dependence
TAIL_CRISIS_THRESHOLD = 0.20      # excess > 0.20: Crisis contagion
TAIL_ELEVATED_THRESHOLD = 0.15    # excess > 0.15: Elevated contagion
TAIL_MILD_THRESHOLD = 0.10        # excess > 0.10: Mild excess
TAIL_DIVERSIFIED_THRESHOLD = -0.10  # excess < -0.10: Tail diversified

# Default tail percentiles
TAIL_DEFAULT_Q = 0.10  # 10% tail

# =============================================================================
# Network Analysis Constants
# =============================================================================

MST_MIN_NODES = 3
CORRELATION_THRESHOLD = 0.1  # Minimum |correlation| for edge inclusion

# Percentile thresholds for alerts
PERCENTILE_EXTREME = 90
PERCENTILE_ELEVATED = 75

# =============================================================================
# Volatility Constants
# =============================================================================

RV_DEFAULT_WINDOW = 5
RV_ANNUALIZATION_FACTOR = 252  # Trading days per year

# RV Percentile thresholds
RV_EXTREME_PERCENTILE = 90
RV_ELEVATED_PERCENTILE = 75

# =============================================================================
# Analysis Window Constants
# =============================================================================

DEFAULT_WINDOW = 60
DEFAULT_STEP = 5
DEFAULT_LOOKBACK = 252

# Multi-scale windows for synchronization analysis
DEFAULT_SYNC_WINDOWS: List[int] = [5, 20, 60]

# Multi-scale network analysis windows (for full MST snapshots)
# 1W=5, 1M=22, 3M=66, 1Y=252 trading days
NETWORK_SCALE_WINDOWS: Dict[str, int] = {
    '1W': 5,
    '1M': 22,
    '3M': 66,
    '1Y': 252,
}

# =============================================================================
# Visualization Colors (Dark Theme - GitHub Style)
# =============================================================================

COLORS: Dict[str, str] = {
    'bg': '#0d1117',
    'panel': '#161b22',
    'text': '#e6edf3',
    'grid': '#30363d',
    'danger': '#f85149',
    'warning': '#d29922',
    'safe': '#3fb950',
    'accent': '#58a6ff',
}

# Category colors for assets
CATEGORY_COLORS: Dict[str, str] = {
    'EQ': '#3fb950',     # Green - Equity
    'FX': '#f85149',     # Red - Foreign Exchange
    'IR_JP': '#f78166',  # Orange - Japan Rates
    'IR_KR': '#58a6ff',  # Blue - Korea Rates
    'IR_US': '#a5d6ff',  # Light Blue - US Rates
    'VOL': '#8b949e',    # Gray - Volatility
    'CMD': '#d29922',    # Gold - Commodities
}

# =============================================================================
# Dashboard Constants
# =============================================================================

DEFAULT_FIGSIZE = (24, 20)
DEFAULT_DPI = 150
DEFAULT_HEATMAP_DAYS = 90

# =============================================================================
# File Output Constants
# =============================================================================

DEFAULT_REPORT_PREFIX = "network_report"
DEFAULT_DASHBOARD_PREFIX = "network_monitor"
DEFAULT_INDICATORS_FILENAME = "indicators.pkl"

# =============================================================================
# Data Loading Constants
# =============================================================================

# Excel file structure (MARKET_WATCH.xlsx format)
EXCEL_HEADER_ROW = 2  # Asset names are in row 3 (0-indexed: 2)
EXCEL_DATA_START_ROW = 4  # Data starts from row 5 (0-indexed: 4)

# Working day filter - use USDKRW as reference for Korean market days
WORKING_DAY_REFERENCE = 'USDKRW'

# =============================================================================
# Sync Thresholds
# =============================================================================

# Network sync divergence threshold for regime shift warning
SYNC_DIVERGENCE_WARNING = 0.08

# Default sync warning level
SYNC_WARNING_LEVEL = 0.20
