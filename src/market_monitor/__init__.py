"""
Market Monitor - Network Topology Based Market Stress Monitoring Tool

A Python package for analyzing market network structure, transfer entropy,
and volatility dynamics across multiple asset classes.
"""

__version__ = "2.4.0"
__author__ = "Market Monitor Team"

from .core.config import Config, ConfigLoader
from .core.exceptions import MarketMonitorError

__all__ = [
    "Config",
    "ConfigLoader",
    "MarketMonitorError",
    "__version__",
]
