"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    assets = ["SPX", "NKY", "USDJPY", "USDKRW", "VIX", "GOLD", "WTI"]

    data = np.random.randn(len(dates), len(assets)) * 0.01
    returns = pd.DataFrame(data, index=dates, columns=assets)

    # Add some correlations
    returns["VIX"] = -returns["SPX"] * 0.5 + np.random.randn(len(dates)) * 0.01

    return returns


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "assets": {
            "SPX": "S&P 500",
            "NKY": "Nikkei 225",
            "USDJPY": "USD/JPY",
            "USDKRW": "USD/KRW",
            "VIX": "VIX Index",
            "GOLD": "Gold",
            "WTI": "WTI Crude",
        },
        "rate_assets": ["USDJPY", "USDKRW"],
        "categories": {
            "EQUITY": ["SPX", "NKY"],
            "FX": ["USDJPY", "USDKRW"],
            "CMDTY": ["GOLD", "WTI"],
            "VOL": ["VIX"],
        },
        "analysis": {
            "window": 60,
            "rv_windows": [5, 10, 20],
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config_dict):
    """Create temporary config file."""
    import yaml

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)

    return config_path
