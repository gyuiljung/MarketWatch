"""Tests for network analysis module."""

import pytest
import pandas as pd
import numpy as np

from market_monitor.analysis.network import NetworkAnalyzer, NetworkSnapshot
from market_monitor.core.config import ConfigLoader


class TestNetworkAnalyzer:
    """Tests for NetworkAnalyzer class."""

    @pytest.fixture
    def analyzer(self, temp_config_file):
        """Create NetworkAnalyzer instance."""
        config = ConfigLoader.load(temp_config_file)
        return NetworkAnalyzer(config)

    def test_compute_snapshot(self, analyzer, sample_returns):
        """Test computing network snapshot."""
        network_assets = ["SPX", "NKY", "VIX", "GOLD", "WTI"]

        snapshot = analyzer.compute_snapshot(
            sample_returns,
            network_assets,
            window=60
        )

        assert isinstance(snapshot, NetworkSnapshot)
        assert snapshot.top_hub_bt in network_assets
        assert 0 <= snapshot.network_sync <= 1
        assert len(snapshot.top3_bt) == 3

    def test_compute_timeseries(self, analyzer, sample_returns):
        """Test computing time series indicators."""
        network_assets = ["SPX", "NKY", "VIX", "GOLD", "WTI"]

        indicators = analyzer.compute_timeseries(
            sample_returns,
            network_assets,
            step=5
        )

        assert isinstance(indicators, pd.DataFrame)
        assert "top_hub_bt" in indicators.columns
        assert "hub_influence" in indicators.columns
        assert len(indicators) > 0

    def test_empty_returns(self, analyzer):
        """Test handling empty returns DataFrame."""
        empty_returns = pd.DataFrame()

        with pytest.raises(Exception):
            analyzer.compute_snapshot(empty_returns, [], window=60)

    def test_insufficient_assets(self, analyzer, sample_returns):
        """Test handling insufficient assets."""
        # MST needs at least 3 nodes
        snapshot = analyzer.compute_snapshot(
            sample_returns[["SPX", "NKY"]],
            ["SPX", "NKY"],
            window=60
        )

        # Should return empty or minimal snapshot
        assert snapshot is not None


class TestNetworkSnapshot:
    """Tests for NetworkSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating NetworkSnapshot manually."""
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C")])

        snapshot = NetworkSnapshot(
            mst=G,
            corr=pd.DataFrame(),
            betweenness={"A": 0.0, "B": 1.0, "C": 0.0},
            top3_bt=[("B", 1.0), ("A", 0.0), ("C", 0.0)],
            top_hub_bt="B",
            neighbors=["A", "C"],
            hub_avg_corr=0.5,
            hub_influence=0.8,
            network_sync=0.6,
            cat_betweenness={},
        )

        assert snapshot.top_hub_bt == "B"
        assert len(snapshot.neighbors) == 2
