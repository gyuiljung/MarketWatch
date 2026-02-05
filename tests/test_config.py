"""Tests for configuration module."""

import pytest
from market_monitor.core.config import Config, ConfigLoader
from market_monitor.core.exceptions import ConfigError


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        config = ConfigLoader.load(temp_config_file)

        assert isinstance(config, Config)
        assert "SPX" in config.assets
        assert "USDJPY" in config.rate_assets
        assert "EQUITY" in config.categories

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(ConfigError):
            ConfigLoader.load(tmp_path / "nonexistent.yaml")

    def test_load_or_default(self, tmp_path):
        """Test load_or_default returns default config when file missing."""
        config = ConfigLoader.load_or_default(tmp_path / "missing.yaml")

        assert isinstance(config, Config)
        assert len(config.assets) > 0


class TestConfig:
    """Tests for Config dataclass."""

    def test_is_clustered_false(self, temp_config_file):
        """Test is_clustered property when no cluster config."""
        config = ConfigLoader.load(temp_config_file)

        # Default config has no clustered section
        assert not config.is_clustered

    def test_network_assets(self, temp_config_file):
        """Test network_assets excludes rate assets."""
        config = ConfigLoader.load(temp_config_file)

        network_assets = [a for a in config.assets if a not in config.rate_assets]

        assert "USDJPY" not in network_assets
        assert "SPX" in network_assets
