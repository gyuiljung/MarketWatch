"""
Configuration management for Market Monitor.

Provides dataclass-based configuration with YAML loading and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
import logging

from .exceptions import (
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    MissingRequiredKeyError,
)
from .constants import (
    DEFAULT_WINDOW,
    DEFAULT_STEP,
    DEFAULT_LOOKBACK,
    DEFAULT_SYNC_WINDOWS,
    TE_DEFAULT_BINS,
    TE_DEFAULT_LAG,
    TE_DEFAULT_SURROGATES,
    TE_DEFAULT_ALPHA,
    TE_DEFAULT_TOP_N,
    RV_EXTREME_PERCENTILE,
    RV_ELEVATED_PERCENTILE,
    SYNC_WARNING_LEVEL,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    DEFAULT_HEATMAP_DAYS,
    COLORS,
    CATEGORY_COLORS,
    DEFAULT_REPORT_PREFIX,
    DEFAULT_DASHBOARD_PREFIX,
    RV_DEFAULT_WINDOW,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Config Dataclasses
# =============================================================================

@dataclass
class AnalysisConfig:
    """Analysis parameters configuration."""
    window: int = DEFAULT_WINDOW
    step: int = DEFAULT_STEP
    rv_window: int = RV_DEFAULT_WINDOW
    lookback: int = DEFAULT_LOOKBACK
    windows: List[int] = field(default_factory=lambda: DEFAULT_SYNC_WINDOWS.copy())
    hub_window: int = DEFAULT_WINDOW
    te_window_short: int = DEFAULT_WINDOW
    te_window_long: int = DEFAULT_LOOKBACK
    tail_window: int = DEFAULT_LOOKBACK
    corr_window: int = DEFAULT_WINDOW
    corr_change_lookback: int = 20


@dataclass
class TransferEntropyConfig:
    """Transfer Entropy configuration."""
    bins: int = TE_DEFAULT_BINS
    lag: int = TE_DEFAULT_LAG
    n_surrogates: int = TE_DEFAULT_SURROGATES
    alpha: float = TE_DEFAULT_ALPHA
    top_n: int = TE_DEFAULT_TOP_N


@dataclass
class ThresholdConfig:
    """Threshold configuration for alerts."""
    rv_extreme: float = RV_EXTREME_PERCENTILE
    rv_elevated: float = RV_ELEVATED_PERCENTILE
    sync_warning: float = SYNC_WARNING_LEVEL
    network_sync_danger: Optional[float] = None
    network_sync_warning: Optional[float] = None
    hub_influence_danger: Optional[float] = None
    hub_influence_warning: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    heatmap_days: int = DEFAULT_HEATMAP_DAYS
    colors: Dict[str, str] = field(default_factory=lambda: COLORS.copy())


@dataclass
class OutputConfig:
    """Output configuration."""
    report_prefix: str = DEFAULT_REPORT_PREFIX
    dashboard_prefix: str = DEFAULT_DASHBOARD_PREFIX
    save_indicators: bool = True
    save_report: bool = True
    save_dashboard: bool = True


@dataclass
class ClusterConfig:
    """Configuration for a single cluster."""
    representative: str
    assets: Dict[str, str]
    color: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""

    # Required fields
    assets: Dict[str, str]
    rate_assets: List[str]
    categories: Dict[str, List[str]]
    category_colors: Dict[str, str]

    # Optional nested configs
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    transfer_entropy: TransferEntropyConfig = field(default_factory=TransferEntropyConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Clustered mode fields (optional)
    core_assets: Optional[Dict[str, str]] = None
    clusters: Optional[Dict[str, ClusterConfig]] = None
    key_pairs: Optional[List[List[str]]] = None

    @property
    def is_clustered(self) -> bool:
        """Check if config is in clustered mode."""
        return self.core_assets is not None and self.clusters is not None

    def get_all_assets(self) -> Dict[str, str]:
        """Get all assets (core + cluster assets for clustered mode)."""
        if self.is_clustered:
            all_assets = dict(self.core_assets) if self.core_assets else {}
            if self.clusters:
                for cluster in self.clusters.values():
                    all_assets.update(cluster.assets)
            return all_assets
        return self.assets

    def get_network_assets(self) -> List[str]:
        """Get assets for network analysis (core + representatives for clustered mode)."""
        if self.is_clustered:
            network_assets = list(self.core_assets.keys()) if self.core_assets else []
            if self.clusters:
                for cluster in self.clusters.values():
                    if cluster.representative not in network_assets:
                        network_assets.append(cluster.representative)
            return network_assets
        return list(self.assets.keys())


# =============================================================================
# Config Loader
# =============================================================================

class ConfigLoader:
    """Configuration file loader and validator."""

    REQUIRED_KEYS = ['assets', 'rate_assets', 'categories']
    REQUIRED_KEYS_CLUSTERED = ['core_assets', 'clusters', 'rate_assets', 'categories']

    @classmethod
    def load(cls, path: Path) -> Config:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Config object

        Raises:
            ConfigNotFoundError: If file doesn't exist
            ConfigValidationError: If validation fails
        """
        path = Path(path)
        if not path.exists():
            raise ConfigNotFoundError(str(path))

        logger.info(f"Loading configuration from {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML: {e}")

        cls.validate(raw_config)
        return cls._build_config(raw_config)

    @classmethod
    def load_or_default(cls, path: Optional[Path] = None) -> Config:
        """
        Load config from path, falling back to default if not found.

        Args:
            path: Optional path to config file

        Returns:
            Config object (from file or default)
        """
        if path:
            try:
                return cls.load(path)
            except ConfigNotFoundError:
                logger.warning(f"Config not found at {path}, using default")

        # Try default locations
        default_paths = [
            Path('config/config.yaml'),
            Path('./config.yaml'),
            Path('../config/config.yaml'),
        ]

        for p in default_paths:
            if p.exists():
                logger.info(f"Found config at {p}")
                return cls.load(p)

        logger.info("Using default configuration")
        return cls.get_default()

    @classmethod
    def validate(cls, raw_config: dict) -> None:
        """
        Validate raw configuration dictionary.

        Args:
            raw_config: Dictionary from YAML

        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []

        # Check if clustered mode
        is_clustered = 'core_assets' in raw_config or 'clusters' in raw_config

        if is_clustered:
            required = cls.REQUIRED_KEYS_CLUSTERED
        else:
            required = cls.REQUIRED_KEYS

        # Check required keys
        for key in required:
            if key not in raw_config:
                errors.append(f"Missing required key: '{key}'")

        # Validate types
        if 'assets' in raw_config and not isinstance(raw_config['assets'], dict):
            errors.append("'assets' must be a dictionary")

        if 'rate_assets' in raw_config and not isinstance(raw_config['rate_assets'], list):
            errors.append("'rate_assets' must be a list")

        if 'categories' in raw_config and not isinstance(raw_config['categories'], dict):
            errors.append("'categories' must be a dictionary")

        # Validate clusters structure
        if 'clusters' in raw_config:
            for name, cluster in raw_config['clusters'].items():
                if not isinstance(cluster, dict):
                    errors.append(f"Cluster '{name}' must be a dictionary")
                elif 'representative' not in cluster:
                    errors.append(f"Cluster '{name}' missing 'representative'")
                elif 'assets' not in cluster:
                    errors.append(f"Cluster '{name}' missing 'assets'")

        if errors:
            raise ConfigValidationError(errors)

    @classmethod
    def _build_config(cls, raw: dict) -> Config:
        """Build Config object from raw dictionary."""

        # Build nested configs
        analysis = cls._build_analysis_config(raw.get('analysis', {}))
        te_config = cls._build_te_config(raw.get('transfer_entropy', {}))
        thresholds = cls._build_threshold_config(raw.get('thresholds', {}))
        viz_config = cls._build_viz_config(raw.get('visualization', {}))
        output_config = cls._build_output_config(raw.get('output', {}))

        # Build clusters if present
        clusters = None
        if 'clusters' in raw:
            clusters = {}
            for name, data in raw['clusters'].items():
                clusters[name] = ClusterConfig(
                    representative=data['representative'],
                    assets=data.get('assets', {}),
                    color=data.get('color'),
                )

        # Determine assets source
        if 'core_assets' in raw:
            # Clustered mode
            assets = {}
            if clusters:
                for cluster in clusters.values():
                    assets.update(cluster.assets)
        else:
            assets = raw.get('assets', {})

        return Config(
            assets=assets,
            rate_assets=raw.get('rate_assets', []),
            categories=raw.get('categories', {}),
            category_colors=raw.get('category_colors', CATEGORY_COLORS.copy()),
            analysis=analysis,
            transfer_entropy=te_config,
            thresholds=thresholds,
            visualization=viz_config,
            output=output_config,
            core_assets=raw.get('core_assets'),
            clusters=clusters,
            key_pairs=raw.get('key_pairs'),
        )

    @classmethod
    def _build_analysis_config(cls, raw: dict) -> AnalysisConfig:
        """Build AnalysisConfig from raw dict."""
        return AnalysisConfig(
            window=raw.get('window', DEFAULT_WINDOW),
            step=raw.get('step', DEFAULT_STEP),
            rv_window=raw.get('rv_window', RV_DEFAULT_WINDOW),
            lookback=raw.get('lookback', DEFAULT_LOOKBACK),
            windows=raw.get('windows', DEFAULT_SYNC_WINDOWS.copy()),
            hub_window=raw.get('hub_window', DEFAULT_WINDOW),
            te_window_short=raw.get('te_window_short', DEFAULT_WINDOW),
            te_window_long=raw.get('te_window_long', DEFAULT_LOOKBACK),
            tail_window=raw.get('tail_window', DEFAULT_LOOKBACK),
            corr_window=raw.get('corr_window', DEFAULT_WINDOW),
            corr_change_lookback=raw.get('corr_change_lookback', 20),
        )

    @classmethod
    def _build_te_config(cls, raw: dict) -> TransferEntropyConfig:
        """Build TransferEntropyConfig from raw dict."""
        return TransferEntropyConfig(
            bins=raw.get('bins', TE_DEFAULT_BINS),
            lag=raw.get('lag', TE_DEFAULT_LAG),
            n_surrogates=raw.get('n_surrogates', TE_DEFAULT_SURROGATES),
            alpha=raw.get('alpha', TE_DEFAULT_ALPHA),
            top_n=raw.get('top_n', TE_DEFAULT_TOP_N),
        )

    @classmethod
    def _build_threshold_config(cls, raw: dict) -> ThresholdConfig:
        """Build ThresholdConfig from raw dict."""
        return ThresholdConfig(
            rv_extreme=raw.get('rv_extreme', RV_EXTREME_PERCENTILE),
            rv_elevated=raw.get('rv_elevated', RV_ELEVATED_PERCENTILE),
            sync_warning=raw.get('sync_warning', SYNC_WARNING_LEVEL),
            network_sync_danger=raw.get('network_sync_danger'),
            network_sync_warning=raw.get('network_sync_warning'),
            hub_influence_danger=raw.get('hub_influence_danger'),
            hub_influence_warning=raw.get('hub_influence_warning'),
        )

    @classmethod
    def _build_viz_config(cls, raw: dict) -> VisualizationConfig:
        """Build VisualizationConfig from raw dict."""
        figsize = raw.get('figsize', DEFAULT_FIGSIZE)
        if isinstance(figsize, list):
            figsize = tuple(figsize)

        colors = COLORS.copy()
        if 'colors' in raw:
            colors.update(raw['colors'])

        return VisualizationConfig(
            figsize=figsize,
            dpi=raw.get('dpi', DEFAULT_DPI),
            heatmap_days=raw.get('heatmap_days', DEFAULT_HEATMAP_DAYS),
            colors=colors,
        )

    @classmethod
    def _build_output_config(cls, raw: dict) -> OutputConfig:
        """Build OutputConfig from raw dict."""
        return OutputConfig(
            report_prefix=raw.get('report_prefix', DEFAULT_REPORT_PREFIX),
            dashboard_prefix=raw.get('dashboard_prefix', DEFAULT_DASHBOARD_PREFIX),
            save_indicators=raw.get('save_indicators', True),
            save_report=raw.get('save_report', True),
            save_dashboard=raw.get('save_dashboard', True),
        )

    @classmethod
    def get_default(cls) -> Config:
        """Get default configuration."""
        return Config(
            assets={
                'USDKRW': '종합 USDKRW 스팟 (~15:30)',
                'USDJPY': '이종통화 종합 JPY',
                'DXY': '달러인덱스 DOLLARS',
                'SPX': 'S&P 500',
                'NKY': '니케이 225',
                'VIX': 'CBOE VIX VOLATILITY INDEX',
                'GOLD': '금 2026-2 (연결선물)',
                'KTB_2Y': '금투협 장외거래대표수익률 국고채권(2년)',
                'KTB_10Y': '금투협 장외거래대표수익률 국고채권(10년)',
                'KTB_30Y': '금투협 장외거래대표수익률 국고채권(30년)',
                'JGB_2Y': '일본 2년',
                'JGB_10Y': '일본 10년',
                'JGB_30Y': '일본 30년',
                'UST_2Y': '미국(종합) 2년',
                'UST_10Y': '미국(종합) 10년',
                'UST_30Y': '미국(종합) 30년',
            },
            rate_assets=[
                'KTB_2Y', 'KTB_10Y', 'KTB_30Y',
                'JGB_2Y', 'JGB_10Y', 'JGB_30Y',
                'UST_2Y', 'UST_10Y', 'UST_30Y',
            ],
            categories={
                'EQ': ['SPX', 'NKY'],
                'FX': ['USDKRW', 'USDJPY', 'DXY'],
                'IR_JP': ['JGB_2Y', 'JGB_10Y', 'JGB_30Y'],
                'IR_KR': ['KTB_2Y', 'KTB_10Y', 'KTB_30Y'],
                'IR_US': ['UST_2Y', 'UST_10Y', 'UST_30Y'],
                'VOL': ['VIX'],
                'CMD': ['GOLD'],
            },
            category_colors=CATEGORY_COLORS.copy(),
        )
