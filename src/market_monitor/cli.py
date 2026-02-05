"""
Command Line Interface for Market Monitor.

Provides the main entry point for running analysis.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Market Network Monitor - Network Topology Based Stress Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python -m market_monitor run -d MARKET_WATCH.xlsx

    # With custom config and output directory
    python -m market_monitor run -d MARKET_WATCH.xlsx -c config.yaml -o ./reports

    # Verbose mode
    python -m market_monitor run -d MARKET_WATCH.xlsx -v

    # Report only (no dashboard image)
    python -m market_monitor run -d MARKET_WATCH.xlsx --report-only
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run analysis')
    run_parser.add_argument('-d', '--data', required=True, type=Path,
                          help='Path to MARKET_WATCH.xlsx')
    run_parser.add_argument('-o', '--output', default='./output', type=Path,
                          help='Output directory (default: ./output)')
    run_parser.add_argument('-c', '--config', type=Path,
                          help='Path to config.yaml')
    run_parser.add_argument('--mode', choices=['basic', 'clustered'],
                          default='clustered', help='Analysis mode (default: clustered)')
    run_parser.add_argument('--report-only', action='store_true',
                          help='Generate report only (no dashboard image)')
    run_parser.add_argument('-v', '--verbose', action='store_true',
                          help='Verbose output')
    run_parser.add_argument('-q', '--quiet', action='store_true',
                          help='Quiet mode (warnings only)')

    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate config file')
    validate_parser.add_argument('config', type=Path, help='Config file path')

    # Version command
    parser.add_argument('--version', action='store_true', help='Show version')

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"Market Monitor v{__version__}")
        return 0

    if args.command == 'run':
        return run_analysis(args)
    elif args.command == 'validate-config':
        return validate_config(args)
    else:
        parser.print_help()
        return 0


def run_analysis(args) -> int:
    """Run the main analysis pipeline."""
    import logging
    from .utils.logging import setup_logging
    from .core.config import ConfigLoader
    from .core.exceptions import MarketMonitorError
    from .data.loader import DataLoader, ClusteredDataLoader
    from .analysis.network import NetworkAnalyzer
    from .analysis.volatility import VolatilityAnalyzer
    from .report.generator import ReportGenerator
    from .report.visualizer import Visualizer

    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("  MARKET NETWORK MONITOR")
    print("=" * 70)

    try:
        # Load config
        print("\n[1/5] Loading configuration...")
        config = ConfigLoader.load_or_default(args.config)
        print(f"  Mode: {'Clustered' if config.is_clustered else 'Basic'}")

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("\n[2/5] Loading market data...")
        if args.mode == 'clustered' and config.is_clustered:
            loader = ClusteredDataLoader(args.data, config)
        else:
            loader = DataLoader(args.data, config)
        data = loader.load()

        # Get network assets
        if hasattr(loader, 'get_network_assets'):
            network_assets = loader.get_network_assets()
        else:
            network_assets = data.assets

        # Compute network metrics
        print("\n[3/5] Computing network metrics...")
        network = NetworkAnalyzer(config)
        indicators = network.compute_timeseries(data.returns, network_assets)

        window = config.analysis.window
        recent = data.returns.iloc[-window:]
        snapshot = network.compute_snapshot(recent, network_assets, window)
        print(f"  Top Hub: {snapshot.top_hub_bt} (Bt={snapshot.top3_bt[0][1]:.4f})")

        # Compute volatility
        print("\n[4/5] Computing volatility metrics...")
        vol = VolatilityAnalyzer(config)
        rv_pct = vol.compute_all(data.returns)
        print("  Done")

        # Generate outputs
        print("\n[5/5] Generating outputs...")

        date = data.returns.index[-1]
        date_str = date.strftime('%Y%m%d')

        # Report
        if config.output.save_report:
            report_gen = ReportGenerator(config)
            report = report_gen.generate(snapshot, indicators, rv_pct, date)

            report_path = output_dir / f"{config.output.report_prefix}_{date_str}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  Report: {report_path}")

            if not args.quiet:
                print("\n" + "=" * 70)
                print(report)

        # Dashboard
        if config.output.save_dashboard and not args.report_only:
            viz = Visualizer(config)
            dashboard_path = output_dir / f"{config.output.dashboard_prefix}_{date_str}.png"
            viz.create_dashboard(snapshot, indicators, rv_pct, str(dashboard_path))
            print(f"  Dashboard: {dashboard_path}")

        # Indicators
        if config.output.save_indicators:
            indicators_path = output_dir / 'indicators.pkl'
            indicators.to_pickle(indicators_path)
            print(f"  Indicators: {indicators_path}")

        print("\n" + "=" * 70)
        print("  Done!")
        print("=" * 70)

        return 0

    except MarketMonitorError as e:
        logger.error(str(e))
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


def validate_config(args) -> int:
    """Validate a config file."""
    from .core.config import ConfigLoader
    from .core.exceptions import ConfigError

    print(f"Validating: {args.config}")

    try:
        config = ConfigLoader.load(args.config)
        print(f"✓ Config is valid")
        print(f"  Assets: {len(config.assets)}")
        print(f"  Rate assets: {len(config.rate_assets)}")
        print(f"  Categories: {len(config.categories)}")
        print(f"  Mode: {'Clustered' if config.is_clustered else 'Basic'}")
        return 0
    except ConfigError as e:
        print(f"✗ Config validation failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
