#!/usr/bin/env python3
"""
=============================================================================
MARKETWATCH v1.0 - Unified Market Monitor
=============================================================================
V8 팩터 시그널이 최상단, 매크로 레이어는 시그널 해석 컨텍스트.

Pipeline:
[1] Data Load      → MARKET_WATCH.xlsx + v8 시그널
[2] Turbulence     → 개별 자산 RV percentile, 상태 판정
[3] Network        → MST 구조, Hub 식별, Multi-scale Sync
[4] Transfer Entropy → 인과 흐름 (듀얼윈도우 60d/252d)
[5] Tail Dependence → 위기 전염 감지 (excess over correlation)
[6] Pattern Match  → 유사 과거 구간 → 향후 수익률
[7] V8 Context     → 현재 레짐에서 v8 시그널 의미
[8] Sanity Check   → Ground truth 대조
[9] Output         → 리포트 + 대시보드
=============================================================================
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import MarketDataLoader
from src.data.v8_bridge import get_v8_signals, get_v8_signal_summary
from src.macro.turbulence import TurbulenceMonitor
from src.macro.network import NetworkAnalyzer
from src.macro.sync import SyncAnalyzer
from src.macro.transfer_entropy import TransferEntropyCalculator
from src.macro.tail import TailDependenceCalculator
from src.macro.impulse import ImpulseResponseAnalyzer
from src.factor.regime_context import interpret_signal_in_regime
from src.pattern.matcher import HistoricalPatternMatcher
from src.pattern.trajectory import summarize_trajectories
from src.validation.sanity_check import SanityChecker
from src.output.report import ReportGenerator
from src.output.dashboard import DashboardGenerator


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class KeyPairsAnalyzer:
    """Key pairs correlation tracking"""

    def __init__(self, config: dict):
        self.key_pairs = config.get('key_pairs', [])

    def compute_current(self, returns, window=60):
        recent = returns.iloc[-window:] if len(returns) >= window else returns
        corr = recent.corr()
        result = {}
        for pair in self.key_pairs:
            a, b = pair[0], pair[1]
            if a in corr.columns and b in corr.columns:
                result[f'{a}/{b}'] = corr.loc[a, b]
        return result

    def compute_change(self, returns, window=60, lookback=20):
        if len(returns) < window + lookback:
            return {}
        recent = returns.iloc[-window:]
        past = returns.iloc[-(window + lookback):-lookback]
        corr_now = recent.corr()
        corr_past = past.corr()

        result = {}
        for pair in self.key_pairs:
            a, b = pair[0], pair[1]
            if a in corr_now.columns and b in corr_now.columns:
                now = corr_now.loc[a, b]
                past_val = corr_past.loc[a, b]
                result[f'{a}/{b}'] = {'now': now, 'past': past_val, 'change': now - past_val}
        return result

    def compute_timeseries(self, returns, step=5):
        window = 60
        records = []
        for i in range(window, len(returns), step):
            date = returns.index[i]
            subset = returns.iloc[i - window:i]
            corr = subset.corr()
            record = {'date': date}
            for pair in self.key_pairs:
                a, b = pair[0], pair[1]
                if a in corr.columns and b in corr.columns:
                    record[f'{a}/{b}'] = corr.loc[a, b]
            records.append(record)
        import pandas as pd
        return pd.DataFrame(records).set_index('date')


def main():
    parser = argparse.ArgumentParser(description='MarketWatch v1.0')
    parser.add_argument('-d', '--data', default=None,
                        help='Path to MARKET_WATCH.xlsx')
    parser.add_argument('-c', '--config', default=None,
                        help='Path to config.yaml')
    parser.add_argument('-o', '--output', default='./output', help='Output directory')
    parser.add_argument('--no-v8', action='store_true', help='Skip V8 signal loading')
    parser.add_argument('--no-pattern', action='store_true', help='Skip pattern matching')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard generation')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')

    args = parser.parse_args()

    # Resolve default paths
    config_path = args.config or str(PROJECT_ROOT / 'config' / 'config.yaml')
    data_path = args.data

    print("=" * 70)
    print("  MARKETWATCH v1.0 - Unified Market Monitor")
    print("=" * 70)

    # [1] Load config
    print("\n[1/9] Loading configuration...")
    config = load_config(config_path)
    print(f"  Config: {config_path}")

    # Find data file
    if data_path is None:
        for candidate in [
            str(PROJECT_ROOT / 'MARKET_WATCH.xlsx'),
            str(PROJECT_ROOT.parent / 'market_monitor' / 'MARKET_WATCH.xlsx'),
            str(PROJECT_ROOT.parent / 'market-watch' / 'MARKET_WATCH.xlsx'),
        ]:
            if os.path.exists(candidate):
                data_path = candidate
                break
        if data_path is None:
            print("  ERROR: MARKET_WATCH.xlsx not found. Use -d flag.")
            return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # [1b] Load V8 signals (최우선)
    v8_signals = {}
    v8_cfg = config.get('v8', {})
    if not args.no_v8 and v8_cfg.get('enabled', True):
        print("\n[1b] Loading V8 signals...")
        v5_path = v8_cfg.get('project_path', '')
        if os.path.exists(v5_path):
            v8_signals = get_v8_signals(v5_path)
            print(get_v8_signal_summary(v8_signals))
        else:
            print(f"  WARNING: v5 path not found: {v5_path}")

    # [2] Load market data
    print("\n[2/9] Loading market data...")
    loader = MarketDataLoader(data_path, config)
    loader.load()
    network_assets = loader.get_network_assets()

    # Get analysis parameters
    analysis_cfg = config.get('analysis', {})
    hub_window = analysis_cfg.get('hub_window', 60)
    te_window_short = analysis_cfg.get('te_window_short', 60)
    te_window_long = analysis_cfg.get('te_window_long', 252)
    tail_window = analysis_cfg.get('tail_window', 252)
    corr_window = analysis_cfg.get('corr_window', 60)

    # [2b] Turbulence (L1)
    print("\n[2b] Computing turbulence...")
    turb_monitor = TurbulenceMonitor(config)
    rv_pct = turb_monitor.compute_all_rv(loader.returns)
    turb_states = turb_monitor.get_current_states(loader.returns)
    overall_pct, overall_status = turb_monitor.get_overall_status(turb_states)
    extreme_assets = turb_monitor.get_extreme_assets(turb_states)
    print(f"  Overall: {overall_status.upper()} ({overall_pct:.0f}%ile)")
    if extreme_assets:
        print(f"  Extreme: {', '.join(extreme_assets)}")

    # [3] Network (L2)
    print("\n[3/9] Computing network metrics...")
    network = NetworkAnalyzer(config)
    indicators = network.compute_timeseries(loader.returns, network_assets)
    snapshot = network.compute_snapshot(loader.returns, network_assets, hub_window)
    print(f"  Top Hub (BT): {snapshot['top_hub_bt']}, (EV): {snapshot['top_hub_ev']}")

    # Multi-scale sync
    sync_analyzer = SyncAnalyzer(config)
    sync_values = sync_analyzer.compute_current(loader.returns, network_assets)
    sync_divergence = sync_analyzer.compute_divergence(sync_values)
    if sync_divergence.get('regime_shift'):
        print(f"  !! REGIME SHIFT detected: divergence={sync_divergence['divergence']:+.4f}")

    # [4] Transfer Entropy (dual window)
    print(f"\n[4/9] Computing Transfer Entropy ({te_window_short}d + {te_window_long}d)...")
    te_calc = TransferEntropyCalculator(config)

    # Short-term
    recent_short = loader.returns.iloc[-te_window_short:]
    print(f"  [{te_window_short}d] Computing all pairs...")
    te_all_short = te_calc.compute_all_pairs(recent_short)
    te_net_flow_short = te_calc.compute_net_flow(recent_short)

    sig_count_short = sum(1 for v in te_all_short.values() if v['significant'])
    print(f"  [{te_window_short}d] Significant: {sig_count_short}/{len(te_all_short)}")

    # Long-term
    recent_long = loader.returns.iloc[-te_window_long:] if len(loader.returns) >= te_window_long else loader.returns
    print(f"  [{te_window_long}d] Computing all pairs...")
    te_all_long = te_calc.compute_all_pairs(recent_long)
    te_net_flow_long = te_calc.compute_net_flow(recent_long)

    sig_count_long = sum(1 for v in te_all_long.values() if v['significant'])
    print(f"  [{te_window_long}d] Significant: {sig_count_long}/{len(te_all_long)}")

    # For backward compatibility (dashboard TE bars)
    te_results = {k: v['te_raw'] for k, v in te_all_short.items()}

    # [5] Key Pairs + Tail Dependence
    print(f"\n[5/9] Computing correlations and tail dependence...")
    pairs_analyzer = KeyPairsAnalyzer(config)
    key_pairs = pairs_analyzer.compute_current(loader.returns, corr_window)
    pair_changes = pairs_analyzer.compute_change(loader.returns, corr_window)
    pair_ts = pairs_analyzer.compute_timeseries(loader.returns, step=config['analysis']['step'])

    # Tail dependence (long window)
    tail_calc = TailDependenceCalculator(config)
    recent_tail = loader.returns.iloc[-tail_window:] if len(loader.returns) >= tail_window else loader.returns
    tail_long = tail_calc.compute_for_pairs(recent_tail, config.get('key_pairs', []))

    high_excess = [(k, v) for k, v in tail_long.items() if v.get('excess_lower', 0) > 0.15]
    if high_excess:
        print(f"  !! Crisis contagion ({len(high_excess)}):")
        for pair, data in sorted(high_excess, key=lambda x: x[1].get('excess_lower', 0), reverse=True)[:3]:
            print(f"    {pair}: excess={data['excess_lower']:+.0%}")
    else:
        print("  No elevated contagion")

    # [5b] Impulse Response
    impulse = ImpulseResponseAnalyzer(config)
    top_hub = snapshot['top_hub_bt']
    mst_neighbors = list(snapshot['mst'].neighbors(top_hub)) if top_hub in snapshot['mst'] else []
    lead_lag = impulse.compute_lead_lag(loader.returns, top_hub, mst_neighbors)
    cond_response = impulse.compute_conditional_response(loader.returns, top_hub, mst_neighbors)

    # [6] Pattern Matching
    pattern_df = None
    pattern_summary = None
    if not args.no_pattern:
        print("\n[6/9] Running pattern matcher...")
        try:
            matcher = HistoricalPatternMatcher(loader.returns, loader.prices, window=60, step=5)
            pattern_df = matcher.find_similar_periods(top_k=5, min_gap_days=30)
            if pattern_df is not None and len(pattern_df) > 0:
                pattern_summary = summarize_trajectories(pattern_df)
        except Exception as e:
            print(f"  Pattern matching failed: {e}")
    else:
        print("\n[6/9] Pattern matching skipped")

    # [7] V8 Regime Context
    print("\n[7/9] Computing V8 regime context...")
    macro_state = {
        'turbulence_status': overall_status,
        'extreme_assets': extreme_assets,
        'sync': sync_values,
        'top_hub': snapshot['top_hub_bt'],
        'regime_shift': sync_divergence.get('regime_shift', False),
    }
    regime_context = interpret_signal_in_regime(macro_state, v8_signals) if v8_signals else {}

    for label in ['kospi', '3ybm']:
        if label in regime_context:
            ctx = regime_context[label]
            print(f"  {label.upper()}: {ctx['context']}")

    # [8] Sanity Check
    print("\n[8/9] Running sanity checks...")
    checker = SanityChecker(loader.returns)
    sanity_results = checker.run_all()
    print(f"  {checker.get_summary()}")

    # [9] Generate Outputs
    print("\n[9/9] Generating outputs...")

    date = loader.returns.index[-1]
    date_str = date.strftime('%Y%m%d')
    output_cfg = config.get('output', {})

    # Report
    if output_cfg.get('save_report', True):
        report_gen = ReportGenerator(config)
        report = report_gen.generate(
            date=date,
            v8_signals=v8_signals,
            regime_context=regime_context,
            snapshot=snapshot,
            indicators=indicators,
            rv_pct=rv_pct,
            te_sig_long=te_all_long,
            te_net_flow_long=te_net_flow_long,
            te_sig_short=te_all_short,
            te_net_flow_short=te_net_flow_short,
            key_pairs=key_pairs,
            pair_changes=pair_changes,
            tail_long=tail_long,
            core_assets=loader.core_assets,
            cluster_reps=loader.cluster_reps,
            hub_timeline=None,
            lead_lag=lead_lag,
            cond_response=cond_response,
            sanity_results=sanity_results,
            pattern_df=pattern_df,
            pattern_summary=pattern_summary,
        )

        report_path = output_dir / f"{output_cfg.get('report_prefix', 'marketwatch_report')}_{date_str}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Report: {report_path}")

        if not args.quiet:
            print("\n" + "=" * 70)
            print(report)

    # Dashboard
    if output_cfg.get('save_dashboard', True) and not args.no_dashboard:
        viz = DashboardGenerator(config)
        dashboard_path = output_dir / f"{output_cfg.get('dashboard_prefix', 'marketwatch_dashboard')}_{date_str}.png"
        viz.create_dashboard(
            snapshot=snapshot,
            indicators=indicators,
            rv_pct=rv_pct,
            pair_ts=pair_ts,
            te_results=te_results,
            core_assets=loader.core_assets,
            cluster_reps=loader.cluster_reps,
            v8_signals=v8_signals,
            regime_context=regime_context,
            output_path=str(dashboard_path),
            # Additional data for v2.0 dashboard
            tail_long=tail_long,
            te_net_flow_long=te_net_flow_long,
            turb_status=overall_status,
            turb_pct=overall_pct,
            extreme_assets=extreme_assets,
            sync_values=sync_values,
            sync_divergence=sync_divergence,
            key_pairs=key_pairs,
            pair_changes=pair_changes,
        )
        print(f"  Dashboard: {dashboard_path}")

    # Indicators
    if output_cfg.get('save_indicators', True):
        indicators_path = output_dir / 'indicators.pkl'
        indicators.to_pickle(indicators_path)
        print(f"  Indicators: {indicators_path}")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
