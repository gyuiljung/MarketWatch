"""
Unified Report Generator
========================
V8 시그널 + 매크로 컨텍스트 통합 리포트.
V8이 최상단, 매크로 레이어는 컨텍스트로 하단 배치.
"""
from datetime import datetime


class ReportGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        analysis = config.get('analysis', {})
        self.te_short = analysis.get('te_window_short', 60)
        self.te_long = analysis.get('te_window_long', 252)
        self.tail_window = analysis.get('tail_window', 252)

    def generate(self, *,
                 date: datetime,
                 v8_signals: dict = None,
                 regime_context: dict = None,
                 snapshot: dict = None,
                 indicators=None,
                 rv_pct=None,
                 te_sig_long: dict = None,
                 te_net_flow_long: dict = None,
                 te_sig_short: dict = None,
                 te_net_flow_short: dict = None,
                 key_pairs: dict = None,
                 pair_changes: dict = None,
                 tail_long: dict = None,
                 core_assets: list = None,
                 cluster_reps: list = None,
                 hub_timeline=None,
                 lead_lag: dict = None,
                 cond_response: dict = None,
                 sanity_results: list = None,
                 pattern_df=None,
                 pattern_summary: str = None) -> str:

        windows = self.config.get('analysis', {}).get('windows', [5, 20, 60])

        report = f"""
{'='*82}
  MARKETWATCH v1.0 - UNIFIED MONITOR
  {date.strftime('%Y-%m-%d %H:%M')}
{'='*82}

"""
        # =====================================================================
        # Section 1: V8 FACTOR SIGNALS (최상단)
        # =====================================================================
        report += self._section_v8(v8_signals, regime_context)

        # =====================================================================
        # Section 2: MACRO CONTEXT (V8 해석을 위한 배경)
        # =====================================================================
        report += self._section_sync(indicators, windows)
        report += self._section_hubs(snapshot, core_assets or [], cluster_reps or [])
        report += self._section_te(te_sig_long, te_net_flow_long, te_sig_short, te_net_flow_short)
        report += self._section_key_pairs(key_pairs, pair_changes)
        report += self._section_tail(tail_long)
        report += self._section_volatility(rv_pct)

        # =====================================================================
        # Section 3: PATTERN ANALYSIS
        # =====================================================================
        if pattern_df is not None and len(pattern_df) > 0:
            report += self._section_pattern(pattern_df, pattern_summary)

        # =====================================================================
        # Section 4: SANITY CHECK
        # =====================================================================
        if sanity_results:
            report += self._section_sanity(sanity_results)

        # =====================================================================
        # Section 5: ALERTS
        # =====================================================================
        report += self._section_alerts(snapshot, indicators, rv_pct, te_sig_long,
                                       te_net_flow_long, tail_long, windows)

        report += "\n" + "=" * 82 + "\n"
        return report

    def _section_v8(self, v8_signals, regime_context):
        s = f"""
{'━'*82}
V8 FACTOR SIGNALS
{'━'*82}

  ※ V8 팩터 앙상블 시그널. 0=숏, 0.5=중립, 1=롱.
     T+1: 한국 팩터 기여도 합, T+2: 해외 팩터 기여도 합.
     Active: 현재 진입/청산 임계값을 넘어 활성화된 팩터 수.

"""
        if not v8_signals:
            s += "  V8 signals not available\n"
            return s

        for label in ['kospi', '3ybm']:
            v8 = v8_signals.get(label, {})
            sig = v8.get('signal')
            if sig is None:
                s += f"  {label.upper()}: N/A\n"
                continue

            direction = "LONG" if sig > 0.6 else ("SHORT" if sig < 0.4 else "NEUTRAL")
            bar = self._signal_bar(sig)
            t1 = v8.get('t1_total', 0)
            t2 = v8.get('t2_total', 0)
            n_active = sum(1 for f in v8.get('factors', []) if abs(f.get('raw_signal', 0)) > 0.01)

            s += f"  {label.upper()}: {bar} {sig:.2f} ({direction})\n"
            s += f"    T+1 (한국): {t1:+.3f}  |  T+2 (해외): {t2:+.3f}  |  Active: {n_active}\n"

            # Theme-based breakdown
            theme_summary = v8.get('theme_summary', {})
            if theme_summary:
                s += f"\n    {'테마':<14} {'기여도':>8}  {'활성':>4}  {'진입대기':>8}  {'청산근접':>8}\n"
                s += f"    {'─'*50}\n"
                sorted_themes = sorted(theme_summary.items(),
                                       key=lambda x: abs(x[1].get('total_contrib', 0)), reverse=True)
                for tid, t in sorted_themes:
                    contrib = t.get('total_contrib', 0)
                    n_act = len(t.get('active', []))
                    n_pot_on = len(t.get('potential_on', []))
                    n_pot_off = len(t.get('potential_off', []))
                    if abs(contrib) < 0.001 and n_pot_on == 0 and n_pot_off == 0:
                        continue
                    tname = t.get('name', tid)
                    pot_on_str = f"+{n_pot_on}" if n_pot_on > 0 else "-"
                    pot_off_str = f"{n_pot_off}" if n_pot_off > 0 else "-"
                    s += f"    {tname:<14} {contrib:+8.3f}  {n_act:>4}  {pot_on_str:>8}  {pot_off_str:>8}\n"
            else:
                # Fallback: individual top factors
                active = [f for f in v8.get('factors', []) if abs(f.get('weighted', 0)) > 0.01]
                active.sort(key=lambda x: abs(x.get('weighted', 0)), reverse=True)
                if active:
                    top_factors = active[:5]
                    factor_strs = []
                    for f in top_factors:
                        name = f.get('matched_col', f.get('factor', '?'))
                        if len(name) > 20:
                            name = name[:20]
                        factor_strs.append(f"{name}({f['weighted']:+.2f})")
                    s += f"    Top: {', '.join(factor_strs)}\n"

            # Near threshold / near exit
            near_th = v8.get('near_threshold', [])
            near_ex = v8.get('near_exit', [])
            if near_th or near_ex:
                s += "\n"
            if near_th:
                s += f"    [진입 근접] ±2% 이내\n"
                for f in near_th[:4]:
                    name = f.get('matched_col', f.get('factor', '?'))
                    if len(name) > 24:
                        name = name[:24]
                    dist = f.get('distance_pct', 0)
                    pot = f.get('potential_contrib', 0)
                    s += f"      {name:<24} 거리:{dist:+.1f}%  예상기여:{pot:+.3f}\n"
            if near_ex:
                s += f"    [청산 근접] ±2% 이내\n"
                for f in near_ex[:4]:
                    name = f.get('matched_col', f.get('factor', '?'))
                    if len(name) > 24:
                        name = name[:24]
                    dist = f.get('distance_pct', 0)
                    cur = f.get('current_contrib', 0)
                    s += f"      {name:<24} 거리:{dist:+.1f}%  현재기여:{cur:+.3f}\n"

            # Regime context
            if regime_context and label in regime_context:
                ctx = regime_context[label]
                s += f"    Context: {ctx['context']}\n"
                for detail in ctx.get('details', [])[:3]:
                    s += f"      - {detail}\n"

            s += "\n"

        return s

    def _section_sync(self, indicators, windows):
        s = f"""
{'━'*82}
MULTI-SCALE SYNCHRONIZATION
{'━'*82}

  ※ 자산 간 수익률 동조화. 높을수록 시장이 한 방향으로 움직임.
     분산투자 효과가 줄어들며, 0.2 초과 시 경고.
     Divergence: 단기-장기 차이. 양(+)이면 단기 동조화 급등 → 레짐 전환 신호.

"""
        if indicators is None or len(indicators) == 0:
            return s + "  No data\n"

        current = indicators.iloc[-1]
        for w in windows:
            sync_key = f'sync_{w}d'
            pct_key = f'{sync_key}_pct'
            sync_val = current.get(sync_key, 0)
            pct_val = current.get(pct_key, 50)
            status = " !! HIGH" if sync_val > self.thresholds.get('sync_warning', 0.20) else ""
            s += f"  {w:3d}d Sync:  {sync_val:+.4f}  ({pct_val:.0f}%ile){status}\n"

        if len(windows) >= 2:
            short_sync = current.get(f'sync_{windows[0]}d', 0)
            long_sync = current.get(f'sync_{windows[-1]}d', 0)
            divergence = short_sync - long_sync
            s += f"\n  Divergence ({windows[0]}d - {windows[-1]}d): {divergence:+.4f}"
            if divergence > 0.08:
                s += " !! REGIME SHIFT"
            s += "\n"

        return s

    def _section_hubs(self, snapshot, core_assets, cluster_reps):
        if not snapshot:
            return ""
        s = f"""
{'━'*82}
TOP HUBS
{'━'*82}

  ※ MST(최소신장트리)에서 가장 연결이 많은 자산 = 시장 주도 자산.
     Betweenness: 정보 흐름의 병목점. Eigenvector: 중요한 자산과 연결된 자산.
     Hub 변화는 시장 주도 테마 변화를 의미.

  BETWEENNESS (topology)          EIGENVECTOR (importance)
"""
        for i, ((bt_node, bt_val), (ev_node, ev_val)) in enumerate(
                zip(snapshot['top3_bt'], snapshot['top3_ev']), 1):
            bt_type = "CORE" if bt_node in core_assets else "CLUS"
            ev_type = "CORE" if ev_node in core_assets else "CLUS"
            s += f"  #{i}  {bt_node:<12} [{bt_type}] {bt_val:.4f}   #{i}  {ev_node:<12} [{ev_type}] {ev_val:.4f}\n"
        return s

    def _section_te(self, te_sig_long, te_net_flow_long, te_sig_short, te_net_flow_short):
        if not te_net_flow_long:
            return ""

        s = f"""
{'━'*82}
TRANSFER ENTROPY - Top 10 Net Flows ({self.te_long}d structural)
{'━'*82}

  ※ 자산 간 정보 흐름의 방향과 강도. A→B: A의 변동이 B를 선행.
     Net Z: 순방향 흐름 강도 (Z-score). |Z|>2.5이면 통계적으로 유의미.
     Lag: 최적 시차 (A→B/B→A). REGIME CHANGE: 60일 vs 252일 흐름 차이.

  Rank  Pair              Direction             Net Z    Lag   Sig
  {'─'*65}
"""
        top_flows = sorted(te_net_flow_long.items(), key=lambda x: x[1]['flow_strength'], reverse=True)[:10]
        for rank, (pair, data) in enumerate(top_flows, 1):
            sig_str = "**" if data['both_significant'] else ("*" if data['any_significant'] else "")
            lag_ab = data.get('best_lag_ab', '?')
            lag_ba = data.get('best_lag_ba', '?')
            s += f"  {rank:>2}.  {pair:<16} {data['dominant_direction']:<20} {data['net_flow_z']:+5.2f}   {lag_ab}/{lag_ba}d {sig_str}\n"

        if te_sig_long:
            sig_long = sum(1 for v in te_sig_long.values() if v['significant'])
            s += f"\n  Total significant: {sig_long}/{len(te_sig_long)}\n"

        # Z-score delta (regime change detection)
        if te_net_flow_short and te_net_flow_long:
            s += f"\n  [REGIME CHANGE] Z-delta ({self.te_short}d - {self.te_long}d)\n"
            s += f"  {'─'*60}\n"

            z_deltas = []
            for pair, data_short in te_net_flow_short.items():
                if pair in te_net_flow_long:
                    data_long = te_net_flow_long[pair]
                    delta = data_short['net_flow_z'] - data_long['net_flow_z']
                    z_deltas.append({
                        'pair': pair, 'delta': delta,
                        'direction_change': data_short['dominant_direction'] != data_long['dominant_direction'],
                    })

            z_deltas.sort(key=lambda x: abs(x['delta']), reverse=True)
            for item in z_deltas[:5]:
                flag = " !! FLIP" if item['direction_change'] else ""
                s += f"  {item['pair']:<18} Delta={item['delta']:+5.2f}{flag}\n"

        return s

    def _section_key_pairs(self, key_pairs, pair_changes):
        if not key_pairs:
            return ""
        s = f"""
{'━'*82}
KEY CORRELATION PAIRS
{'━'*82}

  ※ 핵심 자산쌍의 60일 롤링 상관관계.
     D20d: 20일 전 대비 변화량. 급변 시 시장 구조 변화 의미.

"""
        for pair_name, val in key_pairs.items():
            change = pair_changes.get(pair_name, {}).get('change', 0) if pair_changes else 0
            s += f"  {pair_name:<20} {val:+.3f}  (D20d: {change:+.3f})\n"
        return s

    def _section_tail(self, tail_long):
        if not tail_long:
            return ""
        s = f"""
{'━'*82}
TAIL DEPENDENCE - EXCESS ({self.tail_window}d, 10% tails)
{'━'*82}

  ※ 극단적 하락 시 동반 하락 확률. 상관관계만으로 설명 안 되는 초과분(Excess).
     CRISIS CONTAGION: 위기 시 예상보다 훨씬 강하게 동반 하락.
     Tail diversified: 위기 시 오히려 분산 효과 작동 (예: SPX/VIX).

  Pair                Corr    Tail    Expected  Excess   Status
  {'─'*62}
"""
        for pair, data in sorted(tail_long.items(), key=lambda x: x[1].get('excess_lower', 0), reverse=True):
            corr = data.get('correlation', 0)
            tail = data.get('lower_10', 0)
            expected = data.get('expected', 0)
            excess = data.get('excess_lower', 0)
            status = data.get('interpretation', '')
            s += f"  {pair:<18} {corr:+.2f}   {tail:5.1%}   {expected:5.1%}   {excess:+5.1%}   {status}\n"
        return s

    def _section_volatility(self, rv_pct):
        if rv_pct is None or len(rv_pct) == 0:
            return ""
        s = f"""
{'━'*82}
VOLATILITY (Top 10)
{'━'*82}

  ※ 5일 실현변동성의 252일(1년) 백분위.
     EXTREME(>90%ile): 최근 1년 중 상위 10% 변동성.
     ELEVATED(>75%ile): 평소보다 높은 변동성 구간.

"""
        rv_current = rv_pct.iloc[-1]
        rv_sorted = rv_current.sort_values(ascending=False)
        for asset, pct in rv_sorted.head(10).items():
            status = " !! EXTREME" if pct > 90 else (" ! ELEVATED" if pct > 75 else "")
            s += f"  {asset:<12}: {pct:5.1f}%{status}\n"
        return s

    def _section_pattern(self, pattern_df, pattern_summary):
        s = f"""
{'━'*82}
HISTORICAL PATTERN ANALYSIS (Top {len(pattern_df)})
{'━'*82}

  ※ 현재 시장 상태(동조화, Hub, 변동성 조합)와 가장 유사한 과거 구간.
     유사도(sim): 1에 가까울수록 현재와 비슷. 이후 수익률은 참고용.

"""
        for i, row in pattern_df.iterrows():
            s += f"  #{i+1} {row['date'].strftime('%Y-%m-%d')} (sim: {row['similarity']:.3f}) Hub: {row['hub']}\n"
            for horizon in ['5d', '20d']:
                parts = []
                for asset in ['SPX', 'NKY', 'KOSPI', 'USDJPY']:
                    col = f'{horizon}_{asset}'
                    if col in row and not pd.isna(row[col]):
                        parts.append(f"{asset}:{row[col]:+.1f}%")
                if parts:
                    s += f"     {horizon}: {', '.join(parts)}\n"

        if pattern_summary:
            s += f"\n{pattern_summary}\n"

        return s

    def _section_sanity(self, sanity_results):
        passed = sum(1 for r in sanity_results if r.passed)
        total = len(sanity_results)
        s = f"""
{'━'*82}
SANITY CHECK ({passed}/{total} passed)
{'━'*82}

"""
        for r in sanity_results:
            mark = "[PASS]" if r.passed else "[FAIL]"
            s += f"  {mark} {r.name}: {r.actual} ({r.details})\n"
        return s

    def _section_alerts(self, snapshot, indicators, rv_pct, te_sig_long,
                        te_net_flow_long, tail_long, windows):
        alerts = []

        if indicators is not None and len(indicators) > 0:
            current = indicators.iloc[-1]
            if len(windows) >= 2:
                short_sync = current.get(f'sync_{windows[0]}d', 0)
                long_sync = current.get(f'sync_{windows[-1]}d', 0)
                if short_sync - long_sync > 0.08:
                    alerts.append("!! REGIME SHIFT: Short-term sync diverging")

        if te_sig_long:
            for pair, data in te_sig_long.items():
                if data['significant'] and data['te_z'] > 2.5:
                    alerts.append(f"!! STRONG TE [{self.te_long}d]: {pair} (Z={data['te_z']:.2f})")

        if tail_long:
            for pair, data in tail_long.items():
                excess = data.get('excess_lower', 0)
                if excess > 0.20:
                    alerts.append(f"!! CRISIS CONTAGION: {pair} excess={excess:+.0%}")

        if snapshot and snapshot['top_hub_bt'] != snapshot['top_hub_ev']:
            alerts.append(f"!! HUB DIVERGENCE: BT={snapshot['top_hub_bt']}, EV={snapshot['top_hub_ev']}")

        if rv_pct is not None and len(rv_pct) > 0:
            rv_current = rv_pct.iloc[-1]
            extreme = [a for a, v in rv_current.items() if v > self.thresholds.get('rv_extreme', 90)]
            if extreme:
                alerts.append(f"!! EXTREME VOL: {', '.join(extreme[:5])}")

        s = f"""
{'━'*82}
ALERTS
{'━'*82}

  ※ 주의가 필요한 항목 자동 탐지 결과.
     STRONG TE: Z>2.5인 유의미한 인과 흐름.
     CRISIS CONTAGION: 꼬리의존성 초과 20% 이상.
     REGIME SHIFT: 단기-장기 동조화 괴리.

"""
        if alerts:
            for a in alerts[:10]:
                s += f"  {a}\n"
        else:
            s += "  No critical alerts\n"

        return s

    @staticmethod
    def _signal_bar(signal: float, width: int = 20) -> str:
        filled = int(signal * width)
        return '[' + '#' * filled + '.' * (width - filled) + ']'


# Needed for pattern section
import pandas as pd
