"""
Unified Dashboard v2.0
======================
V8 시그널 게이지 + 매크로 컨텍스트 시각화.

Layout (6 rows x 4 cols):
Row 0: V8 gauge (KOSPI) | V8 gauge (3YBM) | Environment checklist
Row 1: Factor waterfall (KOSPI + 3YBM, full width)
Row 2: MST Network        | Tail dependence bars
Row 3: Sync timeseries    | TE information flow bars
Row 4: Key pairs corr ts  | Correlation matrix
Row 5: RV percentile heatmap (full width)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import networkx as nx
from datetime import datetime

# Korean font setup
_KR_FONT = None
for _fname in ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Yu Gothic']:
    if any(f.name == _fname for f in fm.fontManager.ttflist):
        _KR_FONT = _fname
        break
if _KR_FONT:
    plt.rcParams['font.family'] = _KR_FONT
    plt.rcParams['axes.unicode_minus'] = False

# Theme display order (most market-relevant first)
_THEME_ORDER = [
    'FX_KRW', 'RATE_KR', 'LIQUIDITY', 'EQUITY_KR', 'RATE_CREDIT',
    'FX_GLOBAL', 'GLOBAL_EQUITY', 'COMMODITY', 'OTHER',
]


class DashboardGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.viz = config.get('visualization', {})
        self.colors = self.viz.get('colors', {
            'bg': '#0d1117', 'panel': '#161b22', 'text': '#e6edf3',
            'grid': '#30363d', 'danger': '#f85149', 'warning': '#d29922',
            'safe': '#3fb950', 'accent': '#58a6ff',
        })

    def create_dashboard(self, *,
                         snapshot: dict,
                         indicators: pd.DataFrame,
                         rv_pct: pd.DataFrame,
                         pair_ts: pd.DataFrame,
                         te_results: dict,
                         core_assets: list,
                         cluster_reps: list,
                         v8_signals: dict = None,
                         regime_context: dict = None,
                         output_path: str,
                         # New parameters
                         tail_long: dict = None,
                         te_net_flow_long: dict = None,
                         turb_status: str = None,
                         turb_pct: float = None,
                         extreme_assets: list = None,
                         sync_values: dict = None,
                         sync_divergence: dict = None,
                         key_pairs: dict = None,
                         pair_changes: dict = None):

        fig = plt.figure(figsize=self.viz.get('figsize', [28, 24]),
                         facecolor=self.colors['bg'])
        gs = GridSpec(6, 4, figure=fig,
                      height_ratios=[0.7, 0.8, 1.2, 1.0, 1.0, 0.7],
                      hspace=0.35, wspace=0.25)

        # ── Row 0: V8 gauges + Environment checklist ──
        ax_gauge_kospi = fig.add_subplot(gs[0, :2])
        ax_gauge_3ybm = fig.add_subplot(gs[0, 2])
        ax_env = fig.add_subplot(gs[0, 3])

        self._draw_signal_gauge(ax_gauge_kospi, v8_signals, 'kospi', regime_context)
        self._draw_signal_gauge(ax_gauge_3ybm, v8_signals, '3ybm', regime_context)
        self._draw_env_checklist(ax_env, turb_status, turb_pct, sync_values,
                                 sync_divergence, tail_long, extreme_assets, snapshot)

        # ── Row 1: Factor waterfall (full width) ──
        ax_wf_kospi = fig.add_subplot(gs[1, :2])
        ax_wf_3ybm = fig.add_subplot(gs[1, 2:])
        self._draw_factor_waterfall(ax_wf_kospi, v8_signals, 'kospi')
        self._draw_factor_waterfall(ax_wf_3ybm, v8_signals, '3ybm')

        # ── Row 2: Network + Tail dependence ──
        ax_net = fig.add_subplot(gs[2, :2])
        self._draw_network(ax_net, snapshot, core_assets, cluster_reps)

        ax_tail = fig.add_subplot(gs[2, 2:])
        self._draw_tail_bars(ax_tail, tail_long)

        # ── Row 3: Sync + TE bars ──
        ax_sync = fig.add_subplot(gs[3, :2])
        self._draw_multi_scale_sync(ax_sync, indicators)

        ax_te = fig.add_subplot(gs[3, 2:])
        self._draw_te_bars(ax_te, te_net_flow_long, te_results)

        # ── Row 4: Key pairs + Corr matrix ──
        ax_pairs = fig.add_subplot(gs[4, :2])
        self._draw_key_pairs(ax_pairs, pair_ts)

        ax_corr = fig.add_subplot(gs[4, 2:])
        self._draw_corr_matrix(ax_corr, snapshot)

        # ── Row 5: RV Heatmap (full width) ──
        ax_rv = fig.add_subplot(gs[5, :])
        self._draw_rv_heatmap(ax_rv, rv_pct, core_assets + cluster_reps)

        # Title
        date = indicators.index[-1] if len(indicators) > 0 else datetime.now()
        fig.suptitle(f'MARKETWATCH v2.0 - {date.strftime("%Y-%m-%d")}',
                     fontsize=20, fontweight='bold', color='white', y=0.995)

        plt.savefig(output_path, dpi=self.viz.get('dpi', 150),
                    facecolor=self.colors['bg'], bbox_inches='tight')
        plt.close()

    # ─────────────────────────────────────────────────────────────
    # Helper: add subtitle (한글 설명)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _add_subtitle(ax, text, y=1.0):
        """Add a small grey subtitle below the main title."""
        ax.text(0.005, y, text, transform=ax.transAxes,
                fontsize=7.5, color='#8b949e', va='top', style='italic')

    # ─────────────────────────────────────────────────────────────
    # Row 0: V8 Signal Gauge (semicircle)
    # ─────────────────────────────────────────────────────────────
    def _draw_signal_gauge(self, ax, v8_signals, label, regime_context):
        ax.set_facecolor(self.colors['panel'])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.3, 1.15)
        ax.set_aspect('equal')
        ax.axis('off')

        # Default values if no signal
        sig = None
        t1 = t2 = 0.0
        n_active = 0
        if v8_signals and label in v8_signals:
            v8 = v8_signals[label]
            sig = v8.get('signal')
            t1 = v8.get('t1_total', 0)
            t2 = v8.get('t2_total', 0)
            n_active = sum(1 for f in v8.get('factors', []) if abs(f.get('raw_signal', 0)) > 0.01)

        title = label.upper()
        ax.text(0, 1.12, title, ha='center', va='top', fontsize=14,
                fontweight='bold', color=self.colors['accent'])

        if sig is None:
            ax.text(0, 0.4, 'N/A', ha='center', va='center', fontsize=20,
                    color=self.colors['grid'])
            return

        # Draw semicircle gauge background: SHORT / NEUTRAL / LONG
        # Angles: 180° (left) to 0° (right)
        # SHORT zone: 180° to 108° (signal 0.0~0.4)
        # NEUTRAL zone: 108° to 72° (signal 0.4~0.6)
        # LONG zone: 72° to 0° (signal 0.6~1.0)
        cx, cy, r = 0, 0, 1.0

        wedge_short = Wedge((cx, cy), r, 108, 180, width=0.25,
                            facecolor=self.colors['danger'], alpha=0.3)
        wedge_neutral = Wedge((cx, cy), r, 72, 108, width=0.25,
                              facecolor=self.colors['grid'], alpha=0.4)
        wedge_long = Wedge((cx, cy), r, 0, 72, width=0.25,
                           facecolor=self.colors['safe'], alpha=0.3)

        ax.add_patch(wedge_short)
        ax.add_patch(wedge_neutral)
        ax.add_patch(wedge_long)

        # Needle: signal maps to angle (0→180°, 1→0°)
        angle_deg = 180 - sig * 180
        angle_rad = np.radians(angle_deg)
        needle_len = 0.7
        nx_pt = cx + needle_len * np.cos(angle_rad)
        ny_pt = cy + needle_len * np.sin(angle_rad)

        ax.annotate('', xy=(nx_pt, ny_pt), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))

        # Center dot
        ax.plot(cx, cy, 'o', color='white', markersize=6, zorder=5)

        # Direction text
        if sig > 0.6:
            direction = "LONG"
            dir_color = self.colors['safe']
        elif sig < 0.4:
            direction = "SHORT"
            dir_color = self.colors['danger']
        else:
            direction = "NEUTRAL"
            dir_color = self.colors['grid']

        ax.text(0, -0.08, f'{sig:.2f}', ha='center', va='top', fontsize=18,
                fontweight='bold', color='white')
        ax.text(0, -0.22, direction, ha='center', va='top', fontsize=11,
                fontweight='bold', color=dir_color)

        # Zone labels
        ax.text(-0.95, 0.15, 'SHORT', ha='center', va='center', fontsize=7,
                color=self.colors['danger'], alpha=0.7)
        ax.text(0.95, 0.15, 'LONG', ha='center', va='center', fontsize=7,
                color=self.colors['safe'], alpha=0.7)

        # T+1, T+2, Active (small text below)
        info_text = f"T+1:{t1:+.2f}  T+2:{t2:+.2f}  Active:{n_active}"
        ax.text(0, -0.35, info_text, ha='center', va='top', fontsize=8,
                color='#8b949e')

        # Near threshold / near exit warnings
        v8 = v8_signals.get(label, {}) if v8_signals else {}
        near_th = v8.get('near_threshold', [])
        near_ex = v8.get('near_exit', [])
        warn_parts = []
        if near_th:
            names = [f.get('matched_col', f.get('factor', '?'))[:12] for f in near_th[:2]]
            warn_parts.append(f"Near ON: {', '.join(names)}")
        if near_ex:
            names = [f.get('matched_col', f.get('factor', '?'))[:12] for f in near_ex[:2]]
            warn_parts.append(f"Near OFF: {', '.join(names)}")
        if warn_parts:
            ax.text(0, -0.48, ' | '.join(warn_parts), ha='center', va='top',
                    fontsize=6.5, color=self.colors['warning'])
        elif regime_context and label in regime_context:
            ctx = regime_context[label].get('context', '')
            if ctx:
                ax.text(0, -0.48, ctx[:50], ha='center', va='top', fontsize=7,
                        color='#8b949e', style='italic')

    # ─────────────────────────────────────────────────────────────
    # Row 0: Environment Checklist
    # ─────────────────────────────────────────────────────────────
    def _draw_env_checklist(self, ax, turb_status, turb_pct, sync_values,
                            sync_divergence, tail_long, extreme_assets, snapshot):
        ax.set_facecolor(self.colors['panel'])
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.97, 'MARKET ENVIRONMENT', ha='center', va='top',
                fontsize=11, fontweight='bold', color=self.colors['accent'])
        self._add_subtitle(ax, '시장 환경 체크리스트', y=0.88)

        items = []

        # 1. Volatility
        vol_status = (turb_status or 'normal').upper()
        vol_pct = turb_pct or 50
        if vol_status == 'EXTREME':
            vol_color = self.colors['danger']
        elif vol_status == 'ELEVATED':
            vol_color = self.colors['warning']
        else:
            vol_color = self.colors['safe']
        vol_detail = f"{vol_pct:.0f}%ile"
        if extreme_assets:
            vol_detail += f" ({', '.join(extreme_assets[:2])})"
        items.append(('Volatility', vol_status, vol_color, vol_detail))

        # 2. Synchronization
        sync_max = 0
        if sync_values:
            sync_max = max(sync_values.values()) if sync_values else 0
        if sync_max > 0.20:
            sync_color = self.colors['danger']
            sync_status = 'HIGH'
        elif sync_max > 0.10:
            sync_color = self.colors['warning']
            sync_status = 'MODERATE'
        else:
            sync_color = self.colors['safe']
            sync_status = 'LOW'
        sync_detail = f"max={sync_max:.3f}"
        items.append(('Synchronization', sync_status, sync_color, sync_detail))

        # 3. Crisis Contagion (tail)
        tail_max_excess = 0
        tail_max_pair = ''
        if tail_long:
            for pair, data in tail_long.items():
                exc = data.get('excess_lower', 0)
                if exc > tail_max_excess:
                    tail_max_excess = exc
                    tail_max_pair = pair
        if tail_max_excess > 0.20:
            tail_color = self.colors['danger']
            tail_status = 'CRISIS'
        elif tail_max_excess > 0.10:
            tail_color = self.colors['warning']
            tail_status = 'ELEVATED'
        else:
            tail_color = self.colors['safe']
            tail_status = 'NORMAL'
        tail_detail = f"{tail_max_pair} {tail_max_excess:+.0%}" if tail_max_pair else "Normal"
        items.append(('Crisis Contagion', tail_status, tail_color, tail_detail))

        # 4. Regime Shift
        regime_shift = False
        if sync_divergence:
            regime_shift = sync_divergence.get('regime_shift', False)
        if regime_shift:
            regime_color = self.colors['danger']
            regime_status = 'DETECTED'
            div_val = sync_divergence.get('divergence', 0)
            regime_detail = f"div={div_val:+.3f}"
        else:
            regime_color = self.colors['safe']
            regime_status = 'STABLE'
            regime_detail = ''
        items.append(('Regime Shift', regime_status, regime_color, regime_detail))

        # Draw items
        y_start = 0.75
        y_step = 0.17
        for i, (name, status, color, detail) in enumerate(items):
            y = y_start - i * y_step
            # Status dot
            ax.plot(0.08, y, 'o', color=color, markersize=10, zorder=5)
            # Label
            ax.text(0.16, y, name, va='center', fontsize=9,
                    color=self.colors['text'], fontweight='bold')
            # Status text
            ax.text(0.70, y, status, va='center', fontsize=9,
                    color=color, fontweight='bold')
            # Detail (small)
            if detail:
                ax.text(0.16, y - 0.06, detail, va='center', fontsize=7,
                        color='#8b949e')

        # Hub info at bottom
        if snapshot:
            hub = snapshot.get('top_hub_bt', '?')
            ax.text(0.5, 0.05, f'Hub: {hub}', ha='center', va='center',
                    fontsize=8, color=self.colors['warning'])

    # ─────────────────────────────────────────────────────────────
    # Row 1: Factor Waterfall (theme_summary 기반)
    # ─────────────────────────────────────────────────────────────
    def _draw_factor_waterfall(self, ax, v8_signals, label):
        ax.set_facecolor(self.colors['panel'])
        title = f'{label.upper()} Theme Contribution'
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "테마별 기여도 합산 (+롱 / -숏). 괄호 안=활성 팩터 수")

        if not v8_signals or label not in v8_signals:
            ax.axis('off')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, fontsize=14,
                    ha='center', va='center', color=self.colors['grid'])
            return

        v8 = v8_signals[label]
        theme_summary = v8.get('theme_summary', {})

        if theme_summary:
            # Use theme_summary: {theme_id: {name, total_contrib, active, potential_on, potential_off}}
            themes = []
            for tid in _THEME_ORDER:
                if tid in theme_summary:
                    t = theme_summary[tid]
                    contrib = t.get('total_contrib', 0)
                    if abs(contrib) > 0.001:
                        n_active = len(t.get('active', []))
                        n_pot_on = len(t.get('potential_on', []))
                        display = t.get('name', tid)
                        if n_pot_on > 0:
                            display += f' (+{n_pot_on})'
                        themes.append((display, contrib, n_active))
            # Also check OTHER or any not in order
            for tid, t in theme_summary.items():
                if tid not in _THEME_ORDER:
                    contrib = t.get('total_contrib', 0)
                    if abs(contrib) > 0.001:
                        n_active = len(t.get('active', []))
                        themes.append((t.get('name', tid), contrib, n_active))

            if not themes:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No active themes', transform=ax.transAxes,
                        fontsize=10, ha='center', va='center', color=self.colors['grid'])
                return

            # Sort by absolute contribution, show all
            themes.sort(key=lambda x: abs(x[1]))  # ascending so largest on top
            names = [f"{t[0]} ({t[2]})" for t in themes]
            vals = [t[1] for t in themes]
        else:
            # Fallback: group individual factors by name prefix
            factors = v8.get('factors', [])
            active = [f for f in factors if abs(f.get('weighted', 0)) > 0.005]
            active.sort(key=lambda x: abs(x.get('weighted', 0)), reverse=True)
            top = active[:7]
            if not top:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No active factors', transform=ax.transAxes,
                        fontsize=10, ha='center', va='center', color=self.colors['grid'])
                return
            names = []
            vals = []
            for f in reversed(top):
                raw_name = f.get('matched_col', f.get('factor', '?'))
                names.append(raw_name[:18])
                vals.append(f.get('weighted', 0))

        y_pos = range(len(names))
        bar_colors = [self.colors['safe'] if v > 0 else self.colors['danger'] for v in vals]

        ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, height=0.6, edgecolor='none')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9, color=self.colors['text'])
        ax.axvline(0, color=self.colors['grid'], lw=0.8)

        # Value labels
        for i, v in enumerate(vals):
            x_offset = 0.005 if v >= 0 else -0.005
            ha = 'left' if v >= 0 else 'right'
            ax.text(v + x_offset, i, f'{v:+.3f}', va='center', ha=ha,
                    fontsize=8, color=self.colors['text'])

        ax.set_xlabel('Theme Contribution', fontsize=8, color='#8b949e')
        ax.tick_params(colors=self.colors['text'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    # ─────────────────────────────────────────────────────────────
    # Row 2 left: Network
    # ─────────────────────────────────────────────────────────────
    def _draw_network(self, ax, snapshot, core_assets, cluster_reps):
        ax.set_facecolor(self.colors['panel'])
        mst = snapshot['mst']
        bt = snapshot['betweenness']

        core_size = self.viz.get('core_node_size', 800)
        cluster_size = self.viz.get('cluster_node_size', 300)

        node_colors = [self._get_node_color(n) for n in mst.nodes()]
        node_sizes = [core_size + bt.get(n, 0) * 3000 if n in core_assets
                      else cluster_size + bt.get(n, 0) * 1500 for n in mst.nodes()]

        pos = nx.kamada_kawai_layout(mst)

        edge_colors = []
        edge_widths = []
        for u, v in mst.edges():
            c = mst[u][v]['corr']
            edge_widths.append(1 + abs(c) * 4)
            edge_colors.append(self.colors['safe'] if c > 0.6 else
                               (self.colors['warning'] if c > 0.3 else self.colors['grid']))

        nx.draw_networkx_edges(mst, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(mst, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                               alpha=0.9, edgecolors='white', linewidths=2)

        labels_core = {n: n for n in mst.nodes() if n in core_assets}
        labels_cluster = {n: n for n in mst.nodes() if n in cluster_reps}

        nx.draw_networkx_labels(mst, pos, labels=labels_core, ax=ax,
                                font_size=10, font_color='white', font_weight='bold')
        nx.draw_networkx_labels(mst, pos, labels=labels_cluster, ax=ax,
                                font_size=8, font_color='#c9d1d9', font_weight='normal')

        # Mark top hubs
        for rank, (node, _) in enumerate(snapshot['top3_bt'], 1):
            if node in pos:
                x, y = pos[node]
                ax.annotate(f'B{rank}', xy=(x, y), xytext=(x + 0.08, y + 0.08),
                            fontsize=12, color=self.colors['danger'], fontweight='bold')

        ax.set_title('MARKET NETWORK (MST)', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "자산 간 MST. 큰 노드=Hub (시장 주도 자산)")
        ax.axis('off')

        # Legend
        legend_items = []
        for cat, color in self.config.get('core_colors', {}).items():
            legend_items.append(mpatches.Patch(facecolor=color, label=cat))
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            legend_items.append(mpatches.Patch(facecolor=cluster_data.get('color', '#8b949e'),
                                               label=cluster_name, alpha=0.5))
        ax.legend(handles=legend_items[:12], loc='lower left', fontsize=7, ncol=2,
                  facecolor=self.colors['panel'], labelcolor=self.colors['text'])

    # ─────────────────────────────────────────────────────────────
    # Row 2 right: Tail Dependence Bars (NEW)
    # ─────────────────────────────────────────────────────────────
    def _draw_tail_bars(self, ax, tail_long):
        ax.set_facecolor(self.colors['panel'])
        ax.set_title('CRISIS CONTAGION', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "상관관계 대비 초과 동반하락 (Excess Tail Dependence)")

        if not tail_long:
            ax.axis('off')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, fontsize=14,
                    ha='center', va='center', color=self.colors['grid'])
            return

        # Sort by excess_lower descending
        sorted_items = sorted(tail_long.items(),
                              key=lambda x: x[1].get('excess_lower', 0), reverse=True)

        pairs = []
        excess_vals = []
        for pair, data in sorted_items:
            pairs.append(pair)
            excess_vals.append(data.get('excess_lower', 0))

        y_pos = range(len(pairs))
        bar_colors = []
        for v in excess_vals:
            if v > 0.20:
                bar_colors.append(self.colors['danger'])
            elif v > 0.10:
                bar_colors.append(self.colors['warning'])
            elif v < -0.10:
                bar_colors.append(self.colors['accent'])
            else:
                bar_colors.append(self.colors['grid'])

        ax.barh(y_pos, [v * 100 for v in excess_vals],
                color=bar_colors, alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=8, color=self.colors['text'])
        ax.axvline(0, color=self.colors['grid'], lw=0.8)
        ax.axvline(20, color=self.colors['danger'], lw=0.8, ls='--', alpha=0.5)
        ax.axvline(10, color=self.colors['warning'], lw=0.8, ls='--', alpha=0.5)

        # Value labels
        for i, v in enumerate(excess_vals):
            pct = v * 100
            x_offset = 1 if pct >= 0 else -1
            ha = 'left' if pct >= 0 else 'right'
            label = f'{pct:+.0f}%'
            status = tail_long[sorted_items[i][0]].get('interpretation', '')
            if 'CRISIS' in status:
                label += ' CRISIS'
            ax.text(pct + x_offset, i, label, va='center', ha=ha,
                    fontsize=7.5, color=self.colors['text'])

        ax.set_xlabel('Excess (%)', fontsize=8, color='#8b949e')
        ax.tick_params(colors=self.colors['text'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    # ─────────────────────────────────────────────────────────────
    # Row 3 left: Multi-scale Sync
    # ─────────────────────────────────────────────────────────────
    def _draw_multi_scale_sync(self, ax, indicators):
        ax.set_facecolor(self.colors['panel'])
        windows = self.config.get('analysis', {}).get('windows', [5, 20, 60])
        colors_list = [self.colors['danger'], self.colors['warning'], self.colors['safe']]

        for i, w in enumerate(windows):
            col = f'sync_{w}d'
            if col in indicators.columns:
                ax.plot(indicators.index, indicators[col],
                        color=colors_list[i % len(colors_list)], lw=1.5, label=f'{w}d')

        ax.axhline(0.20, color=self.colors['danger'], ls='--', lw=1, alpha=0.7,
                   label='Warning (0.2)')
        ax.legend(loc='upper left', fontsize=8, facecolor=self.colors['panel'],
                  labelcolor=self.colors['text'])
        ax.set_ylabel('Network Sync', color=self.colors['text'])
        ax.set_title('SYNCHRONIZATION', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "동조화 상승 = 분산투자 효과 감소. 점선(0.2) 초과 시 경고")
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    # ─────────────────────────────────────────────────────────────
    # Row 3 right: TE bars (improved: net flow with direction)
    # ─────────────────────────────────────────────────────────────
    def _draw_te_bars(self, ax, te_net_flow_long, te_results_fallback):
        ax.set_facecolor(self.colors['panel'])
        ax.set_title('INFORMATION FLOW', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "A\u2192B: A가 B에 선행하는 인과 흐름 (Net Z-score)")

        # Prefer net flow data (has direction)
        if te_net_flow_long:
            top_flows = sorted(te_net_flow_long.items(),
                               key=lambda x: x[1]['flow_strength'], reverse=True)[:10]
            labels = []
            vals = []
            for pair, data in top_flows:
                direction = data.get('dominant_direction', pair)
                net_z = data.get('net_flow_z', 0)
                labels.append(direction)
                vals.append(net_z)

            y_pos = range(len(labels))
            bar_colors = []
            for v in vals:
                if abs(v) > 2.5:
                    bar_colors.append(self.colors['danger'])
                elif abs(v) > 1.5:
                    bar_colors.append(self.colors['warning'])
                else:
                    bar_colors.append(self.colors['accent'])

            ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7.5, color=self.colors['text'])
            ax.axvline(0, color=self.colors['grid'], lw=0.8)
            ax.axvline(2.5, color=self.colors['danger'], lw=0.8, ls='--', alpha=0.4)
            ax.axvline(-2.5, color=self.colors['danger'], lw=0.8, ls='--', alpha=0.4)

            # Value labels
            for i, v in enumerate(vals):
                x_offset = 0.1 if v >= 0 else -0.1
                ha = 'left' if v >= 0 else 'right'
                sig = ''
                pair_key = top_flows[i][0]
                if top_flows[i][1].get('both_significant'):
                    sig = '**'
                elif top_flows[i][1].get('any_significant'):
                    sig = '*'
                ax.text(v + x_offset, i, f'Z={v:.1f}{sig}', va='center', ha=ha,
                        fontsize=7, color=self.colors['text'])

            ax.set_xlabel('Net Z-score', fontsize=8, color='#8b949e')

        elif te_results_fallback:
            # Fallback to old raw TE
            sorted_te = sorted(te_results_fallback.items(), key=lambda x: x[1], reverse=True)[:10]
            pairs = [p[0] for p in sorted_te]
            vals = [p[1] for p in sorted_te]
            colors = [self.colors['danger'] if v > 0.15 else self.colors['accent'] for v in vals]
            y_pos = range(len(pairs))
            ax.barh(y_pos, vals, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pairs, fontsize=7, color=self.colors['text'])
            ax.set_xlabel('Transfer Entropy', color=self.colors['text'])
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, fontsize=14,
                    ha='center', va='center', color=self.colors['grid'])
            return

        ax.tick_params(colors=self.colors['text'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    # ─────────────────────────────────────────────────────────────
    # Row 4 left: Key Pairs
    # ─────────────────────────────────────────────────────────────
    def _draw_key_pairs(self, ax, pair_ts):
        ax.set_facecolor(self.colors['panel'])
        key_pairs_to_show = ['USDJPY/NKY', 'USDKRW/USDJPY', 'JGB_10Y/KTB_10Y',
                             'UST_10Y/JGB_10Y', 'Bund_10Y/UST_10Y']
        colors_list = [self.colors['danger'], self.colors['accent'], self.colors['warning'],
                       self.colors['safe'], '#8b5cf6']

        for i, pair in enumerate(key_pairs_to_show):
            if pair in pair_ts.columns:
                ax.plot(pair_ts.index, pair_ts[pair],
                        color=colors_list[i % len(colors_list)], lw=1.5, label=pair)

        ax.axhline(0, color=self.colors['grid'], ls='-', lw=0.5)
        ax.legend(loc='upper left', fontsize=7, facecolor=self.colors['panel'],
                  labelcolor=self.colors['text'], ncol=2)
        ax.set_ylabel('Correlation', color=self.colors['text'])
        ax.set_title('KEY PAIR CORRELATIONS', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "60일 롤링 상관. 급변 = 시장 구조 변화 신호")
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    # ─────────────────────────────────────────────────────────────
    # Row 4 right: Correlation Matrix
    # ─────────────────────────────────────────────────────────────
    def _draw_corr_matrix(self, ax, snapshot):
        ax.set_facecolor(self.colors['panel'])
        corr = snapshot['corr']
        cmap = LinearSegmentedColormap.from_list('corr',
                                                  ['#f85149', '#161b22', '#3fb950'], N=100)
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=6, color=self.colors['text'])
        ax.set_yticklabels(corr.columns, fontsize=6, color=self.colors['text'])
        ax.set_title('CORRELATION MATRIX', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "자산 간 상관관계 매트릭스")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.colors['text'])

    # ─────────────────────────────────────────────────────────────
    # Row 5: RV Heatmap
    # ─────────────────────────────────────────────────────────────
    def _draw_rv_heatmap(self, ax, rv_pct, network_assets):
        ax.set_facecolor(self.colors['panel'])
        days = self.viz.get('heatmap_days', 90)

        cols = [c for c in network_assets if c in rv_pct.columns]
        rv_recent = rv_pct[cols].iloc[-days:]

        cmap = LinearSegmentedColormap.from_list('stress',
                                                  ['#238636', '#3fb950', '#d29922', '#f85149', '#da3633'], N=100)
        im = ax.imshow(rv_recent.T.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)

        ax.set_yticks(range(len(rv_recent.columns)))
        ax.set_yticklabels(rv_recent.columns, color=self.colors['text'], fontsize=8)

        n = len(rv_recent)
        ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        labels = [rv_recent.index[i].strftime('%m/%d') for i in ticks if i < len(rv_recent)]
        ax.set_xticks(ticks[:len(labels)])
        ax.set_xticklabels(labels, color=self.colors['text'])

        ax.set_title(f'VOLATILITY HEATMAP ({days}d)', fontsize=11, fontweight='bold',
                     color=self.colors['text'], loc='left')
        self._add_subtitle(ax, "자산별 RV 백분위. 빨강 = 극단적 변동성 수준")
        ax.tick_params(colors=self.colors['text'])

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, orientation='horizontal', pad=0.15)
        cbar.set_label('Percentile', color=self.colors['text'])
        cbar.ax.xaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=self.colors['text'])

    # ─────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────
    def _get_node_color(self, asset: str) -> str:
        for cat, assets in self.config.get('core_categories', {}).items():
            if asset in assets:
                return self.config.get('core_colors', {}).get(cat, '#8b949e')
        for cluster_name, cluster_data in self.config.get('clusters', {}).items():
            if asset in cluster_data.get('assets', {}) or asset == cluster_data.get('representative'):
                return cluster_data.get('color', '#8b949e')
        return '#8b949e'
