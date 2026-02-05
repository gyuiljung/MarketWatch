"""
Visualization module.

Generates dashboard images with network graphs and metrics.
"""

from typing import Dict, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from ..core.config import Config
from ..core.constants import COLORS, CATEGORY_COLORS
from ..analysis.network import NetworkSnapshot

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Dashboard visualizer.

    Creates multi-panel dashboards with:
    - Network graph
    - Metrics panel
    - Hub history
    - Hub influence
    - Category stack
    - Network sync
    - RV heatmap
    """

    def __init__(self, config: Config):
        """
        Initialize visualizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.viz = config.visualization
        self.colors = self.viz.colors
        self.cat_colors = config.category_colors or CATEGORY_COLORS

    def create_dashboard(
        self,
        snapshot: NetworkSnapshot,
        indicators: pd.DataFrame,
        rv_pct: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Create and save dashboard image.

        Args:
            snapshot: Network snapshot
            indicators: Time series indicators
            rv_pct: RV percentile DataFrame
            output_path: Output file path
        """
        logger.info(f"Creating dashboard: {output_path}")

        fig = plt.figure(figsize=self.viz.figsize, facecolor=self.colors['bg'])
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.3, 1, 1], hspace=0.3, wspace=0.25)

        # Panel 1: Network
        ax1 = fig.add_subplot(gs[0, :2])
        self._draw_network(ax1, snapshot)

        # Panel 2: Metrics
        ax2 = fig.add_subplot(gs[0, 2])
        self._draw_metrics(ax2, snapshot, indicators)

        # Panel 3: Hub History
        ax3 = fig.add_subplot(gs[1, 0])
        self._draw_hub_history(ax3, indicators)

        # Panel 4: Hub Influence
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_hub_influence(ax4, indicators)

        # Panel 5: Category Stack
        ax5 = fig.add_subplot(gs[1, 2])
        self._draw_category_stack(ax5, indicators)

        # Panel 6: Network Sync
        ax6 = fig.add_subplot(gs[2, 0])
        self._draw_network_sync(ax6, indicators)

        # Panel 7: RV Heatmap
        ax7 = fig.add_subplot(gs[2, 1:])
        self._draw_rv_heatmap(ax7, rv_pct)

        # Title
        date = indicators.index[-1] if len(indicators) > 0 else pd.Timestamp.now()
        fig.suptitle(
            f'NETWORK MONITOR - {date.strftime("%Y-%m-%d")}',
            fontsize=20, fontweight='bold', color='white', y=0.995
        )

        plt.savefig(
            output_path,
            dpi=self.viz.dpi,
            facecolor=self.colors['bg'],
            bbox_inches='tight'
        )
        plt.close()

        logger.info(f"Dashboard saved: {output_path}")

    def _get_category(self, asset: str) -> str:
        """Get category for asset."""
        for cat, assets in self.config.categories.items():
            if asset in assets:
                return cat
        return 'OTHER'

    def _draw_network(self, ax, snapshot: NetworkSnapshot) -> None:
        """Draw network graph panel."""
        ax.set_facecolor(self.colors['panel'])

        mst = snapshot.mst
        bt = snapshot.betweenness

        node_colors = [self.cat_colors.get(self._get_category(n), '#8b949e') for n in mst.nodes()]
        node_sizes = [400 + bt[n] * 6000 for n in mst.nodes()]

        pos = nx.kamada_kawai_layout(mst)

        edge_colors = []
        edge_widths = []
        for u, v in mst.edges():
            c = mst[u][v].get('corr', 0)
            edge_widths.append(1 + abs(c) * 4)
            edge_colors.append(
                self.colors['safe'] if c > 0.6 else
                (self.colors['warning'] if c > 0.3 else self.colors['grid'])
            )

        nx.draw_networkx_edges(mst, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(mst, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                              alpha=0.9, edgecolors='white', linewidths=2)
        nx.draw_networkx_labels(mst, pos, ax=ax, font_size=9, font_color='white', font_weight='bold')

        # Mark top 3
        for rank, (node, _) in enumerate(snapshot.top3_bt, 1):
            if node in pos:
                x, y = pos[node]
                ax.annotate(f'#{rank}', xy=(x, y), xytext=(x+0.08, y+0.08),
                           fontsize=14, color=self.colors['danger'], fontweight='bold')

        ax.set_title(f"CURRENT NETWORK (Top Hub: {snapshot.top_hub_bt})",
                    fontsize=12, fontweight='bold', color=self.colors['text'])
        ax.axis('off')

        # Legend
        legend = [mpatches.Patch(facecolor=c, label=cat) for cat, c in self.cat_colors.items()]
        ax.legend(handles=legend, loc='lower left', fontsize=8,
                 facecolor=self.colors['panel'], labelcolor=self.colors['text'])

    def _draw_metrics(self, ax, snapshot: NetworkSnapshot, indicators: pd.DataFrame) -> None:
        """Draw metrics panel."""
        ax.set_facecolor(self.colors['panel'])
        ax.axis('off')

        current = indicators.iloc[-1] if len(indicators) > 0 else {}

        text = f"""
TOP 3 HUBS
{'─' * 30}
"""
        for i, (node, val) in enumerate(snapshot.top3_bt[:3], 1):
            text += f"#{i}  {node:<10} {val:.4f}\n"

        text += f"""
METRICS
{'─' * 30}
Hub Influence: {snapshot.hub_influence:.4f}
               ({current.get('hub_influence_pct', 0):.0f}%ile)
Network Sync:  {snapshot.network_sync:.4f}
"""

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               color=self.colors['text'], va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.5))

    def _draw_hub_history(self, ax, indicators: pd.DataFrame) -> None:
        """Draw hub history panel."""
        ax.set_facecolor(self.colors['panel'])

        if 'top_hub_bt' not in indicators.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color=self.colors['text'])
            return

        # Color by category
        colors = []
        for h in indicators['top_hub_bt']:
            cat = self._get_category(h) if pd.notna(h) else 'OTHER'
            colors.append(self.cat_colors.get(cat, 'gray'))

        if 'hub_influence' in indicators.columns:
            ax.scatter(indicators.index, indicators['hub_influence'], c=colors, s=40, alpha=0.8)
            ax.plot(indicators.index, indicators['hub_influence'], color=self.colors['text'], lw=0.5, alpha=0.3)
            ax.set_ylabel('Hub Influence', color=self.colors['text'])
        else:
            ax.text(0.5, 0.5, 'No hub data', ha='center', va='center', color=self.colors['text'])

        ax.set_title('TOP HUB CHANGES', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    def _draw_hub_influence(self, ax, indicators: pd.DataFrame) -> None:
        """Draw hub influence panel."""
        ax.set_facecolor(self.colors['panel'])

        if 'hub_influence' not in indicators.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', color=self.colors['text'])
            return

        ax.fill_between(indicators.index, 0, indicators['hub_influence'],
                       color=self.colors['accent'], alpha=0.3)
        ax.plot(indicators.index, indicators['hub_influence'], color=self.colors['accent'], lw=1.5)

        thresh = indicators['hub_influence'].quantile(0.90)
        ax.axhline(thresh, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)

        ax.set_ylabel('Hub Influence', color=self.colors['text'])
        ax.set_title(f'HUB INFLUENCE (90%ile: {thresh:.4f})', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    def _draw_category_stack(self, ax, indicators: pd.DataFrame) -> None:
        """Draw category stack panel."""
        ax.set_facecolor(self.colors['panel'])

        cats = [c for c in self.config.categories.keys() if f'bt_{c}' in indicators.columns]

        if not cats:
            ax.text(0.5, 0.5, 'No category data', ha='center', va='center', color=self.colors['text'])
            return

        cat_data = np.array([indicators[f'bt_{c}'].values for c in cats])

        if cat_data.size > 0:
            cat_sum = cat_data.sum(axis=0)
            cat_sum[cat_sum == 0] = 1  # Avoid division by zero
            cat_data_pct = cat_data / cat_sum * 100
            colors = [self.cat_colors.get(c, 'gray') for c in cats]
            ax.stackplot(indicators.index, cat_data_pct, labels=cats, colors=colors, alpha=0.8)
            ax.legend(loc='upper left', fontsize=7, facecolor=self.colors['panel'],
                     labelcolor=self.colors['text'], ncol=2)

        ax.set_ylabel('Share (%)', color=self.colors['text'])
        ax.set_title('CATEGORY DOMINANCE', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.set_ylim(0, 100)
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    def _draw_network_sync(self, ax, indicators: pd.DataFrame) -> None:
        """Draw network sync panel."""
        ax.set_facecolor(self.colors['panel'])

        # Find sync column (could be sync_60d or network_sync)
        sync_col = None
        for col in ['sync_60d', 'network_sync']:
            if col in indicators.columns:
                sync_col = col
                break

        if sync_col is None:
            ax.text(0.5, 0.5, 'No sync data', ha='center', va='center', color=self.colors['text'])
            return

        thresh = indicators[sync_col].quantile(0.90)
        ax.fill_between(
            indicators.index, thresh, indicators[sync_col].max() * 1.1,
            where=indicators[sync_col] > thresh, alpha=0.3, color=self.colors['danger']
        )
        ax.axhline(thresh, color=self.colors['danger'], ls='--', lw=1, alpha=0.7)
        ax.plot(indicators.index, indicators[sync_col], color=self.colors['safe'], lw=1.5)

        ax.set_ylabel('Avg Correlation', color=self.colors['text'])
        ax.set_title(f'NETWORK SYNC (90%ile: {thresh:.4f})', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])

    def _draw_rv_heatmap(self, ax, rv_pct: pd.DataFrame) -> None:
        """Draw RV heatmap panel."""
        ax.set_facecolor(self.colors['panel'])

        days = self.viz.heatmap_days
        rv_recent = rv_pct.iloc[-days:] if len(rv_pct) >= days else rv_pct

        if len(rv_recent) == 0 or len(rv_recent.columns) == 0:
            ax.text(0.5, 0.5, 'No RV data', ha='center', va='center', color=self.colors['text'])
            return

        cmap = LinearSegmentedColormap.from_list(
            'stress',
            ['#238636', '#3fb950', '#d29922', '#f85149', '#da3633'],
            N=100
        )

        im = ax.imshow(rv_recent.T.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)

        ax.set_yticks(range(len(rv_recent.columns)))
        ax.set_yticklabels(rv_recent.columns, color=self.colors['text'], fontsize=9)

        n = len(rv_recent)
        if n > 0:
            ticks = [0, n//4, n//2, 3*n//4, n-1]
            ticks = [t for t in ticks if t < n]
            labels = [rv_recent.index[i].strftime('%m/%d') for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, color=self.colors['text'])

        ax.set_title(f'RV PERCENTILE ({days}d)', fontsize=11, fontweight='bold', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])

        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Percentile', color=self.colors['text'])
        cbar.ax.yaxis.set_tick_params(color=self.colors['text'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.colors['text'])
