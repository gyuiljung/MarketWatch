"""
Report generation module.

Generates text-based analysis reports.
"""

from typing import Dict, Optional
from datetime import datetime
import logging

import pandas as pd

from ..core.config import Config
from ..analysis.network import NetworkSnapshot

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Text report generator.

    Generates comprehensive analysis reports with network state,
    TE flows, volatility alerts, and key correlations.
    """

    def __init__(self, config: Config):
        """
        Initialize report generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.thresholds = config.thresholds

    def generate(
        self,
        snapshot: NetworkSnapshot,
        indicators: pd.DataFrame,
        rv_pct: pd.DataFrame,
        date: datetime,
        te_results: Optional[Dict] = None,
        te_net_flow: Optional[Dict] = None,
    ) -> str:
        """
        Generate analysis report.

        Args:
            snapshot: Current network snapshot
            indicators: Time series indicators
            rv_pct: RV percentile DataFrame
            date: Report date
            te_results: Transfer entropy results
            te_net_flow: Net flow results

        Returns:
            Formatted report string
        """
        current = indicators.iloc[-1] if len(indicators) > 0 else {}
        rv_current = rv_pct.iloc[-1] if len(rv_pct) > 0 else pd.Series()

        # Thresholds
        thresh_sync = indicators['network_sync'].quantile(0.90) if 'network_sync' in indicators else 0
        thresh_hub = indicators['hub_influence'].quantile(0.90) if 'hub_influence' in indicators else 0

        # Extreme/elevated assets
        extreme = [a for a, v in rv_current.items() if v > self.thresholds.rv_extreme]
        elevated = [a for a, v in rv_current.items()
                   if self.thresholds.rv_elevated < v <= self.thresholds.rv_extreme]

        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         NETWORK MONITOR REPORT                                 ║
║                         {date.strftime('%Y-%m-%d %H:%M')}                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT NETWORK STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Top Hub:           {snapshot.top_hub_bt} (Betweenness: {snapshot.top3_bt[0][1] if snapshot.top3_bt else 0:.4f})
Hub Neighbors:     {', '.join(snapshot.neighbors[:5])}{'...' if len(snapshot.neighbors) > 5 else ''}
Hub Avg |ρ|:       {snapshot.hub_avg_corr:.4f}
Hub Influence:     {snapshot.hub_influence:.4f} ({current.get('hub_influence_pct', 0):.0f}%ile)
Network Sync:      {snapshot.network_sync:.4f}

Top 3 Hubs:
"""
        for i, (node, val) in enumerate(snapshot.top3_bt[:3], 1):
            report += f"  #{i}  {node:<12}  Bt = {val:.4f}\n"

        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY BETWEENNESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        if snapshot.cat_betweenness:
            for cat, val in sorted(snapshot.cat_betweenness.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(val * 20)
                report += f"  {cat:<8}: {val:.4f}  {bar}\n"

        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THRESHOLDS (Data-Driven, 90%ile)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Network Sync:      {thresh_sync:.4f}
Hub Influence:     {thresh_hub:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOLATILITY STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTREME (>{self.thresholds.rv_extreme}%ile): {', '.join(extreme) if extreme else 'None'}
Elevated (>{self.thresholds.rv_elevated}%ile): {', '.join(elevated[:5]) if elevated else 'None'}

"""
        # Top 10 by RV
        if len(rv_current) > 0:
            rv_sorted = rv_current.sort_values(ascending=False)
            report += "Top 10 by RV Percentile:\n"
            for asset, pct in rv_sorted.head(10).items():
                status = "⚠ EXTREME" if pct > 90 else ("△ ELEVATED" if pct > 75 else "")
                report += f"  {asset:<12}: {pct:5.1f}%  {status}\n"

        # Alerts
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALERTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        alerts = []

        sync_pct = current.get('network_sync_pct', current.get('sync_60d_pct', 0))
        if sync_pct > 90:
            alerts.append("⚠ NETWORK SYNC > 90%ile - Diversification breakdown")
        if current.get('hub_influence_pct', 0) > 90:
            alerts.append("⚠ HUB INFLUENCE > 90%ile - Hub strongly connected")
        if extreme:
            alerts.append(f"⚠ EXTREME VOLATILITY: {', '.join(extreme[:5])}")

        if alerts:
            for a in alerts:
                report += f"{a}\n"
        else:
            report += "✓ No critical alerts\n"

        # Key correlations
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CORRELATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        corr = snapshot.corr
        pairs = [('USDJPY', 'NKY'), ('USDKRW', 'USDJPY'), ('SPX', 'VIX')]
        if snapshot.top_hub_bt:
            pairs.append((snapshot.top_hub_bt, 'SPX'))

        for a, b in pairs:
            if a in corr.index and b in corr.columns:
                report += f"  ρ({a}, {b}): {corr.loc[a, b]:+.4f}\n"

        report += "━" * 82 + "\n"

        return report

    def generate_summary(
        self,
        snapshot: NetworkSnapshot,
        rv_pct: pd.DataFrame,
        date: datetime
    ) -> str:
        """
        Generate brief summary report.

        Args:
            snapshot: Network snapshot
            rv_pct: RV percentile DataFrame
            date: Report date

        Returns:
            Brief summary string
        """
        rv_current = rv_pct.iloc[-1] if len(rv_pct) > 0 else pd.Series()
        extreme_count = sum(1 for v in rv_current if v > self.thresholds.rv_extreme)

        return f"""
[{date.strftime('%Y-%m-%d')}] Network Summary
─────────────────────────────────
Top Hub: {snapshot.top_hub_bt}
Network Sync: {snapshot.network_sync:.4f}
Extreme Vol Assets: {extreme_count}
─────────────────────────────────
"""
