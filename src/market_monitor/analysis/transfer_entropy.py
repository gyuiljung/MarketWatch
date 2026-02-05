"""
Transfer Entropy analysis module.

Implements information-theoretic causality measurement between assets.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.constants import (
    TE_DEFAULT_BINS,
    TE_DEFAULT_LAG,
    TE_MIN_SAMPLES,
)
from ..core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass
class TEResult:
    """Transfer Entropy result for a single pair."""
    te_raw: float
    te_z: float
    p_value: float
    significant: bool
    surrogate_mean: float
    surrogate_std: float


@dataclass
class NetFlowResult:
    """Net information flow result for a pair."""
    te_ab: TEResult
    te_ba: TEResult
    net_flow_z: float
    dominant: str
    dominant_direction: str
    flow_strength: float
    both_significant: bool
    any_significant: bool


class TransferEntropyCalculator:
    """
    Transfer Entropy calculator with surrogate-based significance testing.

    Computes directional information flow between all pairs of assets
    and identifies significant causal relationships.
    """

    def __init__(self, config: Config):
        """
        Initialize TE calculator.

        Args:
            config: Configuration object
        """
        te_cfg = config.transfer_entropy
        self.bins = te_cfg.bins
        self.lag = te_cfg.lag
        self.n_surrogates = te_cfg.n_surrogates
        self.alpha = te_cfg.alpha
        self.top_n = te_cfg.top_n

    def _discretize_all(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Discretize all assets into bins.

        Args:
            returns: Returns DataFrame

        Returns:
            2D array of discretized values (n_assets x T)
        """
        n_assets = len(returns.columns)
        T = len(returns)
        result = np.zeros((n_assets, T), dtype=int)

        for i, col in enumerate(returns.columns):
            series = returns[col].values
            valid = ~np.isnan(series)
            ranks = np.argsort(np.argsort(series[valid]))
            n_valid = np.sum(valid)
            result[i, valid] = np.minimum(ranks * self.bins // n_valid, self.bins - 1)

        return result

    def _compute_te_matrix(self, discretized: np.ndarray) -> np.ndarray:
        """
        Compute TE for all pairs.

        Args:
            discretized: Discretized asset data

        Returns:
            TE matrix (n_assets x n_assets)
        """
        n_assets, T = discretized.shape
        n = T - self.lag
        results = np.zeros((n_assets, n_assets))

        for tgt in range(n_assets):
            tgt_d = discretized[tgt]
            tgt_future = tgt_d[self.lag:]
            tgt_past = tgt_d[:-self.lag]

            # P(Y_t, Y_{t-1})
            joint_yy = np.zeros((self.bins, self.bins))
            for t in range(n):
                joint_yy[tgt_future[t], tgt_past[t]] += 1
            joint_yy /= n

            # P(Y_{t-1})
            marginal_y = np.bincount(tgt_past, minlength=self.bins) / n

            for src in range(n_assets):
                if src == tgt:
                    continue

                src_past = discretized[src, :-self.lag]

                # P(Y_t, Y_{t-1}, X_{t-1})
                joint_yyx = np.zeros((self.bins, self.bins, self.bins))
                for t in range(n):
                    joint_yyx[tgt_future[t], tgt_past[t], src_past[t]] += 1
                joint_yyx /= n

                # P(Y_{t-1}, X_{t-1})
                marginal_yx = np.zeros((self.bins, self.bins))
                for t in range(n):
                    marginal_yx[tgt_past[t], src_past[t]] += 1
                marginal_yx /= n

                # Compute TE
                te = 0.0
                for yt in range(self.bins):
                    for yp in range(self.bins):
                        for xp in range(self.bins):
                            p_joint = joint_yyx[yt, yp, xp]
                            if p_joint > 1e-10:
                                p_yy = joint_yy[yt, yp]
                                p_yx = marginal_yx[yp, xp]
                                p_y = marginal_y[yp]
                                if p_yy > 1e-10 and p_yx > 1e-10 and p_y > 1e-10:
                                    te += p_joint * np.log2((p_joint * p_y) / (p_yx * p_yy))

                results[src, tgt] = max(0, te)

        return results

    def compute_all_pairs(self, returns: pd.DataFrame) -> Dict[str, TEResult]:
        """
        Compute TE for all pairs with significance testing.

        Args:
            returns: Returns DataFrame

        Returns:
            Dict of 'A→B' -> TEResult
        """
        assets = list(returns.columns)
        n_assets = len(assets)

        if len(returns) < TE_MIN_SAMPLES:
            logger.warning(f"Insufficient data: {len(returns)} < {TE_MIN_SAMPLES}")
            return {}

        logger.info(f"Computing TE for {n_assets * (n_assets - 1)} pairs...")

        # Discretize
        discretized = self._discretize_all(returns)

        # Raw TE matrix
        te_raw = self._compute_te_matrix(discretized)

        # Surrogate distribution
        te_surrogates = np.zeros((self.n_surrogates, n_assets, n_assets))
        for s in range(self.n_surrogates):
            perm = np.random.permutation(len(returns))
            shuffled = discretized[:, perm]
            te_surrogates[s] = self._compute_te_matrix(shuffled)

        # Z-scores and p-values
        te_mean = np.mean(te_surrogates, axis=0)
        te_std = np.std(te_surrogates, axis=0) + 1e-10
        te_z = (te_raw - te_mean) / te_std
        p_values = np.mean(te_surrogates >= te_raw[np.newaxis, :, :], axis=0)

        # Build results
        results = {}
        for i, src in enumerate(assets):
            for j, tgt in enumerate(assets):
                if i != j:
                    key = f'{src}→{tgt}'
                    results[key] = TEResult(
                        te_raw=float(te_raw[i, j]),
                        te_z=float(te_z[i, j]),
                        p_value=float(p_values[i, j]),
                        significant=p_values[i, j] < self.alpha,
                        surrogate_mean=float(te_mean[i, j]),
                        surrogate_std=float(te_std[i, j]),
                    )

        sig_count = sum(1 for r in results.values() if r.significant)
        logger.info(f"Computed TE: {sig_count}/{len(results)} significant pairs")

        return results

    def compute_net_flow_all(self, returns: pd.DataFrame) -> Dict[str, NetFlowResult]:
        """
        Compute net information flow for all unique pairs.

        Args:
            returns: Returns DataFrame

        Returns:
            Dict of 'A⇄B' -> NetFlowResult
        """
        all_te = self.compute_all_pairs(returns)
        assets = list(returns.columns)

        results = {}
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i < j:  # Unique pairs only
                    key_ab = f'{a}→{b}'
                    key_ba = f'{b}→{a}'

                    if key_ab in all_te and key_ba in all_te:
                        te_ab = all_te[key_ab]
                        te_ba = all_te[key_ba]

                        net_flow = te_ab.te_z - te_ba.te_z
                        dominant = a if net_flow > 0 else b
                        dominant_dir = f'{a}→{b}' if net_flow > 0 else f'{b}→{a}'

                        results[f'{a}⇄{b}'] = NetFlowResult(
                            te_ab=te_ab,
                            te_ba=te_ba,
                            net_flow_z=net_flow,
                            dominant=dominant,
                            dominant_direction=dominant_dir,
                            flow_strength=abs(net_flow),
                            both_significant=te_ab.significant and te_ba.significant,
                            any_significant=te_ab.significant or te_ba.significant,
                        )

        return results

    def get_top_significant(
        self,
        all_te: Dict[str, TEResult],
        n: Optional[int] = None
    ) -> List[Tuple[str, TEResult]]:
        """
        Get top N significant TE pairs by Z-score.

        Args:
            all_te: Dict of all TE results
            n: Number to return (default: self.top_n)

        Returns:
            List of (pair_key, TEResult) tuples
        """
        if n is None:
            n = self.top_n

        sig_pairs = [(k, v) for k, v in all_te.items() if v.significant]
        sig_pairs.sort(key=lambda x: x[1].te_z, reverse=True)
        return sig_pairs[:n]

    def get_top_net_flows(
        self,
        net_flow: Dict[str, NetFlowResult],
        n: Optional[int] = None
    ) -> List[Tuple[str, NetFlowResult]]:
        """
        Get top N net flows by strength.

        Args:
            net_flow: Dict of net flow results
            n: Number to return (default: self.top_n)

        Returns:
            List of (pair_key, NetFlowResult) tuples
        """
        if n is None:
            n = self.top_n

        flows = list(net_flow.items())
        flows.sort(key=lambda x: x[1].flow_strength, reverse=True)
        return flows[:n]
