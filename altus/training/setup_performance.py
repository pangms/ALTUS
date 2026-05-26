"""Online setup-performance tracker — Group G/D feature provider.

Maintains a rolling window of recent (setup × regime) outcomes:
  - setup_id   ("sfs", "sfa", ...)
  - regime_id  (an integer bucket, typically from BOCPD or vol_regime)
  - win        (1 if R-multiple > 0, else 0)
  - r_multiple (signed R)

Provides:
  - get_recent_wr(setup_id, regime_id, n=30): WR over last n outcomes
  - get_recent_r_avg(setup_id, regime_id, n=30): mean R-multiple
  - record_trade(setup_id, regime_id, r_multiple): update on close
  - drift_score(setup_id): how much the conditional WR has drifted from baseline

This is a SIMPLE on-disk JSON-backed tracker. For online live trading the
persistence is critical (must survive restarts); for offline training/sim
it's mostly used to compute B1 / D4 / G2 features bootstrapped from training
data.

Cold-start: when fewer than 10 records exist for a (setup, regime) pair,
falls back to the baseline WR from SETUPS.md (via altus.training.l2_router
BASELINE_WR).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from altus.training.l2_router import BASELINE_WR


@dataclass
class SetupTrade:
    """A single completed trade — what we log for the tracker."""
    setup_id: str
    regime_id: int            # discrete regime bucket
    r_multiple: float         # signed R (win = positive, loss = negative)
    timestamp: str            # ISO format for ordering


class SetupPerformanceTracker:
    """Rolling-window per-(setup × regime) outcome tracker.

    Internal structure: dict mapping (setup_id, regime_id) → deque of recent trades.
    Bounded to a maximum N per cell to prevent unbounded growth.
    """

    def __init__(self, max_recent_per_cell: int = 100,
                 cold_start_threshold: int = 10,
                 persistence_path: Path | str | None = None):
        self.max_recent = max_recent_per_cell
        self.cold_start_threshold = cold_start_threshold
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._cells: dict[tuple[str, int], deque[SetupTrade]] = {}
        if self.persistence_path and self.persistence_path.exists():
            self._load()

    # ---------- Recording ----------
    def record_trade(self, setup_id: str, regime_id: int, r_multiple: float,
                     timestamp: str = "") -> None:
        key = (setup_id, int(regime_id))
        if key not in self._cells:
            self._cells[key] = deque(maxlen=self.max_recent)
        self._cells[key].append(SetupTrade(
            setup_id=setup_id, regime_id=int(regime_id),
            r_multiple=float(r_multiple), timestamp=timestamp,
        ))
        if self.persistence_path:
            self._save()

    # ---------- Reads ----------
    def get_recent_wr(self, setup_id: str, regime_id: int, n: int = 30) -> float:
        """Win rate over last n trades for this cell. Falls back to baseline."""
        key = (setup_id, int(regime_id))
        trades = self._cells.get(key)
        if trades is None or len(trades) < self.cold_start_threshold:
            return BASELINE_WR.get(setup_id, 0.50)
        recent = list(trades)[-n:]
        wins = sum(1 for t in recent if t.r_multiple > 0)
        return wins / len(recent)

    def get_recent_r_avg(self, setup_id: str, regime_id: int, n: int = 30) -> float:
        """Mean R-multiple over last n trades. Falls back to 0 when cold."""
        key = (setup_id, int(regime_id))
        trades = self._cells.get(key)
        if trades is None or len(trades) < self.cold_start_threshold:
            return 0.0
        recent = list(trades)[-n:]
        return float(np.mean([t.r_multiple for t in recent]))

    def get_recent_pooled_wr(self, setup_id: str, n: int = 60) -> float:
        """WR across ALL regimes for a setup — used for D4 feature."""
        all_trades: list[SetupTrade] = []
        for (sid, _), trades in self._cells.items():
            if sid == setup_id:
                all_trades.extend(trades)
        if len(all_trades) < self.cold_start_threshold:
            return BASELINE_WR.get(setup_id, 0.50)
        all_trades.sort(key=lambda t: t.timestamp)
        recent = all_trades[-n:]
        wins = sum(1 for t in recent if t.r_multiple > 0)
        return wins / len(recent)

    def drift_score(self, setup_id: str, regime_id: int, n: int = 30) -> float:
        """How much the conditional WR has drifted from baseline.
        Returns score in [0, 1] — 0 = on-track, 1 = severely degraded.
        """
        recent_wr = self.get_recent_wr(setup_id, regime_id, n)
        baseline = BASELINE_WR.get(setup_id, 0.50)
        # Drift = how much below baseline (only flag down-drift, up is fine)
        drift = max(0.0, baseline - recent_wr) / max(baseline, 0.01)
        return float(min(drift * 2.0, 1.0))  # scale: 5pp drop → 0.5 drift

    # ---------- Bulk bootstrap ----------
    def bootstrap_from_trades(self, trades: Iterable[SetupTrade]) -> None:
        """Populate from a batch of historical trades (e.g., from training data).
        Useful for cold-start before live deployment.
        """
        for t in trades:
            self.record_trade(t.setup_id, t.regime_id, t.r_multiple, t.timestamp)

    # ---------- Stats ----------
    def summary(self) -> dict:
        """Per-cell summary stats for reporting."""
        out = {}
        for (sid, rid), trades in self._cells.items():
            if not trades:
                continue
            r_vals = [t.r_multiple for t in trades]
            out[f"{sid}_r{rid}"] = {
                "n": len(trades),
                "wr": sum(1 for r in r_vals if r > 0) / len(r_vals),
                "r_mean": float(np.mean(r_vals)),
                "r_std": float(np.std(r_vals)),
            }
        return out

    # ---------- Persistence ----------
    def _save(self) -> None:
        data = {
            f"{sid}|{rid}": [
                {"setup_id": t.setup_id, "regime_id": t.regime_id,
                 "r_multiple": t.r_multiple, "timestamp": t.timestamp}
                for t in trades
            ]
            for (sid, rid), trades in self._cells.items()
        }
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self.persistence_path) as f:
            data = json.load(f)
        for key, items in data.items():
            sid, rid = key.split("|")
            cell_key = (sid, int(rid))
            self._cells[cell_key] = deque(maxlen=self.max_recent)
            for it in items:
                self._cells[cell_key].append(SetupTrade(**it))
