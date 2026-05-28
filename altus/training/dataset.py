"""PyTorch Dataset bridging the feature matrix + labels to model inputs.

Each sample is keyed by an entry timestamp T (a row in the labeled set). The
features tensor for that sample is the seq_len-bar window ending at T (inclusive):
features rows [T-seq_len+1, ..., T]. Because features are already causally
shifted (see altus.features.pipeline), every value in that window was knowable
at the moment of entry at the open of bar T. No look-ahead.

Targets at T:
  - long_tp, short_tp:   binary {0, 1}
  - mfe_long, mae_long, mfe_short, mae_short:  non-negative float, points
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from altus.labels.triple_barrier import LabelOutput


@dataclass
class WindowSpec:
    seq_len: int


class ALTUSDataset(Dataset):
    """Holds the full feature matrix + labels in memory. Indexed by labeled-sample id."""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: LabelOutput,
        sample_positions: np.ndarray,
        seq_len: int,
    ) -> None:
        # Align features to label timestamps. Features and labels live on the
        # 1m grid; the labeler already dropped session-break-spanning rows.
        feat_index_pos = features.index.get_indexer(labels.index)
        if (feat_index_pos < 0).any():
            n_missing = int((feat_index_pos < 0).sum())
            raise ValueError(f"{n_missing} label timestamps are absent from features index")
        self._feature_pos_per_label = feat_index_pos.astype(np.int64)

        # Convert features to a single float32 numpy block for fast slicing.
        self._features_np = features.to_numpy(dtype=np.float32)
        self._n_features = features.shape[1]

        # Stash labels
        self._labels = labels
        self.sample_positions = sample_positions.astype(np.int64)
        self.seq_len = seq_len

        # Filter out any sample positions whose window would extend before row 0.
        valid = np.array(
            [self._feature_pos_per_label[p] >= seq_len - 1 for p in self.sample_positions]
        )
        self.sample_positions = self.sample_positions[valid]

    def __len__(self) -> int:
        return len(self.sample_positions)

    def __getitem__(self, idx: int) -> dict:
        pos = self.sample_positions[idx]
        feat_end = self._feature_pos_per_label[pos]
        feat_start = feat_end - self.seq_len + 1
        # (L, F)  — .copy() makes the slice writable so torch.from_numpy doesn't warn
        window = self._features_np[feat_start : feat_end + 1].copy()
        # 3-class direction label derived from the mutually-exclusive
        # long_tp / short_tp barriers:
        #   0 = long_wins  (long_tp=1)
        #   1 = short_wins (short_tp=1)
        #   2 = neither    (timeout / both stops)
        long_tp = int(self._labels.long_tp[pos])
        short_tp = int(self._labels.short_tp[pos])
        if long_tp:
            direction_class = 0
        elif short_tp:
            direction_class = 1
        else:
            direction_class = 2
        return {
            "x": torch.from_numpy(window),
            # Direction class for cross-entropy (post-audit primary target).
            "direction_class": torch.tensor(direction_class, dtype=torch.long),
            # Original binary labels — kept for inference truths + evaluation
            # against the historical baseline.
            "long_tp": torch.tensor(long_tp, dtype=torch.float32),
            "short_tp": torch.tensor(short_tp, dtype=torch.float32),
            "mfe_long": torch.tensor(self._labels.mfe_long[pos], dtype=torch.float32),
            "mae_long": torch.tensor(self._labels.mae_long[pos], dtype=torch.float32),
            "mfe_short": torch.tensor(self._labels.mfe_short[pos], dtype=torch.float32),
            "mae_short": torch.tensor(self._labels.mae_short[pos], dtype=torch.float32),
            # Phase H: inflection auxiliary target (Q26) — emitted always;
            # the loss ignores it when inflection_loss_weight=0.
            "inflection": torch.tensor(self._labels.inflection_label[pos], dtype=torch.float32),
            # Predictive framework labels (2026-05-25 — FRAMEWORK.md C-tier).
            "return_H15": torch.tensor(self._labels.return_H15[pos], dtype=torch.float32),
            "return_H60": torch.tensor(self._labels.return_H60[pos], dtype=torch.float32),
            "path_shape_class": torch.tensor(self._labels.path_shape_class[pos], dtype=torch.long),
            "clears_up_first": torch.tensor(self._labels.clears_up_first[pos], dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "direction_class": torch.stack([b["direction_class"] for b in batch], dim=0),
        "long_tp": torch.stack([b["long_tp"] for b in batch], dim=0),
        "short_tp": torch.stack([b["short_tp"] for b in batch], dim=0),
        "mfe_long": torch.stack([b["mfe_long"] for b in batch], dim=0),
        "mae_long": torch.stack([b["mae_long"] for b in batch], dim=0),
        "mfe_short": torch.stack([b["mfe_short"] for b in batch], dim=0),
        "mae_short": torch.stack([b["mae_short"] for b in batch], dim=0),
        "inflection": torch.stack([b["inflection"] for b in batch], dim=0),
        "return_H15": torch.stack([b["return_H15"] for b in batch], dim=0),
        "return_H60": torch.stack([b["return_H60"] for b in batch], dim=0),
        "path_shape_class": torch.stack([b["path_shape_class"] for b in batch], dim=0),
        "clears_up_first": torch.stack([b["clears_up_first"] for b in batch], dim=0),
    }
