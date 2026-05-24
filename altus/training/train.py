"""Training loop with multi-task loss, early stopping, MPS support."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from altus.config import TrainConfig
from altus.models.hybrid import HybridLayer1
from altus.training.dataset import ALTUSDataset, collate
from altus.training.metrics import MetricsBundle, evaluate_predictions


def _select_device(preferred: str) -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _multi_task_loss(
    out, batch, cls_w: float, reg_w: float, reg_scale: float = 30.0, label_smoothing: float = 0.0,
    inflection_w: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE on the classification heads, Huber on the regression heads.

    Regression targets (MFE/MAE in points) are normalized to ratio-of-stop
    units (divided by reg_scale, default 30 = SL_POINTS) so they're on the
    same order of magnitude as the BCE loss. Otherwise the raw-points Huber
    swamps the classification gradient and the head we actually trade on is
    undertrained.

    Label smoothing softens hard 0/1 targets to (eps, 1-eps), reducing the
    model's ability to memorize and improving calibration.

    Phase H: if inflection_w > 0 AND out.inflection_logit is not None, also
    add BCE loss for the inflection auxiliary head.
    """
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=1.0)

    # Apply label smoothing to the classification targets
    long_tgt = batch["long_tp"] * (1 - 2 * label_smoothing) + label_smoothing
    short_tgt = batch["short_tp"] * (1 - 2 * label_smoothing) + label_smoothing

    l_long = bce(out.long_tp_logit, long_tgt)
    l_short = bce(out.short_tp_logit, short_tgt)
    # Normalize both predictions and targets to the same scale before Huber.
    l_mfeL = huber(out.mfe_long / reg_scale, batch["mfe_long"] / reg_scale)
    l_maeL = huber(out.mae_long / reg_scale, batch["mae_long"] / reg_scale)
    l_mfeS = huber(out.mfe_short / reg_scale, batch["mfe_short"] / reg_scale)
    l_maeS = huber(out.mae_short / reg_scale, batch["mae_short"] / reg_scale)

    cls_loss = l_long + l_short
    reg_loss = l_mfeL + l_maeL + l_mfeS + l_maeS
    total = cls_w * cls_loss + reg_w * reg_loss
    parts = {
        "loss": float(total.detach()),
        "bce_long": float(l_long.detach()),
        "bce_short": float(l_short.detach()),
        "huber_mfeL": float(l_mfeL.detach()),
        "huber_maeL": float(l_maeL.detach()),
        "huber_mfeS": float(l_mfeS.detach()),
        "huber_maeS": float(l_maeS.detach()),
    }

    # Phase H: auxiliary inflection loss
    if inflection_w > 0 and out.inflection_logit is not None and "inflection" in batch:
        infl_tgt = batch["inflection"] * (1 - 2 * label_smoothing) + label_smoothing
        l_infl = bce(out.inflection_logit, infl_tgt)
        total = total + inflection_w * l_infl
        parts["bce_inflection"] = float(l_infl.detach())
        parts["loss"] = float(total.detach())

    return total, parts


@torch.no_grad()
def _predict(
    model: HybridLayer1,
    loader: DataLoader,
    device: torch.device,
    return_embeddings: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run inference, collect predictions + optional fusion embeddings.

    When `return_embeddings=True`, the returned `preds` dict will include a
    'fusion_embedding' array of shape (N, fusion_hidden). For our default
    config that's (N, 192). Saved as float16 by callers to halve disk space.
    """
    model.eval()
    keys_pred = ["long_tp_prob", "short_tp_prob", "mfe_long", "mae_long", "mfe_short", "mae_short"]
    keys_true = ["long_tp", "short_tp", "mfe_long", "mae_long", "mfe_short", "mae_short"]
    preds_buf: dict[str, list[np.ndarray]] = {k: [] for k in keys_pred}
    truths_buf: dict[str, list[np.ndarray]] = {k: [] for k in keys_true}
    # Phase H: inflection predictions + truths (only collected if model emits them)
    infl_pred_buf: list[np.ndarray] = []
    infl_truth_buf: list[np.ndarray] = []
    emb_buf: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, return_embedding=return_embeddings)
        preds_buf["long_tp_prob"].append(out.long_tp_prob.cpu().numpy())
        preds_buf["short_tp_prob"].append(out.short_tp_prob.cpu().numpy())
        preds_buf["mfe_long"].append(out.mfe_long.cpu().numpy())
        preds_buf["mae_long"].append(out.mae_long.cpu().numpy())
        preds_buf["mfe_short"].append(out.mfe_short.cpu().numpy())
        preds_buf["mae_short"].append(out.mae_short.cpu().numpy())
        for k in keys_true:
            truths_buf[k].append(batch[k].numpy())
        if out.inflection_logit is not None:
            infl_pred_buf.append(out.inflection_prob.cpu().numpy())
            if "inflection" in batch:
                infl_truth_buf.append(batch["inflection"].numpy())
        if return_embeddings and out.fusion_embedding is not None:
            emb_buf.append(out.fusion_embedding.cpu().numpy())
    preds = {k: np.concatenate(v) for k, v in preds_buf.items()}
    truths = {k: np.concatenate(v) for k, v in truths_buf.items()}
    if infl_pred_buf:
        preds["inflection_prob"] = np.concatenate(infl_pred_buf)
    if infl_truth_buf:
        truths["inflection"] = np.concatenate(infl_truth_buf)
    if return_embeddings and emb_buf:
        preds["fusion_embedding"] = np.concatenate(emb_buf, axis=0).astype(np.float32)
    return preds, truths


@dataclass
class TrainResult:
    best_val_metric: float
    best_epoch: int
    history: list[dict]
    val_preds: dict[str, np.ndarray]
    val_truths: dict[str, np.ndarray]
    val_metrics: MetricsBundle


def train_model(
    model: HybridLayer1,
    train_set: ALTUSDataset,
    val_set: ALTUSDataset,
    cfg: TrainConfig | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> TrainResult:
    """Train the hybrid model.

    `verbose` toggles the per-epoch print summaries.
    `show_progress` toggles the per-batch tqdm bar (independently). Set to
    False when running headless / in background so output stays clean.
    """
    cfg = cfg or TrainConfig()
    device = _select_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    best_metric = -float("inf")
    best_state = None
    best_epoch = -1
    patience_left = cfg.early_stop_patience
    history = []

    for epoch in range(cfg.n_epochs):
        model.train()
        running = {"loss": 0.0, "n": 0}
        iter_ = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.n_epochs}", disable=not show_progress, leave=False)
        for batch in iter_:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            # Input feature dropout — randomly zero a fraction of features per batch
            # (a cheap form of data augmentation that reduces overfitting to specific feature combos)
            if cfg.input_feature_dropout > 0:
                x = batch["x"]
                mask = torch.rand_like(x[:, :1, :]) > cfg.input_feature_dropout
                batch["x"] = x * mask
            out = model(batch["x"])
            loss, parts = _multi_task_loss(
                out, batch, cfg.cls_loss_weight, cfg.reg_loss_weight,
                label_smoothing=cfg.label_smoothing,
                inflection_w=getattr(cfg, "inflection_loss_weight", 0.0),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running["loss"] += parts["loss"] * batch["x"].shape[0]
            running["n"] += batch["x"].shape[0]
            if show_progress:
                iter_.set_postfix(loss=f"{parts['loss']:.4f}")
        scheduler.step()
        train_avg = running["loss"] / max(running["n"], 1)

        val_preds, val_truths = _predict(model, val_loader, device)
        val_metrics = evaluate_predictions(val_preds, val_truths)
        watch = val_metrics.mean_auc() if cfg.val_metric == "mean_auc" else val_metrics.auc.get(cfg.val_metric, float("-inf"))

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_avg,
                "val_mean_auc": val_metrics.mean_auc(),
                "val_summary": val_metrics.summary_line(),
            }
        )
        if verbose:
            print(f"[epoch {epoch+1}] train_loss={train_avg:.4f} | val: {val_metrics.summary_line()}")

        if watch > best_metric:
            best_metric = watch
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[early-stop] no improvement for {cfg.early_stop_patience} epochs")
                break

    # Restore best weights for the returned model
    if best_state is not None:
        model.load_state_dict(best_state)

    val_preds, val_truths = _predict(model, val_loader, device)
    val_metrics = evaluate_predictions(val_preds, val_truths)
    return TrainResult(
        best_val_metric=best_metric,
        best_epoch=best_epoch,
        history=history,
        val_preds=val_preds,
        val_truths=val_truths,
        val_metrics=val_metrics,
    )
