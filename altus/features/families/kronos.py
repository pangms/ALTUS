"""Family 6: Kronos pretrained candlestick foundation model — feature engineer mode.

Uses frozen Kronos (NeoQuasar/Kronos-base, 102M params, MIT license) as a
PRETRAINED FEATURE ENGINEER, not as a parallel encoder branch.

For each entry bar T:
  1. Take the last `lookback` bars of OHLCV (causal: through T-1)
  2. Run Kronos to sample `n_samples` trajectories of the next `horizon` bars
  3. Derive 12 aggregate features from the predicted distribution
  4. Return as a feature row aligned to T

WHY THIS DESIGN INSTEAD OF AN ENCODER BRANCH:
  - Kronos doesn't natively expose hidden states; using it for embeddings would
    require fork-and-patch (engineering risk that breaks on every Kronos update).
  - Kronos was trained as an autoregressive predictor — using its predictions
    is using the tool as designed.
  - Pre-compute once, cache to disk, then all future training is fast (just
    join the cached features onto our normal feature pipeline).
  - At live deployment: cache covers historical bars; only the current bar
    needs a fresh Kronos call each minute (~50ms on GPU, ~200ms on CPU).

INSTALL (one-time, on the machine that will compute features):
  pip install git+https://github.com/shiyu-coder/Kronos.git
  pip install transformers huggingface_hub

CACHE:
  Pass a `cache_path` (Parquet file). If it exists, features are loaded from
  cache. If not, they're computed and saved. Re-computing happens only when
  the cache is deleted, which is the right tradeoff for a heavy feature.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass  # avoid importing kronos at module load time

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KronosConfig:
    """Knobs for Kronos feature computation. Defaults match our project setup."""
    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "cuda"           # or 'cpu'; Kronos is autoregressive — CPU is slow
    max_context: int = 512         # Kronos-base hard limit
    lookback_bars: int = 240       # how much history to feed Kronos per entry
    horizon_bars: int = 60         # how far ahead to predict (matches our label H)
    n_samples: int = 10            # trajectories sampled per entry
    temperature: float = 1.0
    top_p: float = 0.9
    tp_points: float = 30.0        # for tp-hit probability features (matches our labels)
    sl_points: float = 30.0
    # When pre-computing on long histories, only run Kronos every `decimation`
    # bars and forward-fill between them. Reduces compute by Nx at small accuracy
    # cost — regime context evolves slowly so adjacent bars usually have very
    # similar Kronos features.
    decimation: int = 60           # compute every 60 bars (= 1 hour on 1m bars)


# Public feature column names — kept stable so caches survive across runs.
FEATURE_COLUMNS = (
    "kronos_pred_mfe_long_mean",
    "kronos_pred_mae_long_mean",
    "kronos_pred_mfe_short_mean",
    "kronos_pred_mae_short_mean",
    "kronos_pred_tp_long_prob",
    "kronos_pred_tp_short_prob",
    "kronos_pred_mfe_uncertainty",
    "kronos_pred_mae_uncertainty",
    "kronos_pred_trajectory_slope",
    "kronos_pred_realized_vol",
    "kronos_pred_endpoint_skew",
    "kronos_pred_max_run_length",
)


# ---------------------------------------------------------------------------
# Lazy import of Kronos
# ---------------------------------------------------------------------------

def _import_kronos():
    """Lazy import — Kronos isn't a hard dependency of altus.

    Kronos ships as a script-style codebase (not a pip-installable package).
    Per the official README, the workflow is:
       git clone https://github.com/shiyu-coder/Kronos.git
       cd Kronos && pip install -r requirements.txt
    Then `from model import Kronos, KronosTokenizer, KronosPredictor` works
    only when CWD is the Kronos directory OR Kronos is on sys.path.

    This function tries several discovery strategies and falls back to a
    clear error message with install instructions.
    """
    import sys
    from pathlib import Path

    # Strategy 1: maybe it's pip-installed as a package called 'kronos'
    try:
        from kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
        return Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        pass

    # Strategy 2: search common locations for a cloned Kronos repo, then add to sys.path
    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        repo_root.parent / "Kronos",          # sibling of ALTUS repo (./Kronos)
        Path("/workspace/Kronos"),             # RunPod-style workspace
        Path.home() / "Kronos",                # user home
        repo_root / "Kronos",                  # nested inside ALTUS (less common)
    ]
    for d in candidate_dirs:
        model_file = d / "model.py"
        model_pkg = d / "model" / "__init__.py"
        if model_file.exists() or model_pkg.exists():
            sys.path.insert(0, str(d))
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
                return Kronos, KronosTokenizer, KronosPredictor
            except ImportError:
                sys.path.pop(0)
                continue

    raise ImportError(
        "Kronos not found in any expected location. To install it:\n"
        "    cd /workspace          # or wherever you keep external repos\n"
        "    git clone https://github.com/shiyu-coder/Kronos.git\n"
        "    cd Kronos && pip install -r requirements.txt\n"
        "Then re-run the Kronos-feature build/training script. The kronos module\n"
        "will auto-detect the clone at /workspace/Kronos or as a sibling of ALTUS."
    )


# ---------------------------------------------------------------------------
# Feature derivation from sampled trajectories
# ---------------------------------------------------------------------------

def _features_from_trajectories(
    trajectories: np.ndarray,
    entry_price: float,
    cfg: KronosConfig,
) -> dict[str, float]:
    """Compute the 12 Kronos-derived features from sampled trajectories.

    trajectories: array of shape (n_samples, horizon_bars, 4) for [open, high, low, close].
                  We use high/low/close — open is implicit (close of prior bar).
    entry_price : the price at which we'd theoretically enter (typically open[T]).
    """
    if trajectories.ndim != 3 or trajectories.shape[2] < 4:
        # Defensive: if Kronos returned an unexpected shape, fill with NaN
        return {col: float("nan") for col in FEATURE_COLUMNS}

    highs = trajectories[..., 1]   # (n_samples, H)
    lows = trajectories[..., 2]
    closes = trajectories[..., 3]

    # ---- Per-trajectory metrics -------------------------------------------
    max_high_per = highs.max(axis=1)              # (n_samples,)
    min_low_per = lows.min(axis=1)
    final_close_per = closes[:, -1]

    mfe_long_per = np.maximum(max_high_per - entry_price, 0.0)
    mae_long_per = np.maximum(entry_price - min_low_per, 0.0)
    mfe_short_per = np.maximum(entry_price - min_low_per, 0.0)
    mae_short_per = np.maximum(max_high_per - entry_price, 0.0)

    # TP-hit probabilities: fraction of trajectories where TP-side excursion
    # exceeds tp_points BEFORE the SL side hits sl_points within the horizon.
    # For simplicity we use "did TP excursion reach +tp first" (lighter than
    # full path-dependent simulation, but accurate enough at H=60).
    tp_long_hit_per = np.zeros(trajectories.shape[0], dtype=np.float32)
    tp_short_hit_per = np.zeros(trajectories.shape[0], dtype=np.float32)
    for i in range(trajectories.shape[0]):
        # Walk the trajectory bar by bar, stop at first barrier touch
        for b in range(trajectories.shape[1]):
            if highs[i, b] - entry_price >= cfg.tp_points:
                tp_long_hit_per[i] = 1.0
                break
            if entry_price - lows[i, b] >= cfg.sl_points:
                break  # long SL hit first
        for b in range(trajectories.shape[1]):
            if entry_price - lows[i, b] >= cfg.tp_points:
                tp_short_hit_per[i] = 1.0
                break
            if highs[i, b] - entry_price >= cfg.sl_points:
                break  # short SL hit first

    # ---- Trajectory shape features ----------------------------------------
    # Linear-fit slope per trajectory's close series, then mean across samples
    H = trajectories.shape[1]
    x_axis = np.arange(H, dtype=np.float64)
    slopes_per = np.array([np.polyfit(x_axis, c, 1)[0] for c in closes])

    # Realized vol per trajectory: std of log returns
    log_returns = np.diff(np.log(closes + 1e-9), axis=1)  # (n_samples, H-1)
    realized_vol_per = log_returns.std(axis=1)

    # Skewness of endpoint distribution across samples
    if len(final_close_per) > 2:
        from scipy.stats import skew
        endpoint_skew = float(skew(final_close_per))
    else:
        endpoint_skew = 0.0

    # Max consecutive same-direction bar streak across trajectories
    def _max_run(close_series: np.ndarray) -> int:
        d = np.sign(np.diff(close_series))
        if len(d) == 0:
            return 0
        best, cur, prev = 1, 1, d[0]
        for v in d[1:]:
            if v == prev and v != 0:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
                prev = v
        return best
    max_runs_per = np.array([_max_run(c) for c in closes])

    # ---- Aggregate to the 12 features -------------------------------------
    return {
        "kronos_pred_mfe_long_mean": float(mfe_long_per.mean()),
        "kronos_pred_mae_long_mean": float(mae_long_per.mean()),
        "kronos_pred_mfe_short_mean": float(mfe_short_per.mean()),
        "kronos_pred_mae_short_mean": float(mae_short_per.mean()),
        "kronos_pred_tp_long_prob": float(tp_long_hit_per.mean()),
        "kronos_pred_tp_short_prob": float(tp_short_hit_per.mean()),
        "kronos_pred_mfe_uncertainty": float(mfe_long_per.std() + mfe_short_per.std()) / 2.0,
        "kronos_pred_mae_uncertainty": float(mae_long_per.std() + mae_short_per.std()) / 2.0,
        "kronos_pred_trajectory_slope": float(slopes_per.mean()),
        "kronos_pred_realized_vol": float(realized_vol_per.mean()),
        "kronos_pred_endpoint_skew": endpoint_skew,
        "kronos_pred_max_run_length": float(max_runs_per.mean()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator entry point — CACHE ONLY. Does NOT compute Kronos inline.

    Why: Kronos feature computation is a ~25 GPU-hour operation on 5 years of
    data. It should never happen inline during a training run. The discipline
    is: precompute once into a parquet cache, then training loads instantly.

    To build the cache (one-time):
        python scripts/build_kronos_cache.py --start 2021-01-01 --end 2026-04-01

    Returns features from the cache aligned to df_1m.index. Raises with a
    clear message if cache is missing.
    """
    from altus.config import ARTIFACT_DIR
    cache_path = ARTIFACT_DIR / "kronos_features.parquet"
    if not cache_path.exists():
        raise RuntimeError(
            f"Kronos features cache not found at {cache_path}.\n"
            f"Kronos features must be pre-computed before training (it's a heavy GPU op).\n"
            f"Run on a CUDA-equipped machine:\n"
            f"    python scripts/build_kronos_cache.py --start 2021-01-01 --end 2026-04-01\n"
            f"This takes ~25 GPU-hours on RTX 4090 once. After that, all training runs\n"
            f"load these features instantly.\n"
            f"To train WITHOUT Kronos features for now, omit 'kronos' from --families."
        )
    log.info(f"Loading Kronos features from cache: {cache_path}")
    cached = pd.read_parquet(cache_path)
    return cached.reindex(df_1m.index)


def build_cache(
    df_1m: pd.DataFrame,
    cache_path: Path | str,
    cfg: KronosConfig | None = None,
) -> pd.DataFrame:
    """HEAVY operation. Computes Kronos features and saves them to `cache_path`.

    Use this from a one-time build script (scripts/build_kronos_cache.py), NOT
    during a training run. Resumes from existing cache if present.
    """
    cfg = cfg or KronosConfig()
    cache_path = Path(cache_path)

    # ---- Cache check ------------------------------------------------------
    if cache_path.exists():
        log.info(f"Loading Kronos features from existing cache: {cache_path}")
        cached = pd.read_parquet(cache_path)
        return cached.reindex(df_1m.index)

    # ---- Lazy-import Kronos ----------------------------------------------
    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()

    log.info(f"Loading Kronos model: {cfg.model_name}")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_name)
    model = Kronos.from_pretrained(cfg.model_name)
    predictor = KronosPredictor(
        model, tokenizer, device=cfg.device, max_context=cfg.max_context
    )

    # ---- Iterate over entries ---------------------------------------------
    # We compute only at decimated points to save GPU time, then forward-fill
    # between them. Adjacent 1m bars have very similar Kronos predictions
    # (regime evolves slowly), so this is a safe ~60x speedup.
    n = len(df_1m)
    out = pd.DataFrame(
        np.full((n, len(FEATURE_COLUMNS)), np.nan, dtype=np.float32),
        index=df_1m.index,
        columns=list(FEATURE_COLUMNS),
    )

    cols = ["open", "high", "low", "close", "volume"]
    if not all(c in df_1m.columns for c in cols):
        raise ValueError(f"df_1m missing required columns; need {cols}")

    # Entry positions: every `decimation`-th bar starting from lookback
    entry_positions = list(range(cfg.lookback_bars, n, cfg.decimation))
    log.info(
        f"Computing Kronos features at {len(entry_positions):,} entry points "
        f"(decimation={cfg.decimation}, lookback={cfg.lookback_bars}, "
        f"horizon={cfg.horizon_bars}, n_samples={cfg.n_samples})"
    )

    from tqdm.auto import tqdm
    for pos in tqdm(entry_positions, desc="kronos"):
        # Causal slice: bars [pos - lookback, pos - 1] inclusive (NOT pos itself)
        hist = df_1m.iloc[pos - cfg.lookback_bars : pos][cols]
        entry_price = float(df_1m["open"].iloc[pos])

        # Future timestamps that Kronos will predict for
        y_ts = pd.date_range(
            df_1m.index[pos], periods=cfg.horizon_bars, freq="1min"
        )

        try:
            pred_df = predictor.predict(
                df=hist,
                x_timestamp=pd.Series(hist.index),
                y_timestamp=pd.Series(y_ts),
                pred_len=cfg.horizon_bars,
                T=cfg.temperature,
                top_p=cfg.top_p,
                sample_count=cfg.n_samples,
            )
            # Expected: pred_df shape includes sample dimension. The exact layout
            # may need reshape — we handle this defensively.
            trajectories = _normalize_predict_output(
                pred_df, cfg.n_samples, cfg.horizon_bars
            )
            feats = _features_from_trajectories(trajectories, entry_price, cfg)
        except Exception as e:
            log.warning(f"Kronos predict failed at pos {pos}: {e}")
            feats = {col: np.nan for col in FEATURE_COLUMNS}

        for col, val in feats.items():
            out.at[df_1m.index[pos], col] = val

    # Forward-fill between decimation points (NaN -> last computed value)
    out = out.ffill()
    # The first lookback_bars region has NaN; leave it that way so downstream
    # dropna() trims the warmup region cleanly.

    # ---- Save cache -------------------------------------------------------
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path)
    log.info(f"Saved Kronos features to cache: {cache_path}")

    return out


def _normalize_predict_output(
    pred_df, n_samples: int, horizon: int
) -> np.ndarray:
    """Coerce KronosPredictor.predict() output into a (n_samples, horizon, 4) numpy array.

    KronosPredictor returns a DataFrame; the exact layout for multi-sample
    predictions isn't fully documented in their model card. We handle the most
    likely cases and fail loudly on the rest.
    """
    if isinstance(pred_df, pd.DataFrame):
        cols = ["open", "high", "low", "close"]
        if not all(c in pred_df.columns for c in cols):
            raise ValueError(f"prediction df missing OHLC; got {pred_df.columns.tolist()}")
        # Single trajectory case
        if len(pred_df) == horizon:
            return pred_df[cols].to_numpy(dtype=np.float64)[None, :, :]  # (1, H, 4)
        # Stacked-samples case: rows = n_samples * horizon
        if len(pred_df) == n_samples * horizon:
            return (
                pred_df[cols]
                .to_numpy(dtype=np.float64)
                .reshape(n_samples, horizon, 4)
            )
    # If we get a list of DataFrames (one per sample), stack them
    if isinstance(pred_df, list) and len(pred_df) == n_samples:
        arrs = []
        for p in pred_df:
            arrs.append(p[["open", "high", "low", "close"]].to_numpy(dtype=np.float64))
        return np.stack(arrs, axis=0)
    raise ValueError(
        f"unexpected predict output: type={type(pred_df)}, "
        f"len={len(pred_df) if hasattr(pred_df, '__len__') else '?'}"
    )
