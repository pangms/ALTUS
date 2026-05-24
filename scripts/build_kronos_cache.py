"""Build the Kronos features cache for ALTUS training (one-time heavy GPU op).

After this script completes, all future training runs that use the kronos
feature family will load these features instantly from the cache.

Usage (on a CUDA-equipped machine):
    # First time only: clone + install Kronos (per its README)
    cd /workspace
    git clone https://github.com/shiyu-coder/Kronos.git
    cd Kronos && pip install -r requirements.txt

    # Then run this script from ALTUS root
    cd /workspace/ALTUS
    python3 scripts/build_kronos_cache.py

Defaults: 3 years of MNQ (2023-04 to 2026-03), decimation=60 (compute every
hour), 10 samples per entry. Estimated runtime: 60-90 min on RTX 4090.
Cost: ~$0.50-1.

Override via CLI flags if needed. The cache path defaults to
artifacts/kronos_features.parquet which is where altus.features.families.kronos
expects to find it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-04-01",
                        help="Start of MNQ data range (default 2023-04-01)")
    parser.add_argument("--end", default="2026-03-31",
                        help="End of MNQ data range (default 2026-03-31)")
    parser.add_argument("--decimation", type=int, default=60,
                        help="Compute Kronos features every N 1m bars (default 60 = once per hour). "
                             "Lower = more accurate but proportionally more compute.")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Trajectories sampled per entry point (default 10)")
    parser.add_argument("--lookback", type=int, default=240,
                        help="Bars of OHLCV history fed to Kronos per call (default 240)")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Bars Kronos predicts ahead (default 60, matches our label H)")
    parser.add_argument("--cache-path", default=None,
                        help="Where to save the parquet cache (default: artifacts/kronos_features.parquet)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for Kronos inference (default cuda)")
    args = parser.parse_args()

    from altus.config import ARTIFACT_DIR
    from altus.data import load_mnq
    from altus.features.families.kronos import KronosConfig, build_cache

    cache_path = Path(args.cache_path) if args.cache_path else ARTIFACT_DIR / "kronos_features.parquet"

    print("=" * 70)
    print(" KRONOS FEATURES CACHE BUILDER")
    print("=" * 70)
    print(f"  data range  : {args.start} -> {args.end}")
    print(f"  decimation  : every {args.decimation} 1m bars")
    print(f"  samples     : {args.n_samples} trajectories per entry")
    print(f"  lookback    : {args.lookback} bars")
    print(f"  horizon     : {args.horizon} bars")
    print(f"  device      : {args.device}")
    print(f"  cache path  : {cache_path}")
    print("=" * 70)

    t0 = time.time()
    print("\n[1/3] Loading MNQ data...")
    df = load_mnq(start=args.start, end=args.end)
    print(f"      Loaded {len(df):,} bars ({df.index[0]} -> {df.index[-1]})")

    print("\n[2/3] Building Kronos features (heavy GPU op)...")
    cfg = KronosConfig(
        device=args.device,
        lookback_bars=args.lookback,
        horizon_bars=args.horizon,
        n_samples=args.n_samples,
        decimation=args.decimation,
    )
    features = build_cache(df, cache_path, cfg=cfg)

    elapsed_min = (time.time() - t0) / 60
    cache_size_mb = cache_path.stat().st_size / 1e6

    print("\n[3/3] Done.")
    print("=" * 70)
    print(f"  features    : {features.shape}")
    print(f"  cache file  : {cache_path}")
    print(f"  cache size  : {cache_size_mb:.1f} MB")
    print(f"  elapsed     : {elapsed_min:.1f} min")
    print("=" * 70)
    print("\nNow run training with kronos enabled:")
    print("  python3 -u scripts/train_cloud.py --full \\")
    print("    --families vol,trend,anomaly,kronos --use-revin")


if __name__ == "__main__":
    main()
