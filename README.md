# ALTUS

Day-trading bot for MNQ futures (TopStep). Layer 1 is a ModernTCN +
Mamba/xLSTM hybrid that predicts triple-barrier outcomes from
multi-timeframe 1m price action.

## Running on a cloud GPU (e.g. RunPod)

Spin up any pod with a PyTorch 2.x image + CUDA GPU (RTX 4090 is plenty
for our model size). In the pod's web terminal:

```bash
git clone https://github.com/pangms/ALTUS.git
cd ALTUS
bash scripts/setup_cloud.sh
```

`setup_cloud.sh` installs deps, sanity-checks CUDA, then runs
`scripts/train_cloud.py` end-to-end (full configs, ~30–60 min on 4090).

Trained model weights + metrics land in `artifacts/cloud_<timestamp>/`.
Download via RunPod's file browser, then **stop the pod** to halt billing.

## Running locally on a Mac (MPS)

Smoke-validated configs:

```bash
pip install -r requirements.txt
python scripts/smoke_test.py     # 3-month sanity run, ~18 min on M-series
```

The smoke run is intentionally small — it validates the pipeline before
committing to a long training run. For real training, use a CUDA GPU.

## Layout

```
altus/
├── data/loader.py           # MNQ + cross-asset parquet loaders
├── features/pipeline.py     # multi-TF features (causal, no leakage)
├── labels/triple_barrier.py # TP/SL/timeout + MFE/MAE regression targets
├── splits/purged.py         # walk-forward + embargo + OOS lockbox
├── models/
│   ├── modern_tcn.py        # local-pattern branch
│   ├── mamba.py             # selective SSM (long-context branch)
│   ├── xlstm.py             # mLSTM/sLSTM (alternative long-context)
│   ├── hybrid.py            # 2 peers + fusion + 6 output heads
│   └── baseline_momentum.py
└── training/
    ├── train.py             # multi-task loss, early stop
    ├── metrics.py           # AUC, PR-AUC, Brier, IC, top-K
    ├── calibration.py       # isotonic + temperature scaling
    └── sim_pnl.py           # Layer-1 standalone trading sim
```

## Layer 1 acceptance criteria (move to Layer 2 when ALL hold on OOS)

- AUC ≥ 0.54 per side
- Top-decile signal win rate ≥ 58%
- Simulated PnL positive after costs
- Sharpe ≥ 0.4, Max DD < 15% of test-period peak
- ≥ 65% of monthly buckets positive

See `altus/config.py` for the canonical values.
