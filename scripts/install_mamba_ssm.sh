#!/usr/bin/env bash
# Install mamba-ssm with CUDA kernels for the Mamba peer encoder fast path.
#
# Why this is non-trivial:
#   mamba-ssm and causal-conv1d have a build step that imports torch — but
#   pip's default --build-isolation creates a fresh env where torch isn't
#   installed. So we need --no-build-isolation, which means torch + build
#   deps must be pre-installed in the active env.
#
# Pre-reqs: torch with CUDA support already installed (the RunPod images
# we use ship with torch 2.8.0+cu128). Verified by the first step below.
#
# Compatibility note: mamba-ssm wheels exist for many sm_arch values but
# may NOT exist for Blackwell (sm_120) as of early 2026. If wheels are
# absent, pip will compile from source — this takes ~10-30 minutes and
# requires NVCC. If you see "no matching wheel", be patient.
#
# Usage:
#   cd /workspace/ALTUS
#   bash scripts/install_mamba_ssm.sh

set -uo pipefail

echo "============================================================"
echo " mamba-ssm install for fast Mamba peer encoder"
echo "============================================================"

# Step 0: verify torch + CUDA
echo ""
echo "[0/4] Verifying torch + CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — mamba-ssm needs CUDA'
print(f'  torch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  device: {torch.cuda.get_device_name(0)}')
print(f'  capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
" || { echo "torch+CUDA check failed"; exit 1; }

# Step 1: install build dependencies that mamba-ssm/causal-conv1d need
echo ""
echo "[1/4] Installing build dependencies (packaging, ninja, setuptools)..."
pip install -q packaging ninja "setuptools>=61"

# Step 2: install causal-conv1d (mamba-ssm depends on it)
echo ""
echo "[2/4] Installing causal-conv1d (no build isolation so it sees torch)..."
pip install -q --no-build-isolation causal-conv1d \
    || { echo "  causal-conv1d install failed — likely Blackwell wheel missing"; exit 1; }

# Step 3: install mamba-ssm
echo ""
echo "[3/4] Installing mamba-ssm (no build isolation so it sees torch)..."
pip install -q --no-build-isolation mamba-ssm \
    || { echo "  mamba-ssm install failed — likely Blackwell wheel missing"; exit 1; }

# Step 4: verify import + ALTUS picks it up
echo ""
echo "[4/4] Verifying the fast path is enabled..."
python3 -c "
import torch
from altus.models.mamba import _HAS_MAMBA_SSM, MambaEncoder
print(f'  _HAS_MAMBA_SSM flag: {_HAS_MAMBA_SSM}')
assert _HAS_MAMBA_SSM, 'mamba_ssm imported but ALTUS flag is False — investigate'

# Quick forward pass on CUDA to confirm kernel works
enc = MambaEncoder(in_features=20, d_model=64, n_blocks=2).cuda()
x = torch.randn(4, 60, 20, device='cuda')
import time
t0 = time.time()
y = enc(x)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f'  forward pass on CUDA: {elapsed*1000:.1f} ms (should be <100ms)')
print('  SUCCESS — Mamba fast path is live.')
"
