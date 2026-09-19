#!/bin/bash
# ================================================================
# train_v100.sh — V100 Server Training
# ================================================================
# Run this ON the V100 server:
#   chmod +x scripts/train_v100.sh
#   ./scripts/train_v100.sh
#
# Multi-GPU (if server has 2-4x V100):
#   N_GPUS=4 ./scripts/train_v100.sh
# ================================================================

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

N_GPUS="${N_GPUS:-1}"   # override with: N_GPUS=4 ./train_v100.sh

echo "=============================================="
echo "  Neuro-Neo4j — V100 Training"
echo "  GPUs: $N_GPUS"
echo "  Author: basakanimesh14"
echo "=============================================="

# 1. Verify CUDA + V100
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found!'
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'GPU {i}: {name} | {mem:.0f} GB')
    assert 'v100' in name.lower(), f'Expected V100, got: {name}'
print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
print('V100 check OK — using FP16 (BF16 not supported on Volta)')
"

# 2. Activate venv
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating venv..."
    uv venv .venv --python 3.11
    uv sync
fi
source .venv/bin/activate

# 3. Start Neo4j if docker is available
if command -v docker &>/dev/null; then
    docker compose up -d neo4j 2>/dev/null || echo "[WARN] Neo4j not started — mock mode"
fi

# 4. Run
if [ "$N_GPUS" -gt 1 ]; then
    echo "[INFO] DDP mode: $N_GPUS GPUs"
    torchrun --standalone --nproc_per_node="$N_GPUS" \
        scripts/train_launcher.py \
        --config config/v100_config.yaml \
        --ddp \
        --skip-data-gen   # remove if generating fresh data
else
    echo "[INFO] Single GPU mode"
    python scripts/train_launcher.py \
        --config config/v100_config.yaml
fi

echo "=============================================="
echo "  Done! Check logs/ and checkpoints/"
echo "=============================================="
