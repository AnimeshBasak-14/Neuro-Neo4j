#!/bin/bash
# ================================================================
# train_dgx_spark.sh — DGX Spark (GB10 Blackwell) Training
# ================================================================
# Run this ON the DGX Spark machine:
#   chmod +x scripts/train_dgx_spark.sh
#   ./scripts/train_dgx_spark.sh
# ================================================================

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Neuro-Neo4j — DGX Spark Training"
echo "  Hardware: GB10 Grace Blackwell Superchip"
echo "  Author:   basakanimesh14"
echo "=============================================="

# 1. Verify GPU
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found!'
name = torch.cuda.get_device_name(0)
mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'GPU: {name} | Memory: {mem:.1f} GB')
print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
"

# 2. Set Blackwell max performance
echo "[INFO] Setting max performance mode..."
sudo nvpmodel -m 0 2>/dev/null || echo "[WARN] nvpmodel not available"
sudo jetson_clocks 2>/dev/null  || echo "[WARN] jetson_clocks not available"

# 3. Install dependencies (if venv not present)
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python 3.11 venv..."
    uv venv .venv --python 3.11
    uv sync
fi
source .venv/bin/activate

# 4. Start Neo4j (optional — mock mode used if not running)
if command -v docker &>/dev/null; then
    echo "[INFO] Starting Neo4j container..."
    docker compose up -d neo4j 2>/dev/null || echo "[WARN] Docker not running — using mock mode"
fi

# 5. Run full pipeline
echo "[INFO] Launching training..."
python scripts/train_launcher.py \
    --config config/dgx_spark_config.yaml

echo "=============================================="
echo "  Training complete! Check logs/ and checkpoints/"
echo "=============================================="
