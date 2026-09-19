#!/bin/bash
# ================================================================
# eval_agx.sh — AGX Orin: Training + Evaluation
# ================================================================
# AGX Orin role in this project:
#   - Train on small cohorts / quick iterations
#   - Evaluate final checkpoints from V100/DGX Spark
#   - Test the full pipeline before deploying to clinical hardware
#
# Run this ON the AGX Orin:
#   chmod +x scripts/eval_agx.sh
#   ./scripts/eval_agx.sh
#
# To train (not just evaluate):
#   TRAIN=1 ./scripts/eval_agx.sh
# ================================================================

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN="${TRAIN:-0}"    # default: evaluate only. Set TRAIN=1 to also train.
CHECKPOINT="${CHECKPOINT:-}"   # path to .pt file to evaluate (optional)

echo "=============================================="
echo "  Neuro-Neo4j — AGX Orin Pipeline"
echo "  Mode: $([ "$TRAIN" = "1" ] && echo 'Train + Evaluate' || echo 'Evaluate only')"
echo "  Author: basakanimesh14"
echo "=============================================="

# 1. Verify AGX
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA found. Check JetPack installation.'
name = torch.cuda.get_device_name(0)
mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'Device: {name} | Unified memory: {mem:.0f} GB')
print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
print('NOTE: Using FP16 (AGX Orin does not support BF16 in JetPack 5.x)')
"

# 2. Set max performance for training
echo "[INFO] Setting MAXN performance mode..."
sudo nvpmodel -m 0 2>/dev/null || echo "[WARN] nvpmodel not available"
sudo jetson_clocks   2>/dev/null || echo "[WARN] jetson_clocks not available"

# 3. Activate venv
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating venv (JetPack-aware)..."
    # On Jetson, use system Python which has CUDA-aware torch pre-installed
    python3 -m venv .venv --system-site-packages
    .venv/bin/pip install -r <(uv export --no-hashes 2>/dev/null) 2>/dev/null \
        || echo "[WARN] Could not install all deps. Continuing with system packages."
fi
source .venv/bin/activate 2>/dev/null || true

# 4a. TRAIN mode
if [ "$TRAIN" = "1" ]; then
    echo "[INFO] Training on AGX Orin..."
    python scripts/train_launcher.py \
        --config config/agx_config.yaml

# 4b. EVALUATE only — load checkpoint from V100/DGX
else
    if [ -z "$CHECKPOINT" ]; then
        # Find the most recent checkpoint
        CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -1 || echo "")
    fi

    if [ -z "$CHECKPOINT" ]; then
        echo "[ERROR] No checkpoint found. Run training first or set CHECKPOINT=/path/to/model.pt"
        exit 1
    fi

    echo "[INFO] Evaluating checkpoint: $CHECKPOINT"
    python3 - <<PYEOF
import sys, torch, yaml
from pathlib import Path

sys.path.insert(0, '.')
with open('config/agx_config.yaml') as f:
    cfg = yaml.safe_load(f)

from models.gnn.gcn_gat_model import BrainGNNModel
from models.gnn.train import GNNTrainer
from models.gnn.dataset import BrainConnectomeDataset
from validation.subject_split import subject_level_split
from torch_geometric.loader import DataLoader

device = torch.device('cuda')
mc     = cfg['model']
model  = BrainGNNModel(
    in_channels    = mc['in_channels'],
    hidden_channels= mc['hidden_channels'],
    num_classes    = mc['num_classes'],
    num_heads      = mc['num_heads'],
    dropout        = mc['dropout'],
    num_gcn_layers = mc['num_gcn_layers'],
    num_gat_layers = mc['num_gat_layers'],
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
trainer   = GNNTrainer(model, optimizer, criterion, device, cfg,
                       amp_dtype=torch.float16)
trainer.load_checkpoint('$CHECKPOINT')

ds = BrainConnectomeDataset(
    root           = cfg['data']['processed_dir'],
    subjects_csv   = cfg['data']['subject_metadata_file'],
    edge_list_csv  = cfg['data']['edge_list_file'],
)
ratios = cfg['validation']['split_ratios']
_, _, test_idx = subject_level_split(ds, ratios[0], ratios[1], seed=cfg['validation']['split_seed'])
test_ds     = [ds[i] for i in test_idx]
test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                         shuffle=False, num_workers=4, pin_memory=False)

report = trainer.evaluate(test_loader, split='test')
print(report)
PYEOF
fi

# 5. Monitor resources
echo "[INFO] Resource usage:"
tegrastats --interval 1000 &
TSTAT_PID=$!
sleep 5
kill $TSTAT_PID 2>/dev/null || true

echo "=============================================="
echo "  AGX pipeline complete. Check logs/"
echo "=============================================="
