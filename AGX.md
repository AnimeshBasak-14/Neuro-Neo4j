# AGX Orin — Complete Setup & Run Guide

## Neuro-Neo4j on NVIDIA AGX Orin (JetPack 5.x / 6.x)

**Author:** basakanimesh14  
**Hardware:** NVIDIA AGX Orin — 12-core ARM Cortex-A78AE · 64 GB unified LPDDR5 · 1792-core Ampere GPU  
**Role in this project:** Model training (quick iterations) · Final evaluation · Pipeline testing

---

## Hardware Snapshot

| Spec | Value |
|---|---|
| GPU | 1792-core NVIDIA Ampere iGPU |
| CPU | 12-core ARM Cortex-A78AE |
| Memory | 64 GB unified LPDDR5 (CPU + GPU share) |
| Storage | 64 GB eMMC + NVMe SSD (recommended) |
| OS | Ubuntu 20.04 (JP5) / Ubuntu 22.04 (JP6) |
| CUDA | 11.4 (JP5) / 12.2–12.6 (JP6) |
| TDP | 15–60 W (configurable via nvpmodel) |

> [!IMPORTANT]
> AGX Orin uses **unified memory** — CPU and GPU share the same 64 GB pool.  
> There is no separate VRAM. This means `pin_memory=False` in all DataLoaders  
> (already set in `config/agx_config.yaml`).

> [!CAUTION]
> **Do NOT install PyTorch with `pip install torch` or `uv sync` on AGX.**  
> Standard PyPI wheels are compiled for x86_64 and will silently install  
> a CPU-only ARM build with no CUDA support. Use NVIDIA's Jetson index (Step 5).

---

## Prerequisites Checklist

Before you start, confirm the following:

```bash
# 1. JetPack is flashed and working
cat /etc/nv_tegra_release
# Expected: R35 (JP5) or R36 (JP6)

# 2. CUDA is available
nvcc --version
# Expected: release 11.4 (JP5) or 12.x (JP6)

# 3. Python is present
python3 --version
# Expected: 3.8.x (JP5) or 3.10.x (JP6)

# 4. Internet is reachable
ping -c 2 github.com

# 5. Enough disk space
df -h /
# Need at least 20 GB free
```

---

## Step 0 — Identify Your JetPack Version

```bash
cat /etc/nv_tegra_release
```

| Output | JetPack | Python | CUDA | Wheel URL |
|---|---|---|---|---|
| `R35, REVISION: 3.x / 4.x` | **JP 5.1.x** | 3.8 | 11.4 | `/jp/v512/` |
| `R36, REVISION: 2.x / 3.x` | **JP 6.x** | 3.10 | 12.2–12.6 | `/jp/v61/` |

Note your version — you will use it in **Step 5**.

---

## Step 1 — System Update & Essential Packages

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    git \
    curl \
    wget \
    tmux \
    htop \
    build-essential \
    python3-pip \
    python3-venv \
    python3-dev \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    nano

echo "System packages installed."
```

---

## Step 2 — Set Max Performance Mode

> [!IMPORTANT]
> Always do this before training. The default power mode limits GPU/CPU clocks significantly.

```bash
# MAXN = all cores at maximum frequency (60W TDP)
sudo nvpmodel -m 0

# Pin all clocks to maximum
sudo jetson_clocks

# Verify
sudo nvpmodel -q
# Expected output: NV Power Mode: MAXN

# Verify GPU clock is at max
cat /sys/devices/gpu.0/devfreq/*/cur_freq
# Expected: a large number (e.g. 1300000000 for 1.3 GHz)
```

To monitor power and thermals during training (in a separate terminal):
```bash
# Real-time resource usage
watch -n 1 tegrastats

# Or one-shot
tegrastats --interval 2000
```

---

## Step 3 — Clone the Repository

```bash
cd ~

# Clone (you will be prompted for GitHub credentials)
git clone https://github.com/AnimeshBasak-14/Neuro-Neo4j.git
cd Neuro-Neo4j
```

> **GitHub authentication options:**
>
> **Option A — Personal Access Token (easier):**
> - GitHub → Settings → Developer Settings → Personal access tokens → Classic
> - Generate token with `repo` scope
> - Use it as password when prompted
>
> **Option B — SSH key (recommended for repeated use):**
> ```bash
> ssh-keygen -t ed25519 -C "agx-orin-neuro"
> cat ~/.ssh/id_ed25519.pub   # copy this
> # Paste into: GitHub → Settings → SSH and GPG keys → New SSH key
> git clone git@github.com:AnimeshBasak-14/Neuro-Neo4j.git
> ```

---

## Step 4 — Create Python Virtual Environment

```bash
cd ~/Neuro-Neo4j

# --system-site-packages lets us access Jetson's pre-built CUDA libraries
python3 -m venv .venv --system-site-packages

source .venv/bin/activate

# Confirm venv is active
which python3
# Expected: /home/<user>/Neuro-Neo4j/.venv/bin/python3

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## Step 5 — Install PyTorch (Jetson-Specific Wheel)

> [!CAUTION]
> Use **only** the NVIDIA Jetson index URL below. Standard `pip install torch` gives  
> you a CPU-only wheel — `torch.cuda.is_available()` will return `False`.

### JetPack 6.x (R36) — Python 3.10

```bash
pip install torch torchvision \
    --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/
```

### JetPack 5.1.x (R35) — Python 3.8

```bash
pip install torch torchvision \
    --index-url https://developer.download.nvidia.com/compute/redist/jp/v512/
```

### Verify CUDA is working

```bash
python3 -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
print('CUDA version    :', torch.version.cuda)
print('Device name     :', torch.cuda.get_device_name(0))
mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
print('Unified memory  :', mem, 'GB')

# Quick tensor test
x = torch.randn(100, 100).cuda()
print('GPU tensor shape:', x.shape, '(CUDA OK)')
"
```

Expected output:
```
PyTorch version : 2.3.0+nv24.xx
CUDA available  : True
CUDA version    : 12.2
Device name     : Orin
Unified memory  : 64 GB
GPU tensor shape: torch.Size([100, 100]) (CUDA OK)
```

---

## Step 6 — Install PyTorch Geometric

```bash
# Get installed torch version (e.g. "2.3.0")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
echo "Detected PyTorch: $TORCH_VER"

# Install core PyG (pure Python — works on ARM64)
pip install torch-geometric

# Try to install sparse extensions (may fail on some JP versions — that is OK)
pip install torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html" \
    --no-build-isolation 2>/dev/null \
    || echo "[INFO] Sparse extensions unavailable — dense fallback will be used (no impact on this project)"

# Verify
python3 -c "import torch_geometric; print('PyG version:', torch_geometric.__version__)"
```

---

## Step 7 — Install All Other Dependencies

```bash
pip install \
    "fastapi==0.141.1" \
    "uvicorn[standard]" \
    "httpx" \
    "pydantic>=2.0" \
    "pydantic-settings" \
    "neo4j" \
    "networkx" \
    "scikit-learn" \
    "scipy" \
    "pandas" \
    "numpy" \
    "matplotlib" \
    "seaborn" \
    "rich" \
    "pyyaml" \
    "python-dotenv" \
    "tqdm" \
    "psutil"

echo "All dependencies installed."
```

### Verify all key imports

```bash
python3 -c "
import torch, torch_geometric
import fastapi, uvicorn
import neo4j, networkx
import sklearn, scipy, pandas, numpy
import rich, yaml

print('torch          :', torch.__version__)
print('torch-geometric:', torch_geometric.__version__)
print('fastapi        :', fastapi.__version__)
print('neo4j          :', neo4j.__version__)
print('networkx       :', networkx.__version__)
print()
print('All imports OK — environment is ready.')
"
```

---

## Step 8 — Configure Environment Variables

```bash
# Copy the environment template
cp .env.example .env

# Edit (replace values as needed)
nano .env
```

Set the following in `.env`:

```bash
# Neo4j connection (leave defaults if not running Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neuro_password

# Model checkpoint path
MODEL_CHECKPOINT_PATH=checkpoints/best_model_agx.pt

# Logging
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
```

> [!NOTE]
> **Neo4j is optional for training.** The pipeline automatically switches to  
> mock mode when Neo4j is unreachable. All Neo4j calls are handled internally.

---

## Step 9 — (Optional) Start Neo4j with Docker

Only required if you want the full knowledge graph functionality:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Pull and start Neo4j (ARM64-compatible official image)
docker run -d \
    --name neo4j \
    --restart unless-stopped \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neuro_password \
    -e NEO4J_PLUGINS='["graph-data-science"]' \
    -v $HOME/neo4j_data:/data \
    neo4j:5.15

# Wait 30 seconds for startup
sleep 30
docker logs neo4j | grep -i "started\|error"

# Test connection
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','neuro_password'))
with driver.session() as s:
    s.run('RETURN 1')
print('Neo4j connection OK')
driver.close()
"
```

---

## Step 10 — Create Required Directories

```bash
mkdir -p \
    checkpoints \
    logs \
    data/processed \
    data/synthetic/connectomes

echo "Directory structure created."
```

---

## Step 11 — Generate Synthetic Training Data

```bash
# Generate 250 subjects: 125 Healthy + 125 Diseased (AAL-116 atlas)
python3 data/synthetic/generate_synthetic.py \
    --n-healthy 125 \
    --n-diseased 125 \
    --n-regions 116 \
    --seed 42 \
    --output-dir data/synthetic

# Verify output
echo "Subjects generated:"
wc -l data/synthetic/subjects.csv

echo "Connectome files:"
ls data/synthetic/connectomes/ | wc -l
# Expected: 250
```

---

## Step 12 — Run ETL (Connectomes → Edge List)

```bash
python3 - <<'PYEOF'
import sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '.')
from neo4j_layer.etl.transform import ConnectomeTransformer
from data.synthetic.generate_synthetic import AAL116_REGIONS

conn_dir = Path('data/synthetic/connectomes')
out_dir  = Path('data/processed')
out_dir.mkdir(exist_ok=True)

transformer = ConnectomeTransformer(region_names=AAL116_REGIONS)
all_edges   = []

files = sorted(conn_dir.glob('*.csv'))
print(f"Processing {len(files)} connectome files...")

for f in files:
    parts     = f.stem.split('_')
    subject   = '_'.join(parts[:2])
    diagnosis = parts[2] if len(parts) > 2 else 'Unknown'
    matrix    = pd.read_csv(f, index_col=0).values.astype('float32')
    matrix    = transformer.regress_global_signal(matrix)
    matrix    = transformer.normalize(matrix, method='zscore')
    matrix    = transformer.threshold(matrix, threshold=0.3)
    edges     = transformer.to_edge_list(matrix, subject_id=subject, diagnosis=diagnosis)
    all_edges.append(edges)

edge_df = pd.concat(all_edges, ignore_index=True)
edge_df.to_csv(out_dir / 'edge_list.csv', index=False)
print(f"ETL complete: {len(edge_df):,} edges → data/processed/edge_list.csv")
PYEOF

# Copy subject metadata
cp data/synthetic/subjects.csv data/processed/subjects.csv
echo "subjects.csv copied to data/processed/"
```

---

## Step 13 — Dry Run (Smoke Test)

Verifies all 10 pipeline components work before committing to full training:

```bash
PYTHONIOENCODING=utf-8 python3 dry_run/dry_run_all.py
```

Expected:
```
[1/10]  ETL Pipeline          PASS  ~1500ms
[2/10]  GDS Mock              PASS  ~5ms
[3/10]  SVM Baseline          PASS  ~430ms
[4/10]  RF-GDV Baseline       PASS  ~455ms
[5/10]  node2vec Baseline     PASS  ~4ms
[6/10]  GNN Forward Pass      PASS  ~3000ms
[7/10]  Dataset Loading       PASS  ~50ms
[8/10]  GNN Trainer Dry Run   PASS  ~20ms
[9/10]  GNNExplainer          PASS  ~150ms
[10/10] FastAPI Endpoints     PASS  ~5000ms

All 10 tests PASSED
```

If any test fails, check the error message and re-install that specific dependency.

---

## Step 14 — Train the Model

> [!TIP]
> Run inside `tmux` so training survives SSH disconnects.

```bash
# Start a tmux session
tmux new-session -s neuro_training

# Inside tmux — activate venv and start training
source .venv/bin/activate

python3 scripts/train_launcher.py --config config/agx_config.yaml
```

**Detach from tmux** (training continues in background):  
Press `Ctrl+B`, then `D`

**Reattach to check progress:**
```bash
tmux attach -t neuro_training
```

**Monitor GPU usage in a separate terminal:**
```bash
watch -n 2 tegrastats
# Look for: GR3D_FREQ 100% during training
```

### What training looks like

```
Hardware: AGX Orin | Config: config/agx_config.yaml
[INFO] Generating 250 synthetic subjects...
[INFO] ETL complete: 1,243,500 edges
[INFO] Dataset: 250 graphs | train: 175 | val: 37 | test: 38
[INFO] GNNTrainer: device=cuda | precision=fp16 | epochs=200

Epoch   1/200 | lr=1.00e-04 | train[loss=0.721 f1=0.511] | val[loss=0.698 f1=0.543] | 4.1s
Epoch   2/200 | lr=2.00e-04 | train[loss=0.689 f1=0.567] | val[loss=0.671 f1=0.598] | 3.9s
...
Epoch  45/200 | lr=6.54e-04 | train[loss=0.398 f1=0.841] | val[loss=0.461 f1=0.812] | 3.7s
  >> Best checkpoint (val_f1=0.8120) saved → checkpoints/best_model_agx.pt

[INFO] Training complete. Best val Macro-F1: 0.8234
[INFO] Evaluation [test] — Acc=0.8421 | Macro-F1=0.8389 | ROC-AUC=0.9012
```

---

## Step 15 — View Results

```bash
# Test set metrics
cat logs/test_metrics.json

# Biomarker explanations
python3 -c "
import json
with open('logs/biomarker_explanations.json') as f:
    data = json.load(f)
print(f'Explained {len(data)} subjects')
for s in data[:3]:
    print(f\"  {s['subject_id']} (class={s['target_class']}): top region = {s['top_regions'][0]['region']}\")
"
```

---

## Step 16 — (Optional) Transfer Checkpoint to V100 Server

```bash
# From AGX → V100 server
scp checkpoints/best_model_agx.pt \
    username@v100-server-ip:/data/neuro-neo4j/checkpoints/

# On V100 — continue training from AGX checkpoint:
# (Edit v100_config.yaml to point to the transferred checkpoint, then run)
# python3 scripts/train_launcher.py --config config/v100_config.yaml --skip-data-gen
```

---

## Expected Time Breakdown

| Phase | Duration |
|---|---|
| System setup (Steps 1–8) | ~20–30 min (one-time) |
| Data generation (250 subjects) | 2–3 min |
| ETL | 4–6 min |
| Dry run | ~30 sec |
| **Training (200 epochs, FP16)** | **20–35 min** |
| Test evaluation | < 5 sec |
| **GNNExplainer (38 subjects)** | **15–25 min** |
| **Total (first run)** | **~65–100 min** |
| **Total (subsequent runs)** | **~40–65 min** |

> [!TIP]
> To reduce GNNExplainer time: edit `config/agx_config.yaml` and set `explainer.epochs: 300`
> (down from 1000). This halves explainer time with minor quality reduction.

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

```bash
# Wrong torch was installed. Reinstall from Jetson index:
pip uninstall torch torchvision -y
# JP6:
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/
# JP5:
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v512/
```

### Process killed during training (OOM)

```bash
# Reduce batch size in config/agx_config.yaml:
#   batch_size: 32  →  16
#   num_workers: 4  →  2

# Also free memory before training:
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### `ImportError: cannot import name 'neo4j'` (name collision)

```bash
# Always run from the project root — never from inside a subdirectory
cd ~/Neuro-Neo4j
python3 scripts/train_launcher.py --config config/agx_config.yaml
```

### Training is very slow (< 1 it/s)

```bash
# Check you're in MAXN mode:
sudo nvpmodel -q     # must show MAXN
sudo jetson_clocks   # must re-run after reboot

# Check GPU utilisation:
tegrastats | grep GR3D
# Must show GR3D_FREQ 80%+ during training
```

### `ModuleNotFoundError: No module named 'torch_geometric'`

```bash
source .venv/bin/activate
pip install torch-geometric
```

---

## Quick Reference

```bash
# Full setup (after clone) — run once:
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/
pip install torch-geometric fastapi uvicorn httpx "pydantic>=2.0" pydantic-settings \
    neo4j networkx scikit-learn scipy pandas numpy matplotlib seaborn rich pyyaml \
    python-dotenv tqdm psutil
cp .env.example .env
sudo nvpmodel -m 0 && sudo jetson_clocks
mkdir -p checkpoints logs data/processed

# Run training (every time):
source .venv/bin/activate
python3 data/synthetic/generate_synthetic.py --n-healthy 125 --n-diseased 125
# (run ETL — see Step 12)
python3 dry_run/dry_run_all.py                                       # smoke test
python3 scripts/train_launcher.py --config config/agx_config.yaml   # train
cat logs/test_metrics.json                                           # results
```
