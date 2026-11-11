# Neuro-Neo4j: Setup and Run Guide

A graph neural network pipeline that classifies functional brain connectomes (AAL-116 atlas) and extracts explanation subgraphs, with an optional Neo4j graph database layer. This file covers installing, running and reproducing everything in the repository.

## 1. Requirements

| Item | Needed for | Notes |
|---|---|---|
| Python **3.11** | everything | Not 3.10, 3.12 or 3.14: some binary wheels (PyTorch, scientific stack) are not built for them. |
| NVIDIA GPU (CUDA) | training | Tested on Tesla V100S-PCIe-32GB (Volta). CPU works for data prep, evaluation and figures. |
| ~5 GB disk | data and checkpoints | 500 synthetic connectomes plus a 2.5M-row edge list. |
| Docker | optional Neo4j | Only for the database layer. Not needed to train or evaluate. |

No `sudo` is needed. Everything runs in user space.

## 2. Install

Dependencies are declared in `pyproject.toml` (Python pinned to 3.11). `uv sync` installs them, but it pulls the default PyTorch wheel, which may not match your CUDA driver. On a GPU server, install PyTorch first from the right index and then the rest, as below. Plain `python -m venv` and `pip` work the same way.

```bash
# 1. Virtual environment (Python 3.11)
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. PyTorch with CUDA (pick the wheel matching your driver; the reported runs used cu124)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Everything else
uv pip install torch-geometric numpy pandas scipy scikit-learn networkx \
    matplotlib seaborn pyyaml tqdm rich \
    neo4j fastapi "uvicorn[standard]" httpx python-dotenv pydantic pydantic-settings
```

Versions used for the reported results: PyTorch 2.6.0+cu124, PyTorch Geometric 2.8.0, NetworkX 3.6.1, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.0.6.

Optional: `node2vec` (needs `gensim`, so Python 3.11 only). Without it the node2vec baseline silently falls back to a spectral embedding, so note which path ran before quoting its numbers.

`uv sync` is the shortcut if the default PyTorch wheel suits your machine (the V100 launcher script uses it when `.venv` is missing).

**Check the install:**

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__)"

# Offline smoke test: no GPU, database or data files needed (exit code 0 on success)
python scripts/train_launcher.py --dry-run
```

The smoke test checks that the config loads, a synthetic connectome goes through the ETL, the model runs a forward pass, the patient-level split has no overlap, and the metrics report works.

## 3. Repository layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | Dependency list (Python 3.11) |
| `config/` | YAML configs: `config.yaml` (CPU/default), `v100_config.yaml`, `agx_config.yaml`, `dgx_config.yaml`, `dgx_spark_config.yaml` |
| `data/synthetic/generate_synthetic.py` | Synthetic AAL-116 cohort generator (planted disease signature, 3 scanner noise levels) |
| `data/processed/` | `subjects.csv`, `edge_list.csv`, `cached_dataset.pt` (created by the pipeline) |
| `neo4j_layer/` | ETL transformer, Cypher schema and ingestion scripts, GDS feature module |
| `models/gnn/` | `BrainGNNModel`, dataset, trainer, GNNExplainer wrapper |
| `models/baselines/` | RBF-SVM, RF on graphlet vectors, node2vec |
| `models/metrics.py` | Accuracy, macro-F1, ROC-AUC, Youden threshold |
| `validation/` | Patient-level splitting, domain-shift helpers |
| `scripts/` | `train_launcher.py`, `train_v100.sh`, `eval_agx.sh`, `train_dgx_spark.sh`, figure scripts |
| `api/`, `monitoring/` | FastAPI service and drift monitor |
| `checkpoints/`, `logs/` | Model weights and metrics (outputs) |
| `report/` | Final project report (DOCX, PDF) and figures |

## 4. Train and evaluate

One command runs the whole pipeline: it generates the synthetic cohort (skipped if it already exists), runs the ETL, builds the PyG dataset, trains, evaluates on the held-out test set and runs GNNExplainer.

```bash
python scripts/train_launcher.py --config config/v100_config.yaml
```

Omit `--config` and the launcher picks a config from the GPU name (V100, Orin, Blackwell, A100/H100), or `config/config.yaml` on CPU.

| Flag | Effect |
|---|---|
| `--config PATH` | Use a specific YAML config |
| `--skip-data-gen` | Reuse existing `data/synthetic/connectomes/*.npy` |
| `--skip-etl` | Reuse existing `data/processed/edge_list.csv` |
| `--skip-explain` | Train and evaluate only |
| `--ddp` | Multi-GPU; start with `torchrun` (below) |
| `--dry-run` | Run the offline smoke test and exit (no training) |

Multi-GPU on one machine:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_launcher.py \
    --config config/v100_config.yaml --ddp --skip-data-gen
```

`scripts/train_v100.sh` wraps the V100 case (single GPU by default, `N_GPUS=4 ./scripts/train_v100.sh` for DDP). If `.venv` is missing it runs `uv sync`, which installs the default PyTorch wheel; on a CUDA server prefer the install in section 2.

**Outputs**

| File | Content |
|---|---|
| `checkpoints/best_model_v100.pt` | Best checkpoint by validation macro-F1, plus training history |
| `logs/test_metrics.json` | Test accuracy, macro-F1, ROC-AUC and the threshold fields |
| `logs/biomarker_explanations.json` | Top edges and regions per explained test subject |

Reference result for the V100 config (75 held-out subjects): test ROC-AUC 0.746, validation ROC-AUC 0.823. The `calibrated_accuracy` field in `test_metrics.json` uses a threshold fitted on the test set, so treat it as optimistic. Choose the threshold on validation data for any reported accuracy.

**Key V100 settings** (`config/v100_config.yaml`): hidden width 192, 3 GCN + 1 GATv2 layers (4 heads), dropout 0.25, AdamW (lr 1e-3, weight decay 1e-4), label smoothing 0.1, batch size 32, FP16 with `GradScaler`, `pin_memory: true`, early stopping on validation macro-F1 (patience 25).

## 5. Reproduce the report figures

```bash
# Figures 3.3 and 4.1 to 4.7 plus verified_numbers.json (CPU only, about 1 to 2 minutes)
CUDA_VISIBLE_DEVICES= python scripts/make_report_figures.py

# Architecture diagrams (Figures 3.1 and 3.2)
python scripts/make_report_diagrams.py

# Older training/ROC/ablation plots
python scripts/visualize_results.py --output-dir logs/plots
```

The figure script needs `data/processed/`, `data/synthetic/connectomes/` and `checkpoints/best_model_v100.pt`. It writes into `report/figures/`.

## 6. Optional: Neo4j graph database

Training does not need a database. The dataset class reads the CSV files directly, which keeps tests and dry runs working offline.

```bash
docker compose up -d                 # Neo4j 5.15 with the GDS plugin
# Browser: http://localhost:7474   login: neo4j / neuro_password
```

The compose file mounts `data/processed/` into the Neo4j import folder as `data/`. Before ingesting, create the region table that `load_csv.cypher` expects:

```python
import pandas as pd
from data.synthetic.generate_synthetic import AAL116_REGIONS
pd.DataFrame({
    "atlasId": range(116),
    "anatomicalLabel": AAL116_REGIONS,
    "hemisphere": [n[-1] if n[-2:] in ("_L", "_R") else "M" for n in AAL116_REGIONS],
}).to_csv("data/processed/brain_regions.csv", index=False)
```

Then run, in this order, in the Neo4j Browser or `cypher-shell`:

1. `neo4j_layer/schema/constraints.cypher` (constraints and indexes, run once)
2. `neo4j_layer/etl/load_csv.cypher` (subjects, regions, `CONNECTED_TO` edges, `PATIENT_OF` links)

Ingesting all 2.5M edges takes a while. To test quickly, filter `edge_list.csv` to a few subjects first. If Docker is unavailable (shared servers without root), skip this section; nothing else depends on it.

## 7. Optional: run the API

The service is meant for ordinary CPU hosts, not the training GPU.

```bash
uvicorn api.main:app --port 8000
# Interactive docs: http://localhost:8000/docs
```

It exposes health, `/predict`, `/explain` and `/metrics` routes (see `/docs`). It loads `config/config.yaml` and reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` and `NEO4J_DATABASE` from the environment (defaults: `bolt://localhost:7687`, `neo4j`, `neuro_password`, `neo4j`). Without a running Neo4j it falls back to mock mode.

## 8. Project rules (keep these when changing code)

- **Python 3.11 only.** Do not create a directory or module named `neo4j` at the repo root; it shadows the official driver. Local code is imported from `neo4j_layer.*`.
- **No `sudo`.** Everything lives in the venv and local paths.
- **Volta (V100) has no BF16.** Use `torch.float16` with `torch.cuda.amp.GradScaler()`, and `pin_memory=True` in DataLoaders.
- **GPU use.** The training GPU is for training and offline testing. Serving runs on CPU compute.
- **GNNExplainer.** Never wrap it in `torch.no_grad()` (it optimises masks by gradient descent). Do not pass `target=` when `explanation_type='model'` (PyG 2.4 and later).
- **Dataset class.** `BrainConnectomeDataset` is a plain `torch.utils.data.Dataset` with `__len__` and `__getitem__`. Do not make it inherit from PyG `InMemoryDataset`.
- **Offline first.** Tests and dry runs must work without a live Neo4j database.
- **Evaluation hygiene.** Split by patient ID, and choose any decision threshold on validation data only.

## 9. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ImportError` or `AttributeError` on `from neo4j import GraphDatabase` | A local `neo4j/` folder or file is shadowing the driver. Rename it. |
| `CUDA not available` | Wrong PyTorch wheel. Reinstall from the `cu124` (or `cu121`) index; check `nvidia-smi`. |
| BF16 error on V100 | Config has `mixed_precision: bf16`. Set it to `fp16`. |
| `uv sync` installs a CPU-only or wrong-CUDA PyTorch | Install PyTorch from the matching index first (section 2), then the other packages. |
| Edge list or cache looks stale after changing preprocessing | Delete `data/processed/edge_list.csv` and `data/processed/cached_dataset.pt`, then rerun the launcher. |
| Explainer returns all-zero node importances | Known issue (node-feature mask reads as zero). Rely on edge masks until fixed. |
