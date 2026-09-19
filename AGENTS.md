# Neuro-Neo4j — Agent & Workspace Guidelines

These rules are strictly enforced for all development and operations on this repository and the Tesla V100 server:

### 1. Python & Package Environment
- **Python 3.11 Only**: Do not use Python 3.10, 3.12, or 3.14. Virtual environment is `.venv` managed with `uv` (or Conda `neuro_neo4j`).
- **No sudo / Headless Server**: This is a headless shared server without root/sudo access. Never run `sudo` commands. All dependencies, data, and processes must run strictly in user space (`.venv`, local paths).
- **No `neo4j/` directory**: Never name any directory or file `neo4j/` in the workspace root or sys.path. Python will shadow the official `neo4j` driver package (`from neo4j import GraphDatabase`), causing `ImportError` / `AttributeError`.
- **Imports**: All custom local ETL and graph code must be imported from `neo4j_layer.*`.

### 2. Tesla V100 Server (Volta Architecture)
- **FP16 Mixed Precision**: Volta does not support hardware BF16 (`bfloat16`). Never use `torch.bfloat16`. Always use `torch.float16` with `torch.cuda.amp.GradScaler()`.
- **DataLoader Pinned Memory**: V100 uses PCIe bus host-to-device transfers. Always set `pin_memory=True`.
- **GPU Allocation**: The Tesla V100 is strictly reserved for model training and offline testing. Production FastAPI serving runs on clinical/local CPU compute.

### 3. PyG & Deep Learning Pipeline
- **GNNExplainer Gradients**: Never wrap `GNNExplainer` in `torch.no_grad()`. It optimizes edge/node masks using gradient descent and requires gradients enabled.
- **Explainer Target**: Never pass `target=` when `explanation_type='model'`. PyG >= 2.4 strictly requires `target=None`.
- **Dataset Class**: Do not inherit `BrainConnectomeDataset` from PyG `InMemoryDataset`. Keep it implemented with standard `torch.utils.data.Dataset` (`__len__`, `__getitem__`) to avoid PyG 2.8 `_data_list` reset bugs.
- **Testing & Dry-Runs**: Offline testing must always work without requiring a live Neo4j database, falling back to internal NetworkX mock mode.
