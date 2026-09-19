---
trigger: always_on
---

# Neuro-Neo4j & Tesla V100 Engineering Rules

## 1. Python & Environment
- **Strict Python 3.11**: Only use Python 3.11. Never use Python 3.10, 3.12, or 3.14.
- **Headless Server / No sudo**: The server is a headless shared Linux server without sudo privileges. NEVER propose or execute `sudo` commands. All execution and environment setup must be 100% user-space (`uv`, `.venv`, local paths).
- **Package Manager**: Use `uv` (or Conda) with `.venv` rooted at Python 3.11.
- **Neo4j Import Safety**: NEVER create any local directory or file named `neo4j/`. Local graph/ETL code MUST be placed in and imported from `neo4j_layer.*` so as not to shadow the official `neo4j` Python driver.

## 2. Hardware: Tesla V100 (Volta SM 7.0)
- **Precision (FP16 Only)**: NEVER use BF16 (`torch.bfloat16`). Volta lacks hardware BF16 tensor cores; software emulation destroys throughput. Always use FP16 (`torch.float16`) with `torch.cuda.amp.GradScaler()`.
- **Memory Pinning**: Always set `pin_memory=True` in PyTorch DataLoaders. The V100 uses dedicated HBM2 memory across a PCIe bus, so host-to-device transfers require pinned memory.
- **Workload Allocation**: The Tesla V100 is strictly reserved for model training, testing, and hyperparameter tuning. Production FastAPI inference/serving must run on local/clinical CPU compute.

## 3. PyG & Deep Learning Pipeline
- **GNNExplainer Gradients**: NEVER wrap `GNNExplainer` in `torch.no_grad()`. `GNNExplainer` optimizes soft masks via gradient descent and requires active gradients.
- **Explainer Target Parameter**: When `explanation_type='model'`, NEVER pass `target=`. PyG >= 2.4 strictly requires `target=None` for model explanation.
- **Dataset Interface**: NEVER inherit `BrainConnectomeDataset` from PyG's `InMemoryDataset` due to known PyG 2.8 `_data_list` reset issues. Always implement standard `torch.utils.data.Dataset` with `__len__` and `__getitem__`.
- **Mock Fallback**: Never require a live Neo4j database for unit tests or dry runs. Always fall back gracefully to the NetworkX mock mode when credentials or database instances are unavailable.


