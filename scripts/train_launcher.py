"""
train_launcher.py
=================
Neuro-Neo4j — Central Training Launcher

Auto-detects hardware, loads the correct config, runs the full pipeline:
  1. Generate synthetic data  (if no real data present)
  2. ETL → edge list CSV
  3. Build BrainConnectomeDataset
  4. Train GNN (with optional k-fold CV)
  5. Evaluate on held-out test set
  6. Run GNNExplainer on all test subjects
  7. Save metrics, checkpoints, and biomarker reports

Usage
-----
# Auto-detect hardware and config:
    python scripts/train_launcher.py

# Specify config explicitly:
    python scripts/train_launcher.py --config config/v100_config.yaml

# Multi-GPU DDP on V100 server (4 GPUs):
    torchrun --standalone --nproc_per_node=4 scripts/train_launcher.py \\
             --config config/v100_config.yaml --ddp

Author: basakanimesh14
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_launcher")


# ─────────────────────────────────────────────────────────────────────────────
# Hardware auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> tuple[str, str]:
    """
    Detect GPU hardware and return (config_path, description).

    Detection order:
      1. CUDA available → check device name
         - "Blackwell" / "GB10"  → dgx_spark_config.yaml
         - "V100"                → v100_config.yaml
         - "Orin" / "Xavier"    → agx_config.yaml
         - A100 / H100 fallback → dgx_config.yaml
      2. No CUDA              → cpu (dry-run only)
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected — falling back to CPU (dry-run mode).")
        return str(PROJECT_ROOT / "config" / "config.yaml"), "CPU"

    name = torch.cuda.get_device_name(0).lower()
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)

    logger.info("GPU detected: %s | VRAM: %.1f GB | CUDA cap: %d.%d",
                torch.cuda.get_device_name(0), total_gb,
                props.major, props.minor)

    # Blackwell (arch >= 10.0, or name contains gb/blackwell)
    if props.major >= 10 or any(k in name for k in ["gb10", "blackwell", "b100", "b200"]):
        cfg = "dgx_spark_config.yaml"
        desc = "DGX Spark (Blackwell GB10)"
    elif "v100" in name:
        cfg = "v100_config.yaml"
        desc = f"Tesla V100 ({total_gb:.0f} GB)"
    elif any(k in name for k in ["orin", "xavier", "agx"]):
        cfg = "agx_config.yaml"
        desc = "AGX Orin"
    elif any(k in name for k in ["a100", "h100", "h200"]):
        cfg = "dgx_config.yaml"
        desc = "DGX A100/H100"
    else:
        cfg = "config.yaml"
        desc = f"Generic GPU ({torch.cuda.get_device_name(0)})"

    return str(PROJECT_ROOT / "config" / cfg), desc


# ─────────────────────────────────────────────────────────────────────────────
# Mixed precision scaler
# ─────────────────────────────────────────────────────────────────────────────

def get_scaler(mixed_precision: str):
    """Return GradScaler for fp16, None for bf16 (no scaler needed)."""
    if mixed_precision == "fp16":
        return torch.cuda.amp.GradScaler()
    return None  # bf16 is numerically stable, no scaler needed


def get_autocast_dtype(mixed_precision: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map.get(mixed_precision, torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def step_generate_synthetic(cfg: dict) -> None:
    n_subjects = cfg["data"].get("n_synthetic_subjects", 100)
    conn_dir   = Path(cfg["data"]["connectomes_dir"])

    if conn_dir.exists() and len(list(conn_dir.glob("*.npy"))) >= n_subjects:
        logger.info("Synthetic data already exists (%d subjects). Skipping generation.", n_subjects)
        return

    logger.info("Generating %d synthetic subjects...", n_subjects)
    conn_dir.mkdir(parents=True, exist_ok=True)

    from data.synthetic.generate_synthetic import (
        AAL116_REGIONS, make_connectome
    )
    import numpy as np

    diagnoses = ["Healthy", "Diseased"]
    sites     = ["site_A", "site_B", "site_C"]
    rng       = np.random.default_rng(42)
    metadata  = []

    for i in range(n_subjects):
        diagnosis = diagnoses[i % 2]
        site      = sites[i % 3]
        subject   = f"sub_{i:04d}"
        matrix    = make_connectome(
            diagnosis=diagnosis,
            scanner_site=site,
            seed=i,
        )
        out_path = conn_dir / f"{subject}_{diagnosis}_{site}.npy"
        np.save(out_path, matrix)
        metadata.append({
            "PatientID":   subject,
            "Diagnosis":   diagnosis,
            "Label":       0 if diagnosis == "Healthy" else 1,
            "Age":         int(rng.integers(50, 85)),
            "ScannerSite": site,
        })

    import pandas as pd
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_csv(processed_dir / "subjects.csv", index=False)
    logger.info("Generated %d subjects → %s", n_subjects, processed_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: ETL — connectomes → edge list CSV
# ─────────────────────────────────────────────────────────────────────────────

def step_etl(cfg: dict) -> None:
    edge_list_path = Path(cfg["data"]["edge_list_file"])

    if edge_list_path.exists():
        logger.info("Edge list already exists at %s. Skipping ETL.", edge_list_path)
        return

    logger.info("Running ETL pipeline...")
    from neo4j_layer.etl.transform import ConnectomeTransformer
    from data.synthetic.generate_synthetic import AAL116_REGIONS
    import numpy as np
    import pandas as pd

    conn_dir   = Path(cfg["data"]["connectomes_dir"])
    subj_df    = pd.read_csv(Path(cfg["data"]["subject_metadata_file"]))
    transformer = ConnectomeTransformer(region_names=AAL116_REGIONS)
    all_edges  = []

    for npy_file in sorted(conn_dir.glob("*.npy")):
        parts     = npy_file.stem.split("_")
        subject   = "_".join(parts[:2])
        diagnosis = parts[2] if len(parts) > 2 else "Unknown"

        matrix = np.load(npy_file)
        matrix = transformer.regress_global_signal(matrix)
        matrix = transformer.normalize(matrix, method="zscore")
        matrix = transformer.threshold(matrix, threshold=0.3)
        edges  = transformer.to_edge_list(matrix, subject_id=subject, diagnosis=diagnosis)
        all_edges.append(edges)

    edge_df = pd.concat(all_edges, ignore_index=True)
    edge_list_path.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(edge_list_path, index=False)
    logger.info("ETL complete: %d edges → %s", len(edge_df), edge_list_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build dataset and splits
# ─────────────────────────────────────────────────────────────────────────────

def step_build_dataset(cfg: dict):
    from models.gnn.dataset import BrainConnectomeDataset
    from validation.subject_split import subject_level_split
    from torch_geometric.loader import DataLoader

    ds = BrainConnectomeDataset(
        root=cfg["data"]["processed_dir"],
        subjects_csv=cfg["data"]["subject_metadata_file"],
        edge_list_csv=cfg["data"]["edge_list_file"],
    )
    logger.info("Dataset: %d graphs", len(ds))

    ratios  = cfg["validation"]["split_ratios"]
    seed    = cfg["validation"]["split_seed"]
    train_idx, val_idx, test_idx = subject_level_split(
        dataset=ds,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        seed=seed,
    )

    train_ds = [ds[i] for i in train_idx]
    val_ds   = [ds[i] for i in val_idx]
    test_ds  = [ds[i] for i in test_idx]

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 4)
    pm = cfg["training"].get("pin_memory", False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)

    logger.info("Splits — train: %d | val: %d | test: %d",
                len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader, ds


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 + 5: Train + Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def step_train_evaluate(cfg: dict, train_loader, val_loader, test_loader, device):
    from models.gnn.gcn_gat_model import BrainGNNModel
    from models.gnn.train import GNNTrainer
    from models.metrics import MetricsReport

    mc = cfg["model"]
    model = BrainGNNModel(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        num_classes=mc["num_classes"],
        num_heads=mc["num_heads"],
        dropout=mc["dropout"],
        num_gcn_layers=mc["num_gcn_layers"],
        num_gat_layers=mc["num_gat_layers"],
        use_swiglu=mc.get("use_swiglu", True),
        use_gatv2=mc.get("use_gatv2", True),
        use_edge_attr=mc.get("use_edge_attr", True),
        use_dual_pooling=mc.get("use_dual_pooling", True),
    ).to(device)

    tc        = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tc["learning_rate"],
                                  weight_decay=tc["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=tc.get("label_smoothing", 0.0)
    )

    # Mixed precision setup
    mp         = tc.get("mixed_precision", "fp32")
    scaler     = get_scaler(mp)
    amp_dtype  = get_autocast_dtype(mp)

    ckpt_dir   = Path(tc["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = ckpt_dir / tc["best_model_filename"]

    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=cfg,
        scaler=scaler,
        amp_dtype=amp_dtype,
    )

    logger.info("Starting training — %d epochs | device=%s | precision=%s",
                tc["epochs"], device, mp)
    t0 = time.time()
    trainer.fit(train_loader, val_loader,
                epochs=tc["epochs"],
                early_stopping_patience=tc["early_stopping_patience"],
                checkpoint_path=str(ckpt_path))
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # Load best checkpoint and evaluate on test set
    trainer.load_checkpoint(str(ckpt_path))
    test_report = trainer.evaluate(test_loader, split="test")
    logger.info("\n%s", test_report)

    # Save test metrics
    log_dir = Path(tc["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_report.to_dict(), f, indent=2)
    logger.info("Test metrics saved → %s", metrics_path)

    return model, test_report


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: GNNExplainer biomarker extraction
# ─────────────────────────────────────────────────────────────────────────────

def step_explain(cfg: dict, model, test_loader, device):
    from models.gnn.explainer import BrainGNNExplainer
    from data.synthetic.generate_synthetic import AAL116_REGIONS

    explainer = BrainGNNExplainer(model=model, config=cfg)
    log_dir   = Path(cfg["training"]["log_dir"])
    all_results = []

    logger.info("Running GNNExplainer on test set...")
    model.eval()

    for batch in test_loader:
        batch = batch.to(device)
        # Explain each graph in the batch individually
        for i in range(batch.num_graphs):
            data_i = batch.get_example(i)
            try:
                result = explainer.explain_graph(
                    data_i,
                    target_class=int(data_i.y.item()),
                    region_names=AAL116_REGIONS[:data_i.num_nodes],
                )
                all_results.append({
                    "subject_id":   getattr(data_i, "subject_id", f"subj_{i}"),
                    "target_class": result.target_class,
                    "top_regions":  result.top_regions_df.to_dict("records"),
                    "top_edges":    result.top_edges_df.head(10).to_dict("records"),
                })
            except Exception as exc:
                logger.warning("Explainer failed for subject %d: %s", i, exc)

    if all_results:
        import json
        out = log_dir / "biomarker_explanations.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Biomarker explanations saved → %s (%d subjects)", out, len(all_results))


# ─────────────────────────────────────────────────────────────────────────────
# Dry run: self-contained smoke test (CPU, no database, no real data)
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(config_path: str | None = None) -> bool:
    """
    Offline smoke test of the main components. Uses only in-memory synthetic
    data, so it needs neither Neo4j, a GPU, nor the files in data/processed.
    Returns True if every check passes.
    """
    import numpy as np
    import pandas as pd

    results: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            detail = fn()
            results.append((name, True, str(detail) if detail else ""))
        except Exception as exc:  # report every failure, do not stop at the first
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def t_config():
        path = config_path or str(PROJECT_ROOT / "config" / "config.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg and "training" in cfg, "config needs model and training sections"
        return path

    def t_synthetic_etl():
        from data.synthetic.generate_synthetic import AAL116_REGIONS, make_connectome
        from neo4j_layer.etl.transform import ConnectomeTransformer
        m = make_connectome("Diseased", "site_B", seed=1)
        assert m.shape == (116, 116)
        tr = ConnectomeTransformer(AAL116_REGIONS)
        w = tr.threshold(tr.normalize(tr.regress_global_signal(m)), threshold=0.3)
        edges = tr.to_edge_list(w, "sub_test", "Diseased")
        assert len(edges) > 1000, f"only {len(edges)} edges kept"
        return f"{len(edges)} edges"

    def t_model_forward():
        from models.gnn.gcn_gat_model import BrainGNNModel
        mc = {"in_channels": 5, "hidden_channels": 32, "num_classes": 2, "num_heads": 4,
              "dropout": 0.1, "num_gcn_layers": 2, "num_gat_layers": 1}
        model = BrainGNNModel(**mc).eval()
        x = torch.randn(20, 5)
        ei = torch.randint(0, 20, (2, 80))
        ea = torch.randn(80, 1)
        out = model(x, ei, ea, torch.zeros(20, dtype=torch.long))
        assert out.shape == (1, 2)
        return f"{model.count_parameters():,} params"

    def t_split():
        from validation.subject_split import subject_level_split
        df = pd.DataFrame({"PatientID": [f"s{i}" for i in range(100)],
                           "Diagnosis": ["Healthy", "Diseased"] * 50})
        tr_, va_, te_ = subject_level_split(subjects_df=df, ratios=(0.7, 0.15, 0.15), random_seed=42)
        ids = [set(d.PatientID) for d in (tr_, va_, te_)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "subject leakage"
        return f"{len(tr_)}/{len(va_)}/{len(te_)}"

    def t_metrics():
        from models.metrics import MetricsReport
        rng = np.random.default_rng(0)
        y = np.array([0, 1] * 20)
        p = rng.random((40, 2)); p /= p.sum(1, keepdims=True)
        rep = MetricsReport.from_arrays(y_true=y, y_pred=p.argmax(1), y_scores=p, split="smoke")
        return f"AUC {rep.roc_auc:.3f}"

    for name, fn in [("config loads", t_config), ("synthetic cohort + ETL", t_synthetic_etl),
                     ("model forward pass", t_model_forward), ("patient-level split", t_split),
                     ("metrics report", t_metrics)]:
        check(name, fn)

    ok = all(r[1] for r in results)
    for name, passed, detail in results:
        logger.info("%-26s %s  %s", name, "PASS" if passed else "FAIL", detail)
    logger.info("Smoke test: %s", "ALL PASSED" if ok else "FAILURES FOUND")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Neuro-Neo4j Training Launcher")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (auto-detected if omitted)")
    p.add_argument("--ddp", action="store_true",
                   help="Enable DistributedDataParallel (multi-GPU, run via torchrun)")
    p.add_argument("--skip-data-gen", action="store_true",
                   help="Skip synthetic data generation")
    p.add_argument("--skip-etl", action="store_true",
                   help="Skip ETL (edge list already exists)")
    p.add_argument("--skip-explain", action="store_true",
                   help="Skip GNNExplainer after training")
    p.add_argument("--dry-run", action="store_true",
                   help="Run smoke tests only (no training)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Dry run shortcut ──────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry-run mode — running smoke tests.")
        sys.exit(0 if run_smoke_test(args.config) else 1)

    # ── Config resolution ─────────────────────────────────────────────────────
    if args.config:
        config_path = args.config
        hw_desc     = "user-specified"
    else:
        config_path, hw_desc = detect_hardware()

    logger.info("Hardware: %s | Config: %s", hw_desc, config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # fastest convolution for fixed input size

    # ── DDP init (torchrun sets LOCAL_RANK automatically) ─────────────────────
    if args.ddp and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info("DDP rank %d initialised on %s", local_rank, device)

    is_main = not args.ddp or int(os.environ.get("LOCAL_RANK", 0)) == 0

    # ── Pipeline ──────────────────────────────────────────────────────────────
    if is_main:
        if not args.skip_data_gen:
            step_generate_synthetic(cfg)
        if not args.skip_etl:
            step_etl(cfg)

    if args.ddp:
        torch.distributed.barrier()  # wait for data to be ready on all ranks

    train_loader, val_loader, test_loader, ds = step_build_dataset(cfg)
    model, test_report = step_train_evaluate(
        cfg, train_loader, val_loader, test_loader, device
    )

    if is_main and not args.skip_explain:
        step_explain(cfg, model, test_loader, device)

    if args.ddp:
        torch.distributed.destroy_process_group()

    if is_main:
        logger.info("=" * 55)
        logger.info("TRAINING COMPLETE")
        logger.info("  Accuracy  : %.4f", test_report.accuracy)
        logger.info("  Macro-F1  : %.4f", test_report.macro_f1)
        logger.info("  ROC-AUC   : %.4f", test_report.roc_auc)
        logger.info("=" * 55)


if __name__ == "__main__":
    main()
