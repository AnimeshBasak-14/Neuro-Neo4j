"""
train.py
========
GNN Training Loop — production-ready with:
  - Mixed precision (FP16 via GradScaler, BF16 native)
  - Cosine LR scheduling with linear warmup
  - Early stopping on validation Macro-F1
  - Checkpoint save/load
  - Full evaluate() returning MetricsReport
  - Dry-run (no backprop) for smoke testing

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.loader import DataLoader

from models.metrics import MetricsReport, compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor validation Macro-F1; stop if no improvement for `patience` epochs.

    Parameters
    ----------
    patience  : int    Epochs to wait without improvement before stopping
    min_delta : float  Minimum improvement counted as progress
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience      = patience
        self.min_delta     = min_delta
        self.counter       = 0
        self.best_score    = -np.inf
        self.stopped_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.debug("EarlyStopping %d/%d (best=%.4f)", self.counter, self.patience, self.best_score)
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup → cosine annealing."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GNNTrainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """
    Training orchestrator for BrainGNNModel.

    Parameters
    ----------
    model          : nn.Module
    optimizer      : torch.optim.Optimizer
    criterion      : nn.Module             CrossEntropyLoss (with label_smoothing)
    device         : torch.device
    config         : dict                  Full pipeline config dict
    scaler         : GradScaler | None     For FP16; None for BF16/FP32
    amp_dtype      : torch.dtype           torch.float16 | torch.bfloat16 | torch.float32
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    torch.device,
        config:    dict,
        scaler:    Optional[torch.cuda.amp.GradScaler] = None,
        amp_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device
        self.config    = config
        self.scaler    = scaler
        self.amp_dtype = amp_dtype

        train_cfg       = config.get("training", config)
        self.epochs     = train_cfg.get("epochs", 100)
        self.ckpt_dir   = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.best_name  = train_cfg.get("best_model_filename", "best_model.pt")
        self.grad_clip  = train_cfg.get("gradient_clip", 1.0)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict] = []
        logger.info(
            "GNNTrainer initialised | device=%s epochs=%d precision=%s ckpt=%s",
            device, self.epochs,
            "fp16" if scaler else ("bf16" if amp_dtype == torch.bfloat16 else "fp32"),
            self.ckpt_dir,
        )

    # ------------------------------------------------------------------
    # AMP context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _amp_ctx(self):
        """Mixed-precision autocast context (GPU only)."""
        if self.device.type == "cuda" and self.amp_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                yield
        else:
            yield

    def _backward(self, loss: Tensor) -> None:
        """Backward pass — GradScaler-aware for FP16."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Forward helper
    # ------------------------------------------------------------------

    def _forward(self, batch) -> Tensor:
        return self.model(
            x          = batch.x,
            edge_index = batch.edge_index,
            edge_attr  = getattr(batch, "edge_attr", None),
            batch      = batch.batch,
        )

    # ------------------------------------------------------------------
    # Single epoch: training
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict:
        """One training epoch. Returns loss / accuracy / macro_f1."""
        self.model.train()
        total_loss  = 0.0
        all_preds:  list[int]         = []
        all_labels: list[int]         = []
        all_probs:  list[list[float]] = []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self._amp_ctx():
                logits = self._forward(batch)
                loss   = self.criterion(logits, batch.y.view(-1))

            self._backward(loss)

            total_loss += loss.item() * batch.num_graphs
            probs   = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
            preds   = probs.argmax(axis=1).tolist()
            labels  = batch.y.view(-1).cpu().numpy().tolist()
            all_preds  += preds
            all_labels += labels
            all_probs  += probs.tolist()

        n       = max(len(all_labels), 1)
        metrics = compute_all_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        return {"loss": total_loss / n, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Validation pass. Returns loss / accuracy / macro_f1 / roc_auc."""
        self.model.eval()
        total_loss  = 0.0
        all_preds:  list[int]         = []
        all_labels: list[int]         = []
        all_probs:  list[list[float]] = []

        for batch in loader:
            batch = batch.to(self.device)
            with self._amp_ctx():
                logits = self._forward(batch)
                loss   = self.criterion(logits, batch.y.view(-1))

            total_loss += loss.item() * batch.num_graphs
            probs   = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_preds  += probs.argmax(axis=1).tolist()
            all_labels += batch.y.view(-1).cpu().numpy().tolist()
            all_probs  += probs.tolist()

        n       = max(len(all_labels), 1)
        metrics = compute_all_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        return {
            "loss":     total_loss / n,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc":  metrics["roc_auc"],
        }

    # ------------------------------------------------------------------
    # Evaluate (test set) → MetricsReport
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "test") -> MetricsReport:
        """
        Full evaluation pass returning a MetricsReport.

        Used after training completes to assess held-out test performance.

        Parameters
        ----------
        loader : DataLoader  Test DataLoader
        split  : str         Label for the report ("test" / "val")

        Returns
        -------
        MetricsReport
        """
        self.model.eval()
        all_preds:  list[int]         = []
        all_labels: list[int]         = []
        all_probs:  list[list[float]] = []

        for batch in loader:
            batch  = batch.to(self.device)
            with self._amp_ctx():
                logits = self._forward(batch)

            probs   = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_preds  += probs.argmax(axis=1).tolist()
            all_labels += batch.y.view(-1).cpu().numpy().tolist()
            all_probs  += probs.tolist()

        report = MetricsReport.from_arrays(
            y_true   = np.array(all_labels),
            y_pred   = np.array(all_preds),
            y_scores = np.array(all_probs),
            split    = split,
        )
        logger.info("Evaluation [%s] — Acc=%.4f | Macro-F1=%.4f | ROC-AUC=%.4f",
                    split, report.accuracy, report.macro_f1, report.roc_auc)
        return report

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader:             DataLoader,
        val_loader:               DataLoader,
        epochs:                   Optional[int] = None,
        early_stopping_patience:  Optional[int] = None,
        checkpoint_path:          Optional[str] = None,
    ) -> list[dict]:
        """
        Train with warmup-cosine LR, early stopping, and best-checkpoint saving.

        Parameters
        ----------
        train_loader            : DataLoader
        val_loader              : DataLoader
        epochs                  : int  Override config epochs (optional)
        early_stopping_patience : int  Override config patience (optional)
        checkpoint_path         : str  Explicit path for best model (optional)

        Returns
        -------
        list[dict]  Epoch-by-epoch training history
        """
        n_epochs  = epochs or self.epochs
        train_cfg = self.config.get("training", self.config)
        patience  = early_stopping_patience or train_cfg.get("early_stopping_patience", 15)
        warmup    = train_cfg.get("warmup_epochs", 0)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else (self.ckpt_dir / self.best_name)

        scheduler = build_scheduler(self.optimizer, warmup, n_epochs)
        stopper   = EarlyStopping(patience=patience)
        best_f1   = -np.inf

        logger.info("Training: %d epochs | patience=%d | warmup=%d | ckpt=%s",
                    n_epochs, patience, warmup, ckpt_path)

        for epoch in range(1, n_epochs + 1):
            t0            = time.time()
            train_m       = self.train_epoch(train_loader)
            val_m         = self.validate(val_loader)
            scheduler.step()

            epoch_log = {
                "epoch":      epoch,
                "lr":         self.optimizer.param_groups[0]["lr"],
                "train_loss": train_m["loss"],
                "train_f1":   train_m["macro_f1"],
                "val_loss":   val_m["loss"],
                "val_f1":     val_m["macro_f1"],
                "val_auc":    val_m["roc_auc"],
                "time_s":     time.time() - t0,
            }
            self.history.append(epoch_log)

            logger.info(
                "Epoch %3d/%d | lr=%.2e | "
                "train[loss=%.4f f1=%.4f] | "
                "val[loss=%.4f f1=%.4f auc=%.4f] | %.1fs",
                epoch, n_epochs, epoch_log["lr"],
                train_m["loss"], train_m["macro_f1"],
                val_m["loss"], val_m["macro_f1"], val_m["roc_auc"],
                epoch_log["time_s"],
            )

            # Save best checkpoint
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                self.save_checkpoint(ckpt_path)
                logger.info("  >> Best checkpoint (val_f1=%.4f) saved to %s", best_f1, ckpt_path)

            if stopper(val_m["macro_f1"], epoch):
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        logger.info("Training complete. Best val Macro-F1: %.4f", best_f1)
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint IO
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path | str) -> None:
        """Save model + optimizer state to disk."""
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history":              self.history,
        }, path)
        logger.debug("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model + optimizer state from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info("Checkpoint loaded <- %s", path)

    # ------------------------------------------------------------------
    # Dry run (no backprop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dry_run(self, loader: DataLoader) -> None:
        """
        Single forward pass only — validates wiring without backprop.
        Safe to call on CPU with no real data.
        """
        self.model.eval()
        batch = next(iter(loader))
        batch = batch.to(self.device)

        t0     = time.time()
        logits = self._forward(batch)
        loss   = self.criterion(logits, batch.y.view(-1))
        elapsed = time.time() - t0

        print("=" * 50)
        print("GNNTrainer -- Dry Run")
        print("=" * 50)
        print(f"  Batch size     : {batch.num_graphs}")
        print(f"  Node features  : {batch.x.shape}")
        print(f"  Edge index     : {batch.edge_index.shape}")
        print(f"  Logits         : {logits.shape}")
        print(f"  Labels         : {batch.y.view(-1).tolist()}")
        print(f"  Dummy loss     : {loss.item():.4f}")
        print(f"  Forward time   : {elapsed*1000:.2f} ms")
        print("  No backprop executed (dry run)")
        print("=" * 50)
