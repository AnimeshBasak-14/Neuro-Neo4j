"""
train.py
========
GNN Training Loop with early stopping, cosine LR scheduling,
rich progress logging, and dry-run support.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import time
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
    Monitor validation Macro-F1 and stop training if no improvement.

    Parameters
    ----------
    patience  : int    Epochs to wait before stopping
    min_delta : float  Minimum improvement to count as progress
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = -np.inf
        self.stopped_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Returns True if training should stop.

        Parameters
        ----------
        score : float  Current validation metric (higher = better)
        epoch : int    Current epoch index
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping counter: %d / %d (best=%.4f)",
                self.counter, self.patience, self.best_score,
            )
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


# ---------------------------------------------------------------------------
# GNNTrainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """
    Training orchestrator for BrainGNNModel.

    Parameters
    ----------
    model     : nn.Module
    optimizer : torch.optim.Optimizer
    criterion : nn.Module  Loss function (CrossEntropyLoss)
    device    : torch.device
    config    : dict  Training config from config.yaml
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    torch.device,
        config:    dict,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device
        self.config    = config

        train_cfg = config.get("training", config)
        self.epochs     = train_cfg.get("epochs", 100)
        self.ckpt_dir   = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.best_name  = train_cfg.get("best_model_filename", "best_model.pt")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict] = []
        logger.info(
            "GNNTrainer initialised | device=%s epochs=%d ckpt=%s",
            device, self.epochs, self.ckpt_dir,
        )

    # ------------------------------------------------------------------
    # Single epoch: training
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict:
        """
        Run one training epoch.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        dict with keys: loss, accuracy, macro_f1
        """
        self.model.train()
        total_loss = 0.0
        all_preds:  list[int] = []
        all_labels: list[int] = []
        all_probs:  list[list[float]] = []

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
                batch=batch.batch,
            )
            loss = self.criterion(logits, batch.y.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            probs  = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds  = probs.argmax(axis=1).tolist()
            labels = batch.y.view(-1).cpu().numpy().tolist()

            all_preds  += preds
            all_labels += labels
            all_probs  += probs.tolist()

        n = len(all_labels)
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )
        return {
            "loss":     total_loss / max(n, 1),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """
        Run validation pass.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        dict with keys: loss, accuracy, macro_f1, roc_auc
        """
        self.model.eval()
        total_loss = 0.0
        all_preds:  list[int] = []
        all_labels: list[int] = []
        all_probs:  list[list[float]] = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
                batch=batch.batch,
            )
            loss = self.criterion(logits, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=1).tolist()
            labels = batch.y.view(-1).cpu().numpy().tolist()
            all_preds  += preds
            all_labels += labels
            all_probs  += probs.tolist()

        n = len(all_labels)
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )
        return {
            "loss":     total_loss / max(n, 1),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc":  metrics["roc_auc"],
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       Optional[int] = None,
    ) -> list[dict]:
        """
        Train with early stopping and checkpoint saving.

        NOTE: This method performs real backpropagation.
              For dry-run, use dry_run() instead.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader   : DataLoader
        epochs       : int  Override config epochs

        Returns
        -------
        list[dict]  Training history
        """
        n_epochs = epochs or self.epochs
        train_cfg = self.config.get("training", self.config)
        patience = train_cfg.get("early_stopping_patience", 15)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs
        )
        stopper = EarlyStopping(patience=patience)
        best_f1 = -np.inf

        logger.info("Starting training for %d epochs...", n_epochs)
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate(val_loader)
            scheduler.step()

            epoch_log = {
                "epoch":      epoch,
                "train_loss": train_metrics["loss"],
                "train_f1":   train_metrics["macro_f1"],
                "val_loss":   val_metrics["loss"],
                "val_f1":     val_metrics["macro_f1"],
                "val_auc":    val_metrics["roc_auc"],
                "time_s":     time.time() - t0,
            }
            self.history.append(epoch_log)

            logger.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f f1=%.4f | "
                "val_loss=%.4f f1=%.4f auc=%.4f | "
                "%.1fs",
                epoch, n_epochs,
                train_metrics["loss"], train_metrics["macro_f1"],
                val_metrics["loss"], val_metrics["macro_f1"], val_metrics["roc_auc"],
                epoch_log["time_s"],
            )

            # Checkpoint
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                self.save_checkpoint(self.ckpt_dir / self.best_name)
                logger.info("  ✓ New best model saved (val_f1=%.4f)", best_f1)

            if stopper(val_metrics["macro_f1"], epoch):
                logger.info(
                    "Early stopping at epoch %d (patience=%d)", epoch, patience
                )
                break

        logger.info("Training complete. Best val Macro-F1: %.4f", best_f1)
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint IO
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to disk."""
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.debug("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Checkpoint loaded ← %s", path)

    # ------------------------------------------------------------------
    # Dry run (no backprop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dry_run(self, loader: DataLoader) -> None:
        """
        Run one forward pass without backpropagation.

        Validates that model, loader, and loss function are correctly wired.
        Prints tensor shapes and a dummy metric report.

        Parameters
        ----------
        loader : DataLoader
        """
        self.model.eval()
        batch = next(iter(loader))
        batch = batch.to(self.device)

        t0 = time.time()
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            batch=batch.batch,
        )
        loss   = self.criterion(logits, batch.y.view(-1))
        elapsed = time.time() - t0

        print("=" * 50)
        print("GNNTrainer — Dry Run")
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
