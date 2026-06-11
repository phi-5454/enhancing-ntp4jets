"""Export token IDs, mask, and labels to NPZ for downstream evaluation."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np

from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger("TokenExportCallback")


class TokenExportCallback(L.Callback):
    """Write token_ids / mask / labels NPZ consumable by evaluate_token_quality.py."""

    def __init__(self, export_val: bool = False, export_test: bool = True):
        super().__init__()
        self.export_val = export_val
        self.export_test = export_test

    def _export(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        code_idx_attr = f"{stage}_code_idx_concat"
        mask_attr = f"{stage}_mask_concat"
        labels_attr = f"{stage}_labels_concat"

        if not hasattr(pl_module, code_idx_attr):
            logger.warning(f"No {code_idx_attr} on module — skipping token export for {stage}.")
            return

        code_idx = getattr(pl_module, code_idx_attr)
        mask = getattr(pl_module, mask_attr).astype(bool)
        labels = getattr(pl_module, labels_attr)

        out_dir = Path(trainer.default_root_dir) / "saved_tokens"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stage}_tokens_step_{trainer.global_step}.npz"
        np.savez_compressed(path, token_ids=code_idx, mask=mask, labels=labels)
        logger.info(f"Saved tokens to {path}")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not self.export_val:
            return
        self._export(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.export_test:
            return
        self._export(trainer, pl_module, "test")
