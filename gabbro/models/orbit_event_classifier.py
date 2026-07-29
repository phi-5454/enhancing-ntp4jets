"""Five-class Transformer probe for paired ORBIT event representations."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import rankdata

from gabbro.data.orbit_downstream import FIVE_CLASS_TO_LABEL
from gabbro.models.classifiers import ClassifierTransformer
from gabbro.plotting.utils import set_mpl_style


CLASS_NAMES = tuple(FIVE_CLASS_TO_LABEL)


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target.astype(bool)
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def rejection_at_efficiency(target: np.ndarray, score: np.ndarray, efficiency: float) -> float:
    positive_scores = np.sort(score[target.astype(bool)])[::-1]
    if not len(positive_scores):
        return float("nan")
    index = min(max(int(np.ceil(efficiency * len(positive_scores))) - 1, 0), len(positive_scores) - 1)
    threshold = positive_scores[index]
    negative = ~target.astype(bool)
    false_positive_rate = float(np.mean(score[negative] >= threshold)) if np.any(negative) else 0.0
    return float("inf") if false_positive_rate == 0 else 1.0 / false_positive_rate


def roc_curve_points(target: np.ndarray, score: np.ndarray):
    order = np.argsort(score)[::-1]
    target = target.astype(bool)[order]
    positives = max(int(target.sum()), 1)
    negatives = max(len(target) - int(target.sum()), 1)
    return (
        np.r_[0.0, np.cumsum(~target) / negatives, 1.0],
        np.r_[0.0, np.cumsum(target) / positives, 1.0],
    )


def classification_metrics(targets: np.ndarray, logits: np.ndarray) -> dict:
    probabilities = softmax(logits, axis=1)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    recalls = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64),
        where=confusion.sum(axis=1) > 0,
    )
    aucs = {
        name: binary_auc(targets == index, probabilities[:, index])
        for index, name in enumerate(CLASS_NAMES)
    }
    confidence = probabilities.max(axis=1)
    correctness = predictions == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(correctness[selected])) - float(np.mean(confidence[selected]))
            )
    signal_index = FIVE_CLASS_TO_LABEL["ggHbb"]
    signal_target = targets == signal_index
    signal_score = probabilities[:, signal_index]
    return {
        "events": int(len(targets)),
        "accuracy": float(np.mean(correctness)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_auroc": float(np.nanmean(list(aucs.values()))),
        "per_class_auroc": aucs,
        "per_class_recall": dict(zip(CLASS_NAMES, map(float, recalls))),
        "ggHbb_rejection_at_50pct_efficiency": rejection_at_efficiency(
            signal_target, signal_score, 0.5
        ),
        "ggHbb_rejection_at_80pct_efficiency": rejection_at_efficiency(
            signal_target, signal_score, 0.8
        ),
        "expected_calibration_error": float(ece),
        "confusion_matrix": confusion.tolist(),
    }


class OrbitEventClassifierLightning(L.LightningModule):
    """Train on one paired representation and test on both representations."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_enc_blocks: int = 4,
        num_class_blocks: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = ClassifierTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_enc_blocks=num_enc_blocks,
            num_class_blocks=num_class_blocks,
            n_out_nodes=len(CLASS_NAMES),
            dropout_rate=dropout_rate,
            self_attention_model_class="Normformer",
            cross_attention_model_class="ClassAttentionBlock",
        )
        self._validation_outputs = []
        self._test_outputs = {0: [], 1: []}

    def forward(self, features, mask):
        return self.classifier(features, mask)

    def _step(self, batch):
        logits = self(batch["part_features"], batch["part_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        accuracy = (logits.argmax(dim=1) == batch["labels"]).float().mean()
        return loss, accuracy, logits

    def training_step(self, batch, batch_idx):
        loss, accuracy, _ = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        loss, accuracy, logits = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        self._validation_outputs.append(
            (batch["labels"].detach().cpu().numpy(), logits.detach().cpu().numpy())
        )

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return
        targets = np.concatenate([item[0] for item in self._validation_outputs])
        logits = np.concatenate([item[1] for item in self._validation_outputs])
        metrics = classification_metrics(targets, logits)
        self.log("val_macro_auroc", metrics["macro_auroc"], prog_bar=True, sync_dist=False)
        self.log("val_balanced_accuracy", metrics["balanced_accuracy"], sync_dist=False)

    def on_test_epoch_start(self):
        self._test_outputs = {0: [], 1: []}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy, logits = self._step(batch)
        representation = ("original", "decoded")[dataloader_idx]
        self.log(f"test_{representation}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"test_{representation}_accuracy", accuracy, on_step=False, on_epoch=True)
        self._test_outputs[dataloader_idx].append(
            {
                "targets": batch["labels"].detach().cpu().numpy(),
                "logits": logits.detach().cpu().numpy(),
                "event_ids": np.asarray(batch["event_ids"], dtype=object),
            }
        )

    def on_test_epoch_end(self):
        output_dir = Path(self.trainer.default_root_dir) / "downstream_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "trained_on": str(self.trainer.datamodule.hparams.train_representation),
            "class_to_label": FIVE_CLASS_TO_LABEL,
            "sequence_type": self.trainer.datamodule.metadata.get("sequence_type", "particle"),
            "max_sequence_length": int(
                self.trainer.datamodule.hparams.max_sequence_length
            ),
            "representations": {},
        }
        for dataloader_idx, representation in enumerate(("original", "decoded")):
            outputs = self._test_outputs[dataloader_idx]
            targets = np.concatenate([item["targets"] for item in outputs])
            logits = np.concatenate([item["logits"] for item in outputs])
            event_ids = np.concatenate([item["event_ids"] for item in outputs])
            summary["representations"][representation] = classification_metrics(targets, logits)
            np.savez_compressed(
                output_dir / f"test_{representation}_predictions.npz",
                event_ids=event_ids,
                targets=targets,
                logits=logits,
            )
            self._plot_roc(
                targets,
                logits,
                output_dir / f"test_{representation}_roc.png",
                f"Trained on {summary['trained_on']}, tested on {representation}",
            )
            self._plot_confusion(
                np.asarray(summary["representations"][representation]["confusion_matrix"]),
                output_dir / f"test_{representation}_confusion.png",
                f"Trained on {summary['trained_on']}, tested on {representation}",
            )
        (output_dir / "classifier_metrics.json").write_text(json.dumps(summary, indent=2))

    @staticmethod
    def _plot_confusion(confusion: np.ndarray, path: Path, title: str):
        set_mpl_style()
        normalized = np.divide(
            confusion,
            confusion.sum(axis=1, keepdims=True),
            out=np.zeros_like(confusion, dtype=np.float64),
            where=confusion.sum(axis=1, keepdims=True) > 0,
        )
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Row-normalized fraction")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)


    def _plot_roc(targets: np.ndarray, logits: np.ndarray, path: Path, title: str):
        set_mpl_style()
        probabilities = softmax(logits, axis=1)
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        for index, class_name in enumerate(CLASS_NAMES):
            false_positive_rate, true_positive_rate = roc_curve_points(
                targets == index, probabilities[:, index]
            )
            auc = binary_auc(targets == index, probabilities[:, index])
            axis.plot(false_positive_rate, true_positive_rate, label=f"{class_name} ({auc:.3f})")
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
