"""Evaluate saved token sequences with a small bag-of-tokens classifier.

The input NPZ file must contain:
  token_ids: integer array shaped [events, sequence_length]
  mask: boolean array with the same leading shape
  labels: integer class labels shaped [events]

This stays independent of the parquet schema so particle and jet tokenizers can
be compared with the same downstream metric.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BagOfTokensClassifier(nn.Module):
    def __init__(self, num_codes: int, num_classes: int):
        super().__init__()
        self.num_codes = num_codes
        self.classifier = nn.Sequential(
            nn.Linear(num_codes, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_ids = torch.where(mask, token_ids, 0)
        histogram = torch.zeros(
            token_ids.shape[0],
            self.num_codes,
            device=token_ids.device,
        )
        histogram.scatter_add_(1, token_ids, mask.float())
        histogram /= mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(histogram)


def _split_indices(labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        train_end = int(0.7 * len(indices))
        val_end = int(0.85 * len(indices))
        train.extend(indices[:train_end])
        val.extend(indices[train_end:val_end])
        test.extend(indices[val_end:])
    for indices in (train, val, test):
        rng.shuffle(indices)
    return train, val, test


def _loader(token_ids, mask, labels, indices, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(token_ids[indices]).long(),
        torch.from_numpy(mask[indices]).bool(),
        torch.from_numpy(labels[indices]).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for token_ids, mask, labels in loader:
            logits = model(token_ids.to(device), mask.to(device))
            labels = labels.to(device)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)
    return correct / total


def evaluate(path: Path, epochs: int, batch_size: int, seed: int) -> float:
    with np.load(path) as arrays:
        token_ids = arrays["token_ids"].astype(np.int64, copy=False)
        mask = arrays["mask"].astype(bool, copy=False)
        labels = arrays["labels"].astype(np.int64, copy=False)

    if token_ids.ndim == 3 and token_ids.shape[-1] == 1:
        token_ids = token_ids[..., 0]
    if token_ids.shape != mask.shape:
        raise ValueError("token_ids and mask must have matching shapes")
    if token_ids.shape[0] != labels.shape[0]:
        raise ValueError("labels must have one entry per event")
    if np.any(token_ids[mask] < 0):
        raise ValueError("Unmasked token IDs must be non-negative")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_codes = int(token_ids.max()) + 1
    _, labels = np.unique(labels, return_inverse=True)
    num_classes = int(labels.max()) + 1
    train_indices, val_indices, test_indices = _split_indices(labels, seed)
    train_loader = _loader(token_ids, mask, labels, train_indices, batch_size, True)
    val_loader = _loader(token_ids, mask, labels, val_indices, batch_size, False)
    test_loader = _loader(token_ids, mask, labels, test_indices, batch_size, False)

    model = BagOfTokensClassifier(num_codes, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for token_ids_batch, mask_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids_batch.to(device), mask_batch.to(device))
            loss = criterion(logits, labels_batch.to(device))
            loss.backward()
            optimizer.step()
        accuracy = _accuracy(model, val_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return _accuracy(model, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    accuracy = evaluate(args.tokens, args.epochs, args.batch_size, args.seed)
    print(f"test_accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
