"""Wrap the numpy splits into torch DataLoaders."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)).float(),
        torch.from_numpy(np.ascontiguousarray(y)).long(),
    )


def make_loaders(splits, batch_size: int, seed: int = 42):
    """Train loader is shuffled; val and test preserve order so results are comparable."""
    generator = torch.Generator().manual_seed(seed)
    train_ds = _tensor_dataset(splits.X_train, splits.y_train)
    # BatchNorm1d raises in train mode on a batch of one. Whether the final batch
    # has a single row depends on how many rows survived cleaning, so drop it in
    # exactly that case - costing one sample instead of the whole run.
    drop_last = len(train_ds) % batch_size == 1
    train = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )
    val = DataLoader(
        _tensor_dataset(splits.X_val, splits.y_val), batch_size=batch_size, shuffle=False
    )
    test = DataLoader(
        _tensor_dataset(splits.X_test, splits.y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train, val, test


def class_weights(y) -> torch.Tensor:
    """Inverse-frequency weights. Benign flows outnumber attacks roughly four to one,
    so unweighted training would happily ignore the minority class."""
    counts = np.bincount(np.asarray(y), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0  # a class absent from this split must not divide by zero
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
