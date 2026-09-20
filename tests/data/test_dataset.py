import numpy as np
import torch

from edge_ids.data.dataset import class_weights, make_loaders
from edge_ids.data.preprocess import Splits


def _splits(n=100, d=4):
    rng = np.random.default_rng(0)
    f = lambda k: rng.normal(size=(k, d)).astype(np.float32)
    g = lambda k: rng.integers(0, 2, k).astype(np.int64)
    return Splits(
        f(n), g(n), f(n // 2), g(n // 2), f(n // 2), g(n // 2),
        [f"f{i}" for i in range(d)], None,
    )


def test_loaders_yield_correct_dtypes_and_shapes():
    train, _, _ = make_loaders(_splits(), batch_size=16, seed=42)
    x, y = next(iter(train))
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert x.shape[1] == 4


def test_class_weights_upweight_the_minority_class():
    weights = class_weights(np.array([0] * 90 + [1] * 10))
    assert weights[1] > weights[0]


def test_class_weights_are_equal_for_a_balanced_split():
    weights = class_weights(np.array([0] * 50 + [1] * 50))
    assert torch.allclose(weights[0], weights[1])


def test_class_weights_survive_a_missing_class():
    weights = class_weights(np.zeros(10, dtype=np.int64))
    assert torch.isfinite(weights).all()


def test_validation_loader_preserves_order():
    splits = _splits()
    _, val, _ = make_loaders(splits, batch_size=1000, seed=42)
    x, _ = next(iter(val))
    assert np.allclose(x.numpy(), splits.X_val)


def test_trailing_batch_of_one_is_dropped():
    # BatchNorm1d raises on a single-row batch in train mode; 33 rows at batch 16
    # would otherwise leave exactly one row in the final batch.
    splits = _splits(n=33)
    train, _, _ = make_loaders(splits, batch_size=16, seed=42)
    assert all(len(y) > 1 for _, y in train)


def test_no_samples_are_dropped_when_the_last_batch_is_safe():
    splits = _splits(n=34)
    train, _, _ = make_loaders(splits, batch_size=16, seed=42)
    assert sum(len(y) for _, y in train) == 34


def test_train_loader_shuffles():
    splits = _splits()
    train, _, _ = make_loaders(splits, batch_size=1000, seed=42)
    x, _ = next(iter(train))
    assert not np.allclose(x.numpy(), splits.X_train)
