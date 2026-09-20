"""Shared test fixtures.

Every training-related test needs a small, linearly separable dataset shaped like
the real thing: two classes, a handful of numeric features, split three ways.
Defined once here rather than copied into each test module.
"""

import numpy as np
import pytest

from edge_ids.data.preprocess import Splits


def make_separable_splits(n=400, d=6, seed=0, attack_rate=0.5):
    """Two Gaussian populations offset along every feature - easy but not trivial."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < attack_rate).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    hold = n // 4
    return Splits(
        X[: n - 2 * hold], y[: n - 2 * hold],
        X[n - 2 * hold : n - hold], y[n - 2 * hold : n - hold],
        X[n - hold :], y[n - hold :],
        [f"f{i}" for i in range(d)],
        None,
    )


@pytest.fixture
def separable_splits():
    """Factory fixture: call it with overrides, e.g. separable_splits(n=200)."""
    return make_separable_splits
