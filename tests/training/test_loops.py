import torch
import torch.nn.functional as F

from conftest import make_separable_splits as separable_splits
from edge_ids.data.dataset import make_loaders
from edge_ids.models.student import StudentMLP
from edge_ids.training.loops import evaluate, set_seed, train_model


def _ce(logits, targets, _inputs):
    return F.cross_entropy(logits, targets)


def test_training_reduces_loss_and_learns_separable_data():
    set_seed(42)
    splits = separable_splits()
    train, val, test = make_loaders(splits, batch_size=32, seed=42)
    model = StudentMLP(6, hidden=(8, 4))
    history = train_model(
        model, train, val, loss_fn=_ce, max_epochs=15, patience=5, lr=1e-2
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    y_true, y_pred, _ = evaluate(model, test)
    assert (y_true == y_pred).mean() > 0.85


def test_early_stopping_halts_before_max_epochs():
    set_seed(42)
    train, val, _ = make_loaders(separable_splits(), batch_size=32, seed=42)
    model = StudentMLP(6, hidden=(8, 4))
    history = train_model(
        model, train, val, loss_fn=_ce, max_epochs=100, patience=2, lr=1e-2
    )
    assert len(history["train_loss"]) < 100


def test_best_weights_are_restored_not_the_last_epoch():
    set_seed(42)
    train, val, _ = make_loaders(separable_splits(), batch_size=32, seed=42)
    model = StudentMLP(6, hidden=(8, 4))
    history = train_model(
        model, train, val, loss_fn=_ce, max_epochs=12, patience=12, lr=1e-2
    )
    from sklearn.metrics import f1_score

    y_true, y_pred, _ = evaluate(model, val)
    assert f1_score(y_true, y_pred, zero_division=0) == history["best_val_f1"]


def test_evaluate_returns_probabilities_in_unit_range():
    _, _, test = make_loaders(separable_splits(), batch_size=32, seed=42)
    _, _, prob = evaluate(StudentMLP(6, hidden=(8, 4)), test)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_evaluate_returns_one_entry_per_sample():
    splits = separable_splits()
    _, _, test = make_loaders(splits, batch_size=32, seed=42)
    y_true, y_pred, prob = evaluate(StudentMLP(6, hidden=(8, 4)), test)
    assert len(y_true) == len(y_pred) == len(prob) == len(splits.y_test)


def test_set_seed_makes_initialization_reproducible():
    set_seed(1)
    first = StudentMLP(6, hidden=(8, 4)).net[0].weight.clone()
    set_seed(1)
    second = StudentMLP(6, hidden=(8, 4)).net[0].weight.clone()
    assert torch.allclose(first, second)
