"""Train the over-parameterized teacher with class-weighted cross-entropy."""

import torch.nn.functional as F

from ..data.dataset import class_weights, make_loaders
from ..models.teacher import TeacherMLP
from .loops import set_seed, train_model


def train_teacher(splits, cfg):
    """Return (trained teacher, history)."""
    set_seed(cfg.data.seed)
    tc = cfg.teacher

    train_loader, val_loader, _ = make_loaders(splits, tc.batch_size, cfg.data.seed)
    weights = class_weights(splits.y_train)
    model = TeacherMLP(
        splits.n_features, hidden=tuple(tc.hidden), dropout=tc.dropout
    )

    def loss_fn(logits, targets, _inputs):
        return F.cross_entropy(logits, targets, weight=weights)

    history = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn=loss_fn,
        max_epochs=tc.max_epochs,
        patience=tc.patience,
        lr=tc.lr,
        log_prefix="[teacher]",
    )
    return model, history
