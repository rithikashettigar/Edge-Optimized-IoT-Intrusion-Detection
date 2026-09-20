"""Training and evaluation loops shared by every model in the pipeline.

Written once and reused by the teacher, both students, and the pruning fine-tune,
rather than four near-copies that drift apart. `loss_fn` takes
(logits, targets, inputs), so plain cross-entropy and the distillation loss -
which needs the inputs to query the teacher - are interchangeable.
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader):
    """Return (y_true, y_pred, probability_of_the_attack_class)."""
    model.eval()
    trues, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x)
        trues.append(y.numpy())
        preds.append(logits.argmax(dim=1).numpy())
        probs.append(F.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(trues), np.concatenate(preds), np.concatenate(probs)


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    loss_fn,
    max_epochs: int,
    patience: int,
    lr: float,
    log_prefix: str = "",
) -> dict:
    """Train with early stopping on validation F1.

    F1 rather than accuracy, because benign flows outnumber attacks: a model that
    predicts 'benign' for everything scores well on accuracy and is useless.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict = {"train_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y, x)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        train_loss = running / max(seen, 1)

        y_true, y_pred, _ = evaluate(model, val_loader)
        val_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        if log_prefix:
            print(
                f"{log_prefix} epoch {epoch + 1:3d}  "
                f"loss {train_loss:.4f}  val_f1 {val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Return the best model seen, never merely the last epoch's.
    model.load_state_dict(best_state)
    history["best_val_f1"] = best_f1
    history["epochs_run"] = len(history["train_loss"])
    return history
