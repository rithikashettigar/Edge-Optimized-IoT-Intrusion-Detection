"""Structured pruning that physically rebuilds the network.

torch.nn.utils.prune applies a binary mask: the zeroed weights still occupy full
FP32 tensors, so parameter count and file size are unchanged. A pipeline that
reports "50% pruned" from masked weights and then reports an unchanged file size
is reporting a result it did not obtain.

So instead of masking, the surviving neurons are copied into genuinely smaller
Linear layers - dropping rows from layer i and the matching columns from layer
i+1, which is where a transposition bug would hide.
"""

import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..data.dataset import class_weights, make_loaders
from ..models.student import StudentMLP
from ..training.distill import distillation_loss
from ..training.loops import evaluate, train_model


def neuron_scores(layer) -> torch.Tensor:
    """L2 norm of each output neuron's incoming weight vector."""
    return layer.weight.data.norm(p=2, dim=1)


def structured_prune(model: StudentMLP, amount: float, min_neurons: int = 4) -> StudentMLP:
    """Return a new, smaller model keeping the highest-norm neurons."""
    layers = model.linear_layers()
    hidden_layers = layers[:-1]  # the classifier is never pruned

    keep = []
    for layer in hidden_layers:
        n_keep = max(min_neurons, int(round(layer.out_features * (1.0 - amount))))
        n_keep = min(n_keep, layer.out_features)
        indices = torch.topk(neuron_scores(layer), n_keep).indices
        keep.append(indices.sort().values)  # sorted, so surviving order is preserved

    pruned = StudentMLP(
        model.in_features,
        hidden=tuple(len(idx) for idx in keep),
        num_classes=model.num_classes,
        quantizable=model.quantizable,
    )

    previous = None
    with torch.no_grad():
        for i, (old, new) in enumerate(zip(layers, pruned.linear_layers())):
            weight = old.weight.data.clone()
            bias = old.bias.data.clone()
            if previous is not None:
                # Drop the input columns whose source neurons were removed upstream.
                weight = weight[:, previous]
            if i < len(keep):
                # Drop this layer's own weak neurons - rows of the weight matrix.
                weight = weight[keep[i], :]
                bias = bias[keep[i]]
                previous = keep[i]
            new.weight.data.copy_(weight)
            new.bias.data.copy_(bias)
    return pruned


def iterative_prune(model: StudentMLP, splits, cfg, teacher=None):
    """Prune a fraction, fine-tune, repeat.

    Gentler than one-shot pruning at the same final sparsity: each round removes
    less, and fine-tuning lets the survivors absorb the removed neurons' role
    before the next cut.
    """
    pc = cfg.prune
    train_loader, val_loader, _ = make_loaders(
        splits, cfg.student.batch_size, cfg.data.seed
    )
    weights = class_weights(splits.y_train)

    if teacher is not None:
        teacher.eval()

        def loss_fn(logits, targets, inputs):
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            return distillation_loss(
                logits,
                teacher_logits,
                targets,
                cfg.distill.temperature,
                cfg.distill.alpha,
                weights,
            )
    else:
        def loss_fn(logits, targets, _inputs):
            return F.cross_entropy(logits, targets, weight=weights)

    # Per-step rate that compounds to the requested total across all iterations.
    iterations = max(pc.iterations, 1)
    per_step = 1.0 - (1.0 - pc.total_amount) ** (1.0 / iterations)

    current = copy.deepcopy(model)  # never mutate the caller's model
    history = {"hidden_sizes": [current.hidden_sizes()], "val_f1": []}

    for step in range(iterations):
        current = structured_prune(current, per_step, pc.min_neurons)
        step_history = train_model(
            current,
            train_loader,
            val_loader,
            loss_fn=loss_fn,
            max_epochs=pc.finetune_epochs,
            patience=pc.finetune_epochs,
            lr=cfg.student.lr,
            log_prefix=f"[prune {step + 1}/{iterations}]",
        )
        history["hidden_sizes"].append(current.hidden_sizes())
        history["val_f1"].append(step_history["best_val_f1"])

    y_true, y_pred, _ = evaluate(current, val_loader)
    history["final_val_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return current, history
