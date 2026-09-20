"""Knowledge distillation: soft-target KL against the teacher, blended with hard CE.

The teacher's softened probability distribution carries more information than the
hard label does - it says which benign flows *almost* looked like attacks, and how
confidently. That is the signal the student cannot recover from labels alone.
"""

import torch
import torch.nn.functional as F

from ..data.dataset import class_weights, make_loaders
from ..models.student import StudentMLP
from .loops import set_seed, train_model


def distillation_loss(
    student_logits,
    teacher_logits,
    targets,
    temperature: float,
    alpha: float,
    weights=None,
):
    """alpha * T^2 * KL(student_T || teacher_T) + (1 - alpha) * CE(student, y).

    The T^2 factor is mandatory, not cosmetic. Soft-target gradients scale as
    1/T^2, so without the correction the KD term shrinks quadratically as the
    temperature rises - raising T would silently turn this back into plain
    cross-entropy while still looking like distillation.
    """
    T = temperature
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits, targets, weight=weights)
    return alpha * soft + (1.0 - alpha) * hard


def train_student(splits, cfg, teacher=None):
    """Train the student. `teacher=None` trains the no-distillation ablation.

    Both paths share every setting - architecture, optimizer, learning rate, epoch
    budget, class weights - so the difference in their results is attributable to
    the KD term and nothing else.
    """
    set_seed(cfg.data.seed)
    sc = cfg.student

    train_loader, val_loader, _ = make_loaders(splits, sc.batch_size, cfg.data.seed)
    weights = class_weights(splits.y_train)
    model = StudentMLP(splits.n_features, hidden=tuple(sc.hidden))

    if teacher is None:
        def loss_fn(logits, targets, _inputs):
            return F.cross_entropy(logits, targets, weight=weights)

        prefix = "[student-plain]"
    else:
        teacher.eval()

        def loss_fn(logits, targets, inputs):
            # no_grad: the teacher's logits are targets, not a path for gradients.
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

        prefix = "[student-kd]"

    history = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn=loss_fn,
        max_epochs=sc.max_epochs,
        patience=sc.patience,
        lr=sc.lr,
        log_prefix=prefix,
    )
    return model, history
