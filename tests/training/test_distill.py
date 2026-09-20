import torch
import torch.nn.functional as F

from conftest import make_separable_splits as separable_splits
from edge_ids.config import Config
from edge_ids.training.distill import distillation_loss, train_student
from edge_ids.training.train_teacher import train_teacher


def test_alpha_zero_reduces_exactly_to_cross_entropy():
    student, teacher = torch.randn(8, 2), torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    assert torch.allclose(
        distillation_loss(student, teacher, y, 4.0, 0.0),
        F.cross_entropy(student, y),
        atol=1e-6,
    )


def test_alpha_one_reduces_exactly_to_the_scaled_kl_term():
    student, teacher = torch.randn(8, 2), torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    T = 3.0
    expected = (
        F.kl_div(
            F.log_softmax(student / T, dim=1),
            F.softmax(teacher / T, dim=1),
            reduction="batchmean",
        )
        * T
        * T
    )
    assert torch.allclose(distillation_loss(student, teacher, y, T, 1.0), expected, atol=1e-6)


def test_temperature_squared_factor_is_present_at_every_temperature():
    # Soft-target gradients scale as 1/T^2. Without the correction the KD term
    # collapses as T rises and distillation silently becomes plain CE (spec 5).
    student, teacher = torch.randn(16, 2) * 2, torch.randn(16, 2) * 2
    y = torch.randint(0, 2, (16,))

    def raw_kl(T):
        return F.kl_div(
            F.log_softmax(student / T, dim=1),
            F.softmax(teacher / T, dim=1),
            reduction="batchmean",
        )

    for T in (2.0, 4.0, 8.0):
        assert torch.allclose(
            distillation_loss(student, teacher, y, T, 1.0), raw_kl(T) * T * T, atol=1e-6
        )


def test_corrected_soft_loss_does_not_vanish_as_temperature_rises():
    # The bug this guards against: uncorrected, the T=8 term would be roughly a
    # sixteenth of the T=2 term and the teacher's signal would fade away.
    student, teacher = torch.randn(16, 2) * 2, torch.randn(16, 2) * 2
    y = torch.randint(0, 2, (16,))
    hot = distillation_loss(student, teacher, y, 8.0, 1.0)
    cool = distillation_loss(student, teacher, y, 2.0, 1.0)
    assert hot > 0.5 * cool


def test_loss_is_near_zero_when_student_matches_teacher_and_labels():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    y = torch.tensor([0, 1])
    assert distillation_loss(logits, logits, y, 4.0, 0.7).item() < 1e-3


def test_disagreeing_with_the_teacher_costs_more_than_agreeing():
    y = torch.tensor([0, 1])
    teacher = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    agree = distillation_loss(teacher.clone(), teacher, y, 4.0, 1.0)
    disagree = distillation_loss(-teacher.clone(), teacher, y, 4.0, 1.0)
    assert disagree > agree


def _quick_cfg():
    """Small architectures, but a learning rate and epoch budget that actually
    converge - at the production lr of 1e-3 a dozen epochs leaves both students
    underfit and the test measures nothing but impatience."""
    cfg = Config.default()
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [16, 8], 30, 32
    cfg.student.hidden, cfg.student.max_epochs, cfg.student.batch_size = [8, 4], 30, 32
    cfg.teacher.lr = cfg.student.lr = 1e-2
    return cfg


def test_distilled_and_plain_students_both_train():
    cfg = _quick_cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)

    kd_model, kd_history = train_student(splits, cfg, teacher=teacher)
    plain_model, plain_history = train_student(splits, cfg, teacher=None)

    assert kd_history["best_val_f1"] > 0.8
    assert plain_history["best_val_f1"] > 0.8
    # The ablation must differ from the distilled run in exactly one respect.
    assert kd_model.hidden_sizes() == plain_model.hidden_sizes() == (8, 4)


def test_the_teacher_is_not_updated_by_distillation():
    cfg = _quick_cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    before = teacher.net[0].weight.detach().clone()
    train_student(splits, cfg, teacher=teacher)
    assert torch.allclose(before, teacher.net[0].weight)
