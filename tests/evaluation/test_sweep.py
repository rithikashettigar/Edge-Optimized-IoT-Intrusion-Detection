import torch

from conftest import make_separable_splits as separable_splits
from edge_ids.config import Config
from edge_ids.evaluation.sweep import sweep_student_sizes
from edge_ids.training.train_teacher import train_teacher


def _cfg():
    cfg = Config.default()
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [16, 8], 15, 32
    cfg.student.max_epochs, cfg.student.batch_size = 15, 32
    cfg.teacher.lr = cfg.student.lr = 1e-2
    return cfg


def test_sweep_returns_one_row_per_width():
    cfg = _cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    rows = sweep_student_sizes(splits, cfg, teacher, widths=[(8, 4), (4, 2)])
    assert [r["hidden"] for r in rows] == [(8, 4), (4, 2)]


def test_sweep_row_carries_both_variants_and_their_gap():
    cfg = _cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    rows = sweep_student_sizes(splits, cfg, teacher, widths=[(8, 4)])
    row = rows[0]
    for key in ("hidden", "params", "size_kb", "f1_plain", "f1_kd", "delta_f1"):
        assert key in row, f"missing {key}"
    assert row["delta_f1"] == row["f1_kd"] - row["f1_plain"]


def test_narrower_students_have_fewer_parameters():
    cfg = _cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    rows = sweep_student_sizes(splits, cfg, teacher, widths=[(16, 8), (4, 2)])
    assert rows[0]["params"] > rows[1]["params"]


def test_sweep_does_not_mutate_the_teacher():
    cfg = _cfg()
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    before = teacher.net[0].weight.detach().clone()
    sweep_student_sizes(splits, cfg, teacher, widths=[(4, 2)])
    assert torch.allclose(before, teacher.net[0].weight)


def test_sweep_leaves_the_configured_student_width_unchanged():
    # The sweep overrides width per iteration; it must not leak that back into
    # the caller's config and silently change later runs.
    cfg = _cfg()
    original = list(cfg.student.hidden)
    splits = separable_splits()
    teacher, _ = train_teacher(splits, cfg)
    sweep_student_sizes(splits, cfg, teacher, widths=[(4, 2)])
    assert list(cfg.student.hidden) == original
