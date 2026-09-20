from conftest import make_separable_splits as separable_splits
from edge_ids.config import Config
from edge_ids.models import count_parameters
from edge_ids.training.train_teacher import train_teacher


def _small_cfg():
    cfg = Config.default()
    cfg.teacher.hidden = [16, 8]
    cfg.teacher.max_epochs = 12
    cfg.teacher.batch_size = 32
    return cfg


def test_teacher_trains_and_reaches_useful_val_f1():
    model, history = train_teacher(separable_splits(), _small_cfg())
    assert history["best_val_f1"] > 0.8
    assert model.in_features == 6


def test_teacher_honours_the_configured_architecture():
    cfg = _small_cfg()
    model, _ = train_teacher(separable_splits(), cfg)
    assert model.hidden == (16, 8)
    assert count_parameters(model) > 0


def test_teacher_training_is_reproducible():
    first, _ = train_teacher(separable_splits(), _small_cfg())
    second, _ = train_teacher(separable_splits(), _small_cfg())
    import torch

    assert torch.allclose(first.net[0].weight, second.net[0].weight)


def test_teacher_learns_despite_class_imbalance():
    # 10% attacks: an unweighted model would predict benign everywhere and score
    # 0.0 F1 on the attack class. Given a real training budget, class weighting
    # should recover the minority class almost completely.
    cfg = _small_cfg()
    cfg.teacher.max_epochs = 60
    cfg.teacher.patience = 60
    splits = separable_splits(n=600, attack_rate=0.1)
    _, history = train_teacher(splits, cfg)
    assert history["best_val_f1"] > 0.9
