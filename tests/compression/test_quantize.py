import pytest
import torch

from conftest import make_separable_splits as separable_splits
from edge_ids.compression.quantize import (
    quantize_dynamic_model,
    quantize_static,
    resolve_backend,
    saved_size_kb,
)
from edge_ids.config import Config
from edge_ids.data.dataset import make_loaders
from edge_ids.models.student import StudentMLP


def _supported():
    return [e for e in torch.backends.quantized.supported_engines if e != "none"]


def test_resolve_backend_returns_a_supported_engine():
    # This build reports only ['onednn']; a hardcoded fbgemm would fail at convert().
    assert resolve_backend("auto") in _supported()


def test_resolve_backend_honours_an_explicit_supported_choice():
    engine = _supported()[0]
    assert resolve_backend(engine) == engine


def test_resolve_backend_falls_back_when_the_preference_is_unavailable():
    assert resolve_backend("definitely-not-a-backend") in _supported()


def test_dynamic_quantization_shrinks_the_saved_model(tmp_path):
    model = StudentMLP(64, hidden=(128, 64)).eval()
    quantized = quantize_dynamic_model(model)
    assert saved_size_kb(quantized, tmp_path / "q.pt") < saved_size_kb(
        model, tmp_path / "f.pt"
    )


def test_dynamic_quantized_model_still_predicts():
    model = StudentMLP(6, hidden=(16, 8)).eval()
    assert quantize_dynamic_model(model)(torch.randn(4, 6)).shape == (4, 2)


def test_dynamic_quantization_does_not_mutate_the_source_model():
    model = StudentMLP(6, hidden=(16, 8)).eval()
    before = model.net[0].weight.detach().clone()
    quantize_dynamic_model(model)
    assert torch.allclose(before, model.net[0].weight)


def _calibration_loader():
    train, _, _ = make_loaders(separable_splits(), batch_size=32, seed=42)
    return train


def _static_cfg():
    cfg = Config.default()
    cfg.quantize.calibration_batches = 2
    return cfg


def test_static_quantization_shrinks_a_model_whose_weights_dominate(tmp_path):
    # Measured at a width where weight bytes exceed the fixed quantization
    # metadata - see test_static_quantization_can_grow_a_tiny_model for the
    # regime where that is not true.
    wide = StudentMLP(70, hidden=(128, 64), quantizable=True).eval()
    loader, _, _ = make_loaders(separable_splits(n=400, d=70), batch_size=32, seed=42)
    quantized = quantize_static(wide, loader, _static_cfg())
    assert quantized(torch.randn(4, 70)).shape == (4, 2)
    assert saved_size_kb(quantized, tmp_path / "q.pt") < 0.5 * saved_size_kb(
        wide, tmp_path / "f.pt"
    )


def test_static_quantization_can_grow_a_tiny_model(tmp_path):
    # Per-channel scales, zero-points and container metadata are fixed overhead.
    # Below a few thousand parameters they outweigh the INT8 weight savings, so
    # the compressed file is larger. Pinned here so the pipeline reports the
    # real number rather than assuming quantization always shrinks things.
    tiny = StudentMLP(6, hidden=(8, 4), quantizable=True).eval()
    quantized = quantize_static(tiny, _calibration_loader(), _static_cfg())
    assert saved_size_kb(quantized, tmp_path / "q.pt") > saved_size_kb(
        tiny, tmp_path / "f.pt"
    )


def test_static_quantization_does_not_mutate_the_source_model():
    model = StudentMLP(6, hidden=(16, 8), quantizable=True).eval()
    before = model.net[0].weight.detach().clone()
    quantize_static(model, _calibration_loader(), _static_cfg())
    assert torch.allclose(before, model.net[0].weight)


def test_static_quantization_preserves_a_trained_decision_boundary():
    # INT8 should cost a little accuracy, not scramble the boundary. Measured on
    # a trained model: an untrained one has near-tied logits everywhere, so
    # rounding flips argmax constantly and the test would measure noise.
    from edge_ids.training.distill import train_student

    cfg = Config.default()
    cfg.student.hidden, cfg.student.batch_size = [16, 8], 32
    cfg.student.lr, cfg.student.max_epochs = 1e-2, 30
    cfg.quantize.calibration_batches = 5

    splits = separable_splits(n=600)
    trained, _ = train_student(splits, cfg)
    trained.quantizable = True
    trained.eval()

    loader, _, _ = make_loaders(splits, batch_size=32, seed=42)
    quantized = quantize_static(trained, loader, cfg)

    x = torch.from_numpy(splits.X_test).float()
    with torch.no_grad():
        agreement = (trained(x).argmax(1) == quantized(x).argmax(1)).float().mean()
    assert agreement > 0.95


def test_saved_size_reports_measured_bytes(tmp_path):
    path = tmp_path / "m.pt"
    size = saved_size_kb(StudentMLP(6, hidden=(16, 8)), path)
    assert path.exists()
    assert size == pytest.approx(path.stat().st_size / 1024.0)
