import torch

from conftest import make_separable_splits as separable_splits
from edge_ids.compression.quantize import quantize_dynamic_model
from edge_ids.config import Config
from edge_ids.data.dataset import make_loaders
from edge_ids.evaluation.benchmark import benchmark_model, measure_latency
from edge_ids.models import count_parameters, effective_param_count
from edge_ids.models.student import StudentMLP


def _fast_cfg():
    cfg = Config.default()
    cfg.benchmark.warmup, cfg.benchmark.iterations = 5, 20
    return cfg


def _test_loader(d=6):
    _, _, test = make_loaders(separable_splits(n=120, d=d), batch_size=32, seed=42)
    return test


def test_latency_returns_positive_median_and_p95():
    result = measure_latency(StudentMLP(6, hidden=(8, 4)).eval(), torch.randn(1, 6), _fast_cfg())
    assert result["median_us"] > 0
    assert result["p95_us"] >= result["median_us"]


def test_latency_measurement_restores_the_thread_count():
    before = torch.get_num_threads()
    measure_latency(StudentMLP(6, hidden=(8, 4)).eval(), torch.randn(1, 6), _fast_cfg())
    assert torch.get_num_threads() == before


def test_benchmark_row_has_every_reported_field(tmp_path):
    row = benchmark_model(
        "student", StudentMLP(6, hidden=(8, 4)).eval(),
        _test_loader(), torch.randn(1, 6), _fast_cfg(), tmp_path,
    )
    for key in (
        "model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr",
        "params", "size_kb", "weight_memory_kb", "bytes_per_param",
        "median_us", "p95_us", "tn", "fp", "fn", "tp",
    ):
        assert key in row, f"missing {key}"
    assert row["size_kb"] > 0
    assert row["params"] > 0


def test_bigger_model_reports_bigger_size(tmp_path):
    cfg, x = _fast_cfg(), torch.randn(1, 6)
    big = benchmark_model("big", StudentMLP(6, hidden=(64, 32)).eval(), _test_loader(), x, cfg, tmp_path)
    small = benchmark_model("small", StudentMLP(6, hidden=(4, 2)).eval(), _test_loader(), x, cfg, tmp_path)
    assert big["size_kb"] > small["size_kb"]
    assert big["weight_memory_kb"] > small["weight_memory_kb"]


def test_float_model_reports_four_bytes_per_parameter(tmp_path):
    row = benchmark_model(
        "float", StudentMLP(6, hidden=(8, 4)).eval(),
        _test_loader(), torch.randn(1, 6), _fast_cfg(), tmp_path,
    )
    assert row["bytes_per_param"] == 4
    assert row["weight_memory_kb"] == row["params"] * 4 / 1024


def test_quantized_model_reports_one_byte_per_parameter(tmp_path):
    quantized = quantize_dynamic_model(StudentMLP(6, hidden=(16, 8)).eval())
    row = benchmark_model(
        "int8", quantized, _test_loader(), torch.randn(1, 6), _fast_cfg(), tmp_path
    )
    assert row["bytes_per_param"] == 1
    # Quantized weights live in packed params, so plain parameter counting sees
    # nothing; the benchmark must still report the real count.
    assert row["params"] > 0


def test_effective_param_count_sees_packed_quantized_weights():
    model = StudentMLP(6, hidden=(16, 8)).eval()
    quantized = quantize_dynamic_model(model)
    assert count_parameters(quantized) < count_parameters(model)
    assert effective_param_count(quantized) == count_parameters(model)
