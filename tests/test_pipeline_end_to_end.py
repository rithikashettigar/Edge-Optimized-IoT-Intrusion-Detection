import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from edge_ids.config import Config
from run_pipeline import run  # noqa: E402


def _fast_cfg(tmp_path):
    cfg = Config.default()
    cfg.artifacts_dir = str(tmp_path)
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [32, 16], 8, 64
    cfg.student.hidden, cfg.student.max_epochs, cfg.student.batch_size = [16, 8], 8, 64
    cfg.teacher.lr = cfg.student.lr = 1e-2
    cfg.prune.iterations, cfg.prune.finetune_epochs = 2, 3
    cfg.quantize.calibration_batches = 2
    cfg.benchmark.warmup, cfg.benchmark.iterations = 5, 20
    return cfg


def test_pipeline_produces_every_model_and_writes_results(tmp_path):
    rows = run(_fast_cfg(tmp_path), use_synthetic=True)

    names = [r["model"] for r in rows]
    assert names[:4] == ["teacher", "student-plain", "student-kd", "student-pruned"]
    assert any("int8" in name for name in names)

    results = tmp_path / "results"
    for artifact in (
        "metrics.csv", "comparison.md", "size_vs_f1.png",
        "latency.png", "confusion_matrices.png",
    ):
        assert (results / artifact).exists(), f"missing {artifact}"


def test_compression_actually_shrinks_the_model(tmp_path):
    rows = {r["model"]: r for r in run(_fast_cfg(tmp_path), use_synthetic=True)}
    assert rows["student-kd"]["size_kb"] < rows["teacher"]["size_kb"]
    # Physical rebuild, not masking: the pruned file is genuinely smaller.
    assert rows["student-pruned"]["size_kb"] < rows["student-kd"]["size_kb"]
    assert rows["student-pruned"]["params"] < rows["student-kd"]["params"]


def test_quantization_reduces_weight_memory_even_where_the_file_grows(tmp_path):
    rows = {r["model"]: r for r in run(_fast_cfg(tmp_path), use_synthetic=True)}
    int8 = next(r for name, r in rows.items() if "int8" in name)
    pruned = rows["student-pruned"]
    assert int8["bytes_per_param"] == 1
    assert int8["weight_memory_kb"] < pruned["weight_memory_kb"]


def test_all_models_reach_useful_detection_quality(tmp_path):
    rows = run(_fast_cfg(tmp_path), use_synthetic=True)
    for row in rows:
        assert row["f1"] > 0.7, f"{row['model']} scored F1 {row['f1']:.3f}"
