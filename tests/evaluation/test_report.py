import csv

from edge_ids.evaluation.report import (
    plot_confusion_matrices,
    plot_latency,
    plot_roc_curves,
    plot_size_vs_f1,
    write_markdown_table,
    write_metrics_csv,
)

ROWS = [
    {
        "model": "teacher", "accuracy": 0.99, "precision": 0.98, "recall": 0.97,
        "f1": 0.975, "roc_auc": 0.99, "fpr": 0.01, "tn": 90, "fp": 1, "fn": 2, "tp": 7,
        "params": 210882, "bytes_per_param": 4, "weight_memory_kb": 823.8,
        "size_kb": 824.0, "median_us": 210.0, "p95_us": 260.0,
    },
    {
        "model": "student-pruned-int8", "accuracy": 0.97, "precision": 0.95,
        "recall": 0.95, "f1": 0.95, "roc_auc": 0.97, "fpr": 0.03,
        "tn": 88, "fp": 3, "fn": 3, "tp": 6,
        "params": 1290, "bytes_per_param": 1, "weight_memory_kb": 1.26,
        "size_kb": 8.29, "median_us": 40.0, "p95_us": 55.0,
    },
]


def test_csv_has_one_row_per_model(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics_csv(ROWS, path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["model"] == "teacher"


def test_csv_includes_both_size_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    write_metrics_csv(ROWS, path)
    header = next(csv.reader(path.open()))
    assert "size_kb" in header
    assert "weight_memory_kb" in header


def test_markdown_table_includes_headers_and_models(tmp_path):
    path = tmp_path / "comparison.md"
    write_markdown_table(ROWS, path)
    text = path.read_text()
    assert "| model" in text
    assert "teacher" in text
    assert "student-pruned-int8" in text


def test_markdown_table_reports_compression_versus_the_first_row(tmp_path):
    path = tmp_path / "comparison.md"
    write_markdown_table(ROWS, path)
    # 824.0 KB -> 8.29 KB is a 99.0% reduction.
    assert "99.0" in path.read_text()


def test_markdown_table_survives_a_single_row(tmp_path):
    path = tmp_path / "one.md"
    write_markdown_table(ROWS[:1], path)
    assert "teacher" in path.read_text()


def test_markdown_table_survives_no_rows(tmp_path):
    path = tmp_path / "none.md"
    write_markdown_table([], path)
    assert path.exists()


def test_plots_are_written(tmp_path):
    plot_size_vs_f1(ROWS, tmp_path / "size_vs_f1.png")
    plot_latency(ROWS, tmp_path / "latency.png")
    plot_confusion_matrices(ROWS, tmp_path / "cm.png")
    for name in ("size_vs_f1.png", "latency.png", "cm.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_roc_curves_are_written_when_probabilities_are_present(tmp_path):
    import numpy as np

    rows = [dict(r) for r in ROWS]
    rng = np.random.default_rng(0)
    for row in rows:
        row["_y_true"] = rng.integers(0, 2, 50)
        row["_y_prob"] = rng.random(50)
    plot_roc_curves(rows, tmp_path / "roc.png")
    assert (tmp_path / "roc.png").stat().st_size > 0


def test_roc_curves_survive_rows_without_probabilities(tmp_path):
    # Metrics-only rows (e.g. reloaded from CSV) must not crash the plot.
    plot_roc_curves(ROWS, tmp_path / "roc_empty.png")
    assert (tmp_path / "roc_empty.png").stat().st_size > 0


def test_roc_curves_skip_a_single_class_row(tmp_path):
    import numpy as np

    rows = [dict(ROWS[0])]
    rows[0]["_y_true"] = np.zeros(20, dtype=int)
    rows[0]["_y_prob"] = np.linspace(0, 1, 20)
    plot_roc_curves(rows, tmp_path / "roc_one.png")
    assert (tmp_path / "roc_one.png").stat().st_size > 0


def test_plot_confusion_matrices_handles_a_single_model(tmp_path):
    plot_confusion_matrices(ROWS[:1], tmp_path / "one.png")
    assert (tmp_path / "one.png").stat().st_size > 0


def test_report_creates_missing_directories(tmp_path):
    nested = tmp_path / "deep" / "deeper" / "metrics.csv"
    write_metrics_csv(ROWS, nested)
    assert nested.exists()
