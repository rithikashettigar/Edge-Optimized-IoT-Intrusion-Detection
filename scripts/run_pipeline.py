"""Run the whole pipeline: data -> teacher -> students -> prune -> quantize -> benchmark.

Every stage is also importable on its own, so a failure late in the run does not
mean retraining the teacher.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_ids.compression.prune import iterative_prune  # noqa: E402
from edge_ids.compression.quantize import (  # noqa: E402
    quantize_dynamic_model,
    quantize_static,
    resolve_backend,
)
from edge_ids.config import Config  # noqa: E402
from edge_ids.data.dataset import make_loaders  # noqa: E402
from edge_ids.data.download import ensure_dataset, find_data_files  # noqa: E402
from edge_ids.data.preprocess import build_splits, load_raw, save_splits  # noqa: E402
from edge_ids.evaluation.benchmark import benchmark_model  # noqa: E402
from edge_ids.evaluation.report import (  # noqa: E402
    plot_confusion_matrices,
    plot_latency,
    plot_roc_curves,
    plot_size_vs_f1,
    write_markdown_table,
    write_metrics_csv,
)
from edge_ids.training.distill import train_student  # noqa: E402
from edge_ids.training.loops import set_seed  # noqa: E402
from edge_ids.training.train_teacher import train_teacher  # noqa: E402


def synthetic_frame(n: int = 8000, d: int = 20, seed: int = 42) -> pd.DataFrame:
    """Stand-in traffic for smoke tests: two overlapping populations, imbalanced
    like real captures. Not a substitute for CIC-IDS2017 results."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    shift = rng.normal(1.2, 0.4, d)
    X = rng.normal(size=(n, d)) + y[:, None] * shift
    frame = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(d)])
    frame["Label"] = np.where(y == 1, "Bot", "BENIGN")
    return frame


def run(cfg: Config, sample_frac=None, use_synthetic: bool = False) -> list:
    set_seed(cfg.data.seed)
    if sample_frac is not None:
        cfg.data.sample_frac = sample_frac

    if use_synthetic:
        print("using synthetic data (smoke test - not a publishable result)")
        frame = synthetic_frame()
    else:
        raw_dir = ensure_dataset(cfg.data)
        files = find_data_files(raw_dir)
        print(f"reading {len(files)} flow file(s) from {raw_dir}")
        for path in files:
            print(f"  {path.name}  ({path.stat().st_size / 1024**2:.0f} MB)")
        frame = load_raw(files, cfg.data.sample_frac, cfg.data.seed)

    splits = build_splits(frame, cfg.data)
    print(
        f"features: {splits.n_features}  "
        f"train/val/test: {len(splits.y_train):,}/{len(splits.y_val):,}/"
        f"{len(splits.y_test):,}  attack rate: {splits.y_train.mean():.3f}"
    )

    artifacts = Path(cfg.artifacts_dir)
    if not use_synthetic:
        save_splits(splits, artifacts / "processed")

    teacher, _ = train_teacher(splits, cfg)
    student_plain, _ = train_student(splits, cfg, teacher=None)
    student_kd, _ = train_student(splits, cfg, teacher=teacher)
    pruned, prune_history = iterative_prune(student_kd, splits, cfg, teacher=teacher)
    print(f"pruning path: {' -> '.join(str(h) for h in prune_history['hidden_sizes'])}")

    train_loader, _, test_loader = make_loaders(
        splits, cfg.student.batch_size, cfg.data.seed
    )
    sample = torch.from_numpy(splits.X_test[:1]).float()

    candidates = [
        ("teacher", teacher),
        ("student-plain", student_plain),
        ("student-kd", student_kd),
        ("student-pruned", pruned),
    ]

    backend = resolve_backend(cfg.quantize.backend)
    try:
        candidates.append(
            ("student-pruned-int8-static", quantize_static(pruned, train_loader, cfg))
        )
        print(f"static quantization backend: {backend}")
    except Exception as exc:
        print(f"static quantization unavailable on '{backend}' ({exc}); dynamic only")
    candidates.append(("student-pruned-int8-dynamic", quantize_dynamic_model(pruned)))

    print(
        f"\n{'model':<28}{'F1':>8}{'FPR':>9}{'params':>9}"
        f"{'size KB':>10}{'weights KB':>12}{'median us':>11}"
    )
    print("-" * 87)
    rows = []
    for name, model in candidates:
        row = benchmark_model(name, model, test_loader, sample, cfg, artifacts)
        rows.append(row)
        print(
            f"{name:<28}{row['f1']:>8.4f}{row['fpr']:>9.4f}{row['params']:>9,}"
            f"{row['size_kb']:>10.2f}{row['weight_memory_kb']:>12.2f}"
            f"{row['median_us']:>11.1f}"
        )

    results = artifacts / "results"
    write_metrics_csv(rows, results / "metrics.csv")
    write_markdown_table(rows, results / "comparison.md")
    plot_size_vs_f1(rows, results / "size_vs_f1.png")
    plot_latency(rows, results / "latency.png")
    plot_roc_curves(rows, results / "roc_curves.png")
    plot_confusion_matrices(rows, results / "confusion_matrices.png")
    print(f"\nresults written to {results.resolve()}")

    kd = next((r for r in rows if r["model"] == "student-kd"), None)
    plain = next((r for r in rows if r["model"] == "student-plain"), None)
    if kd and plain:
        delta = (kd["f1"] - plain["f1"]) * 100
        print(f"distillation effect: {delta:+.2f} F1 points over the plain student")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-optimized IDS pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="fraction of flows to use, e.g. 0.05 for a quick run",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run on generated data instead of CIC-IDS2017",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = Config.load(config_path) if config_path.exists() else Config.default()
    run(cfg, sample_frac=args.sample_frac, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
