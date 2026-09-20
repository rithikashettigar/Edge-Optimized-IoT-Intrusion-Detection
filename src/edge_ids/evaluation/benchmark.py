"""Measure what a model actually costs: bytes on disk, weight memory, and latency."""

import time
from pathlib import Path

import numpy as np
import torch

from ..compression.quantize import saved_size_kb
from ..models import effective_param_count, is_quantized
from ..training.loops import evaluate
from .metrics import classification_metrics


def measure_latency(model, sample, cfg) -> dict:
    """Single-sample latency on one thread - a router core, not a laptop's parallelism.

    Median and p95 rather than mean: a single scheduling hiccup skews a mean and
    tells you nothing about the typical cost.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.benchmark.threads)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(cfg.benchmark.warmup):
                model(sample)  # discarded: first calls pay lazy-init costs
            times = []
            for _ in range(cfg.benchmark.iterations):
                start = time.perf_counter()
                model(sample)
                times.append((time.perf_counter() - start) * 1e6)
    finally:
        torch.set_num_threads(previous_threads)

    measurements = np.asarray(times)
    return {
        "median_us": float(np.median(measurements)),
        "p95_us": float(np.percentile(measurements, 95)),
    }


def benchmark_model(name, model, test_loader, sample, cfg, artifacts_dir) -> dict:
    """One complete result row: quality, size, and speed."""
    y_true, y_pred, y_prob = evaluate(model, test_loader)

    row = {"model": name}
    row.update(classification_metrics(y_true, y_pred, y_prob))
    # Underscore-prefixed so the CSV writer's field list skips them; kept on the
    # row so the ROC plot does not need a second pass over the test set.
    row["_y_true"] = y_true
    row["_y_prob"] = y_prob

    params = effective_param_count(model)
    bytes_per_param = 1 if is_quantized(model) else 4

    row["params"] = params
    row["bytes_per_param"] = bytes_per_param
    # Two size figures, because they answer different questions. size_kb is what
    # this pipeline really writes; weight_memory_kb isolates the weight saving
    # from serialization overhead, which dominates at small model sizes.
    row["weight_memory_kb"] = params * bytes_per_param / 1024.0
    row["size_kb"] = saved_size_kb(model, Path(artifacts_dir) / "models" / f"{name}.pt")

    row.update(measure_latency(model, sample, cfg))
    return row
