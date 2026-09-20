"""Turn benchmark rows into the comparison table and plots."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to write PNGs
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = [
    "model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr",
    "params", "bytes_per_param", "weight_memory_kb", "size_kb",
    "median_us", "p95_us", "tn", "fp", "fn", "tp",
]


def _prepare(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics_csv(rows, path) -> None:
    path = _prepare(path)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows, path) -> None:
    """Comparison table, with compression measured against the first row (the teacher)."""
    path = _prepare(path)
    if not rows:
        path.write_text("No results.\n")
        return

    baseline_size = rows[0]["size_kb"]
    baseline_weights = rows[0]["weight_memory_kb"]
    header = [
        "model", "F1", "accuracy", "recall", "FPR", "ROC-AUC",
        "params", "size (KB)", "vs teacher", "weights (KB)", "vs teacher",
        "median (us)", "p95 (us)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for row in rows:
        shrink = (1 - row["size_kb"] / baseline_size) * 100 if baseline_size else 0.0
        weight_shrink = (
            (1 - row["weight_memory_kb"] / baseline_weights) * 100
            if baseline_weights
            else 0.0
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    f"{row['f1']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['fpr']:.4f}",
                    f"{row['roc_auc']:.4f}",
                    f"{row['params']:,}",
                    f"{row['size_kb']:.2f}",
                    f"{shrink:.1f}%",
                    f"{row['weight_memory_kb']:.2f}",
                    f"{weight_shrink:.1f}%",
                    f"{row['median_us']:.1f}",
                    f"{row['p95_us']:.1f}",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "`size (KB)` is the measured file on disk. `weights (KB)` is the parameters",
        "alone at their true precision. They diverge on small models because",
        "quantization metadata is fixed overhead - see spec section 9.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_size_vs_f1(rows, path) -> None:
    path = _prepare(path)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for row in rows:
        ax.scatter(row["size_kb"], row["f1"], s=90)
        ax.annotate(
            row["model"], (row["size_kb"], row["f1"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("model size on disk (KB, log scale)")
    ax.set_ylabel("F1 on test set")
    ax.set_title("Compression vs. detection quality")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latency(rows, path) -> None:
    path = _prepare(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([r["model"] for r in rows], [r["median_us"] for r in rows])
    ax.set_ylabel("median single-sample latency (us, 1 thread)")
    ax.set_title("Inference latency")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(rows, path) -> None:
    """One ROC curve per model. Rows without stored probabilities are skipped."""
    from sklearn.metrics import roc_curve

    path = _prepare(path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = 0
    for row in rows:
        y_true, y_prob = row.get("_y_true"), row.get("_y_prob")
        if y_true is None or y_prob is None or len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{row['model']} ({row['roc_auc']:.4f})")
        plotted += 1

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves" if plotted else "ROC curves (no probability data)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(rows, path) -> None:
    path = _prepare(path)
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.6))
    axes = [axes] if len(rows) == 1 else list(axes)
    for ax, row in zip(axes, rows):
        matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", fontsize=8)
        ax.set_title(row["model"], fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
