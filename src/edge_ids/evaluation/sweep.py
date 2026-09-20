"""Sweep student capacity, with and without distillation.

Both completed runs showed no distillation benefit, for opposite reasons: on the
botnet capture the task was saturated so the student needed no help, and on the
Infiltration capture the teacher itself barely beat chance so it had nothing to
teach. Distillation needs both a competent teacher and a student too small to
match it unaided.

This sweep supplies the missing variable. Holding the task and the teacher fixed,
it narrows the student until capacity actually binds, and reports the F1 gap
between the distilled and plain students at every width.
"""

import copy
import tempfile
from pathlib import Path

from ..compression.quantize import saved_size_kb
from ..data.dataset import make_loaders
from ..models import count_parameters
from ..training.distill import train_student
from ..training.loops import evaluate
from .metrics import classification_metrics


def sweep_student_sizes(splits, cfg, teacher, widths) -> list[dict]:
    """Train a plain and a distilled student at each width; return one row each.

    The teacher is shared across every width so the only variable is student
    capacity. The caller's config is never mutated.
    """
    _, _, test_loader = make_loaders(splits, cfg.student.batch_size, cfg.data.seed)
    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        for width in widths:
            local = copy.deepcopy(cfg)  # never leak the width back to the caller
            local.student.hidden = list(width)

            plain, _ = train_student(splits, local, teacher=None)
            distilled, _ = train_student(splits, local, teacher=teacher)

            metrics_plain = classification_metrics(*evaluate(plain, test_loader))
            metrics_kd = classification_metrics(*evaluate(distilled, test_loader))

            rows.append(
                {
                    "hidden": tuple(width),
                    "params": count_parameters(plain),
                    "size_kb": saved_size_kb(plain, Path(tmp) / f"s{width}.pt"),
                    "f1_plain": metrics_plain["f1"],
                    "f1_kd": metrics_kd["f1"],
                    "delta_f1": metrics_kd["f1"] - metrics_plain["f1"],
                    "fpr_plain": metrics_plain["fpr"],
                    "fpr_kd": metrics_kd["fpr"],
                    "auc_plain": metrics_plain["roc_auc"],
                    "auc_kd": metrics_kd["roc_auc"],
                }
            )
            print(
                f"[sweep] {str(width):>10}  {rows[-1]['params']:>6,} params  "
                f"plain {metrics_plain['f1']:.4f}  kd {metrics_kd['f1']:.4f}  "
                f"delta {rows[-1]['delta_f1']:+.4f}"
            )
    return rows


def write_sweep_table(rows, path, teacher_f1=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["student", "params", "size (KB)", "F1 plain", "F1 distilled",
              "delta F1", "FPR plain", "FPR distilled"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["hidden"]),
                    f"{row['params']:,}",
                    f"{row['size_kb']:.2f}",
                    f"{row['f1_plain']:.4f}",
                    f"{row['f1_kd']:.4f}",
                    f"{row['delta_f1']:+.4f}",
                    f"{row['fpr_plain']:.4f}",
                    f"{row['fpr_kd']:.4f}",
                ]
            )
            + " |"
        )
    if teacher_f1 is not None:
        lines += ["", f"Teacher F1 on the same test set: {teacher_f1:.4f}."]
    lines += [
        "",
        "A positive `delta F1` means distillation beat plain training at that",
        "width. Distillation needs both a competent teacher and a student too",
        "small to match it unaided; this sweep varies the second condition.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_sweep(rows, path, teacher_f1=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [r["params"] for r in rows]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True, height_ratios=[2, 1]
    )
    top.plot(params, [r["f1_plain"] for r in rows], "o-", label="plain student")
    top.plot(params, [r["f1_kd"] for r in rows], "s-", label="distilled student")
    if teacher_f1 is not None:
        top.axhline(teacher_f1, linestyle="--", linewidth=1, color="grey",
                    label=f"teacher ({teacher_f1:.4f})")
    top.set_xscale("log")
    top.set_ylabel("F1 on test set")
    top.set_title("Does distillation help as the student shrinks?")
    top.legend(fontsize=8)
    top.grid(alpha=0.3)

    bottom.axhline(0, color="black", linewidth=0.8)
    bottom.plot(params, [r["delta_f1"] for r in rows], "d-", color="tab:purple")
    bottom.set_xscale("log")
    bottom.set_xlabel("student parameters (log scale)")
    bottom.set_ylabel("F1 gain from KD")
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
