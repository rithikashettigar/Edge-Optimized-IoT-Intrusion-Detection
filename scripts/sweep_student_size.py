"""Sweep student capacity on a fixed task, with and without distillation.

Answers the question the two full pipeline runs left open: distillation showed no
benefit on the botnet capture because the task was saturated, and none on the
Infiltration capture because the teacher itself barely beat chance. Neither run
put the student under real capacity pressure.

Holding the task and the teacher fixed, this narrows the student until capacity
binds and reports the F1 gap at every width.

  python scripts/sweep_student_size.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_ids.config import Config  # noqa: E402
from edge_ids.data.dataset import make_loaders  # noqa: E402
from edge_ids.data.download import ensure_dataset, find_data_files  # noqa: E402
from edge_ids.data.preprocess import build_splits, load_raw  # noqa: E402
from edge_ids.evaluation.metrics import classification_metrics  # noqa: E402
from edge_ids.evaluation.sweep import (  # noqa: E402
    plot_sweep,
    sweep_student_sizes,
    write_sweep_table,
)
from edge_ids.models.teacher import TeacherMLP  # noqa: E402
from edge_ids.training.loops import evaluate, set_seed  # noqa: E402
from edge_ids.training.train_teacher import train_teacher  # noqa: E402

DEFAULT_WIDTHS = [(32, 16), (16, 8), (8, 4), (4, 2), (2, 2)]


def load_or_train_teacher(splits, cfg):
    """Reuse the teacher the pipeline already saved, rather than retraining it."""
    saved = Path(cfg.artifacts_dir) / "models" / "teacher.pt"
    model = TeacherMLP(
        splits.n_features, hidden=tuple(cfg.teacher.hidden), dropout=cfg.teacher.dropout
    )
    if saved.exists():
        try:
            # weights_only: this file holds a tensor state dict and nothing else,
            # so there is no reason to let the unpickler build arbitrary objects.
            model.load_state_dict(
                torch.load(saved, map_location="cpu", weights_only=True)
            )
            model.eval()
            print(f"loaded teacher from {saved}")
            return model
        except (RuntimeError, KeyError) as exc:
            print(f"saved teacher unusable ({exc}); retraining")
    print("training teacher")
    model, _ = train_teacher(splits, cfg)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Student capacity sweep")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-epochs", type=int, default=30,
                        help="student epoch cap; the sweep trains 2 per width")
    args = parser.parse_args()

    cfg = Config.load(args.config) if Path(args.config).exists() else Config.default()
    cfg.student.max_epochs = args.max_epochs
    cfg.student.patience = 5
    set_seed(cfg.data.seed)

    raw_dir = ensure_dataset(cfg.data)
    frame = load_raw(find_data_files(raw_dir), cfg.data.sample_frac, cfg.data.seed)
    splits = build_splits(frame, cfg.data)
    print(f"features: {splits.n_features}  train: {len(splits.y_train):,}")

    teacher = load_or_train_teacher(splits, cfg)
    _, _, test_loader = make_loaders(splits, cfg.student.batch_size, cfg.data.seed)
    teacher_f1 = classification_metrics(*evaluate(teacher, test_loader))["f1"]
    print(f"teacher F1 on test: {teacher_f1:.4f}\n")

    rows = sweep_student_sizes(splits, cfg, teacher, DEFAULT_WIDTHS)

    out = Path(cfg.artifacts_dir) / "results"
    write_sweep_table(rows, out / "student_sweep.md", teacher_f1)
    plot_sweep(rows, out / "student_sweep.png", teacher_f1)
    print(f"\nwritten to {out.resolve()}")

    gains = [r for r in rows if r["delta_f1"] > 0]
    if gains:
        best = max(gains, key=lambda r: r["delta_f1"])
        print(
            f"distillation helps at {len(gains)} of {len(rows)} widths; "
            f"largest gain {best['delta_f1']:+.4f} F1 at {best['hidden']} "
            f"({best['params']:,} params)"
        )
    else:
        print("distillation did not help at any width tested")


if __name__ == "__main__":
    main()
