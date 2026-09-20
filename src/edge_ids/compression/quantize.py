"""Post-training INT8 quantization.

Static PTQ quantizes weights *and* activations, which is what the design calls
for, and needs a calibration pass to learn activation ranges. Dynamic PTQ
quantizes weights only, needs no calibration, and is reported alongside it as a
robust second data point.
"""

import copy
from pathlib import Path

import torch
import torch.ao.quantization as tq
import torch.nn as nn

# Ordered by preference. Builds vary: fbgemm is common on x86 Linux but absent
# from some Windows wheels, which ship only onednn.
BACKEND_PREFERENCE = ("fbgemm", "onednn", "qnnpack")


def resolve_backend(preferred: str = "auto") -> str:
    """Pick a quantization engine this torch build actually ships.

    Hardcoding one backend fails at convert() time on a build that lacks it,
    after the whole pipeline has already run.
    """
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not supported:
        raise RuntimeError("This torch build supports no quantization engine.")
    if preferred != "auto" and preferred in supported:
        return preferred
    for candidate in BACKEND_PREFERENCE:
        if candidate in supported:
            return candidate
    return supported[0]


def saved_size_kb(model, path) -> float:
    """Measured bytes on disk - never computed from parameter counts, which
    cannot detect a pruning implementation that masked instead of rebuilt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size / 1024.0


def _fuse_linear_relu(model):
    """Fuse each Linear+ReLU pair so the quantized graph has fewer requant steps."""
    modules = list(model.net)
    pairs = [
        [f"net.{i}", f"net.{i + 1}"]
        for i in range(len(modules) - 1)
        if isinstance(modules[i], nn.Linear) and isinstance(modules[i + 1], nn.ReLU)
    ]
    if pairs:
        tq.fuse_modules(model, pairs, inplace=True)
    return model


def quantize_static(model, calib_loader, cfg):
    """Weights and activations to INT8, with activation ranges learned by calibration."""
    backend = resolve_backend(cfg.quantize.backend)
    quantized = copy.deepcopy(model)  # never mutate the caller's model
    quantized.quantizable = True
    quantized.eval()

    torch.backends.quantized.engine = backend
    _fuse_linear_relu(quantized)
    quantized.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(quantized, inplace=True)

    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= cfg.quantize.calibration_batches:
                break
            quantized(x)  # calibration uses TRAINING data only

    tq.convert(quantized, inplace=True)
    return quantized


def quantize_dynamic_model(model):
    """Weights to INT8, activations quantized on the fly. No calibration needed."""
    return tq.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
    )
