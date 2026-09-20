import torch.nn as nn


def count_parameters(model) -> int:
    """Plain nn.Parameter count.

    Returns almost nothing for converted quantized models, whose weights live in
    packed parameters rather than nn.Parameter - use effective_param_count for
    a figure that is comparable across the float and INT8 versions.
    """
    return sum(p.numel() for p in model.parameters())


def is_quantized(model) -> bool:
    """True if any submodule came from torch.ao.nn.quantized."""
    return any("quantized" in type(m).__module__ for m in model.modules())


def effective_param_count(model) -> int:
    """Weight and bias element count, whether stored as Parameters or packed.

    Quantized Linear layers expose weight() and bias() as methods rather than
    attributes, so plain parameter counting misses them entirely and would
    report a compressed model as having no weights at all.
    """
    total = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if not callable(weight) or isinstance(module, nn.Module) and weight is None:
            continue
        try:
            total += weight().numel()
            bias = module.bias() if callable(getattr(module, "bias", None)) else None
            if bias is not None:
                total += bias.numel()
        except (TypeError, RuntimeError, AttributeError):
            continue  # not a quantized layer after all
    return total
