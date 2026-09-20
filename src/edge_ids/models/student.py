"""The student: a compact MLP small enough for router-class hardware.

Plain Linear+ReLU only, with no BatchNorm. That is a deliberate constraint rather
than an oversight: BatchNorm needs layer fusion before static quantization, and
its running statistics would have to be sliced in step with the surviving neurons
during the pruning rebuild. Keeping the student a bare stack of Linear layers
makes both compression stages auditable, and at this width the accuracy cost is
negligible.
"""

import torch.nn as nn
from torch.ao.quantization import DeQuantStub, QuantStub


class StudentMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden=(32, 16),
        num_classes: int = 2,
        quantizable: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.quantizable = quantizable

        # Inert until static quantization turns them on.
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        layers: list[nn.Module] = []
        previous = in_features
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.quantizable:
            x = self.quant(x)
        x = self.net(x)
        if self.quantizable:
            x = self.dequant(x)
        return x

    def linear_layers(self) -> list[nn.Linear]:
        """Every Linear, ordered input to output. The last one is the classifier."""
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def hidden_sizes(self) -> tuple:
        """Widths of the hidden layers only - the output layer is never pruned."""
        return tuple(layer.out_features for layer in self.linear_layers()[:-1])
