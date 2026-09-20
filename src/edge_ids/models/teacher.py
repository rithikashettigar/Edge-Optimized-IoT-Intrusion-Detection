"""The teacher: an over-parameterized MLP that is not meant to be deployable.

Its job is to be as accurate as this data permits, establishing the ceiling that
every compressed model is measured against, and to supply the soft probability
distributions the student learns from. At roughly 211K parameters for ~70 input
features it is deliberately far larger than the problem requires.
"""

import torch.nn as nn


class TeacherMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden=(512, 256, 128, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden = tuple(hidden)
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        previous = in_features
        for width in self.hidden:
            layers += [
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
