"""Configuration for the edge-IDS pipeline.

One dataclass per stage, nested under `Config`. Every value has a default that
matches the design spec, so `Config.default()` alone is a valid, reproducible run.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

# Columns that identify *which capture a row came from* rather than what the
# traffic looked like. Left in, a model memorises attacker addresses and reports
# near-perfect accuracy while having learned nothing that generalises.
LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Dst Port",
    "Timestamp",
]


@dataclass
class DataConfig:
    # CSE-CIC-IDS2018 by default: its processed CSVs sit in a public S3 bucket and
    # download directly, whereas the CIC-IDS2017 host now gates every path behind
    # a form. Friday-02-03-2018 is the botnet capture - the one this project needs.
    source: str = "ids2018"
    files: list = field(
        default_factory=lambda: ["Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"]
    )
    url: str = (
        "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/"
        "CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
    )
    # One directory per experiment: find_data_files recurses, so two datasets
    # sharing a raw_dir would be silently concatenated into a single run.
    raw_dir: str = "data/botnet"
    processed_dir: str = "data/processed"
    drop_columns: list = field(default_factory=lambda: list(LEAKAGE_COLUMNS))
    val_size: float = 0.2
    test_size: float = 0.2
    clip: float = 5.0
    sample_frac: float = 1.0
    seed: int = 42


@dataclass
class TeacherConfig:
    hidden: list = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 40
    patience: int = 5


@dataclass
class StudentConfig:
    hidden: list = field(default_factory=lambda: [32, 16])
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 8


@dataclass
class DistillConfig:
    temperature: float = 4.0
    alpha: float = 0.7


@dataclass
class PruneConfig:
    total_amount: float = 0.5
    iterations: int = 3
    finetune_epochs: int = 5
    min_neurons: int = 4


@dataclass
class QuantizeConfig:
    backend: str = "auto"  # resolved at runtime; this build exposes only onednn
    calibration_batches: int = 200


@dataclass
class BenchmarkConfig:
    warmup: int = 100
    iterations: int = 1000
    threads: int = 1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    artifacts_dir: str = "artifacts"

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def load(cls, path) -> "Config":
        """Load a YAML config, falling back to defaults for anything absent."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        for f in fields(cls):
            if f.name not in raw:
                continue
            section = getattr(cfg, f.name)
            if is_dataclass(section):
                for key, value in (raw[f.name] or {}).items():
                    if hasattr(section, key):
                        setattr(section, key, value)
            else:
                setattr(cfg, f.name, raw[f.name])
        return cfg
