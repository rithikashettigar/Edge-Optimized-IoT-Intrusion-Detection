# Edge-Optimized IDS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that trains a large intrusion-detection MLP on CIC-IDS2017, distills it into a tiny student, structurally prunes and INT8-quantizes that student, and benchmarks all five resulting models on accuracy, size, and latency.

**Architecture:** A config-driven Python package (`src/edge_ids/`) with one responsibility per module — data, models, training, compression, evaluation — orchestrated by a single `scripts/run_pipeline.py`. Every stage is independently runnable so a failure late in the pipeline does not force retraining from scratch. All reported sizes are measured bytes on disk; all reported latencies are single-threaded and warmed up.

**Tech Stack:** Python 3.12, PyTorch (CPU), scikit-learn, pandas, NumPy, matplotlib, PyYAML, pytest. Environment managed by `uv` at `D:\rithika\.venv`.

**Spec:** `docs/superpowers/specs/2026-09-10-edge-optimized-ids-design.md`

## Global Constraints

- Project root is `D:\rithika`. All paths below are relative to it.
- Python 3.12; PyTorch CPU-only wheels (`--index-url https://download.pytorch.org/whl/cpu`).
- Run tests with `.venv\Scripts\python.exe -m pytest`.
- Random seed 42 everywhere a split, shuffle, or init is involved.
- The `StandardScaler` is fitted on the **training split only**, never on full data.
- Leakage columns (`Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, `Timestamp`) are dropped before training.
- Pruning must **physically rebuild** layers. Masking is not acceptable — see spec §6.
- Model sizes are measured with `torch.save` + `Path.stat().st_size`, never computed from parameter counts.
- Latency is measured with `torch.set_num_threads(1)`, 100 warmup iterations discarded, 1000 measured, reporting median and p95.
- The quantization backend is **auto-detected**, never hardcoded. Verified: this build (torch 2.14.0+cpu) reports `supported_engines == ['onednn']`; `fbgemm` is absent and would fail at conversion.
- Persisted artifacts are written and read by this pipeline only. `joblib` is used for the `StandardScaler` (the sklearn convention); `feature_names` is stored as JSON rather than a pickled object array, so `np.load` never needs `allow_pickle=True`.
- The plain (non-distilled) student must be trained under settings identical to the distilled student except for the KD term.

**Git note:** `git init` happens in Task 1. Commit steps are written into each task, but confirm with the user before the first commit — this project has no repository yet.

---

### Task 1: Scaffolding, dependencies, and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `configs/default.yaml`
- Create: `src/edge_ids/__init__.py`, `src/edge_ids/config.py`
- Create: `src/edge_ids/{data,models,training,compression,evaluation}/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Config.load(path: str | Path) -> Config` with nested attribute access (`cfg.teacher.hidden`, `cfg.distill.temperature`, …) and `Config.default() -> Config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from edge_ids.config import Config

def test_loads_nested_values_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  seed: 7\nteacher:\n  hidden: [8, 4]\n")
    cfg = Config.load(p)
    assert cfg.data.seed == 7
    assert cfg.teacher.hidden == [8, 4]

def test_missing_keys_fall_back_to_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  seed: 7\n")
    cfg = Config.load(p)
    assert cfg.distill.temperature == 4.0
    assert cfg.prune.min_neurons == 4

def test_default_config_matches_spec_values():
    cfg = Config.default()
    assert cfg.teacher.hidden == [512, 256, 128, 64]
    assert cfg.student.hidden == [32, 16]
    assert cfg.distill.alpha == 0.7
    assert cfg.prune.total_amount == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edge_ids'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "edge-ids"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch", "numpy", "pandas", "scikit-learn",
    "matplotlib", "pyyaml", "joblib", "tqdm", "requests",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/edge_ids"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.py[cod]
data/
artifacts/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 5: Write `src/edge_ids/config.py`**

```python
"""Configuration loading. One dataclass per pipeline stage, nested under Config."""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
import yaml

LEAKAGE_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Source Port", "Src Port",
    "Destination IP", "Dst IP", "Destination Port", "Dst Port", "Timestamp",
]

@dataclass
class DataConfig:
    url: str = ("http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/"
                "CIC-IDS-2017/CSVs/MachineLearningCSV.zip")
    raw_dir: str = "data/raw"
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
    backend: str = "auto"       # resolved at runtime; this build exposes only onednn
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
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        for f in fields(cls):
            if f.name in raw:
                section = getattr(cfg, f.name)
                if is_dataclass(section):
                    for key, value in (raw[f.name] or {}).items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                else:
                    setattr(cfg, f.name, raw[f.name])
        return cfg
```

- [ ] **Step 6: Write `configs/default.yaml`**

Mirror every value from `config.py` explicitly so a run is reproducible from the config file alone. Empty `__init__.py` files in each subpackage.

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Init repo and commit**

```bash
git init
git add pyproject.toml .gitignore configs/ src/ tests/ docs/
git commit -m "feat: project scaffolding and configuration"
```

---

### Task 2: Dataset acquisition

**Files:**
- Create: `src/edge_ids/data/download.py`
- Test: `tests/data/test_download.py`

**Interfaces:**
- Consumes: `DataConfig.url`, `DataConfig.raw_dir`
- Produces: `ensure_dataset(cfg: DataConfig) -> Path` returning the directory containing the extracted CSVs; `find_csvs(root: Path) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_download.py
import pytest
from edge_ids.data.download import find_csvs, ensure_dataset
from edge_ids.config import DataConfig

def test_find_csvs_locates_nested_files(tmp_path):
    nested = tmp_path / "MachineLearningCVE"
    nested.mkdir()
    (nested / "Monday.pcap_ISCX.csv").write_text("a,b\n1,2\n")
    (nested / "notes.txt").write_text("ignore me")
    found = find_csvs(tmp_path)
    assert len(found) == 1 and found[0].name == "Monday.pcap_ISCX.csv"

def test_ensure_dataset_skips_download_when_csvs_present(tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n")
    cfg = DataConfig(raw_dir=str(tmp_path))
    assert ensure_dataset(cfg) == tmp_path

def test_ensure_dataset_raises_actionable_error_when_unavailable(tmp_path, monkeypatch):
    cfg = DataConfig(raw_dir=str(tmp_path), url="http://127.0.0.1:1/nope.zip")
    with pytest.raises(RuntimeError, match="manually"):
        ensure_dataset(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'edge_ids.data.download'`

- [ ] **Step 3: Write the implementation**

```python
"""Fetch and extract CIC-IDS2017. Falls back to clear manual instructions."""
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

def find_csvs(root) -> list[Path]:
    return sorted(Path(root).rglob("*.csv"))

def ensure_dataset(cfg) -> Path:
    raw = Path(cfg.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if find_csvs(raw):
        return raw
    archive = raw / "MachineLearningCSV.zip"
    try:
        _download(cfg.url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)
        archive.unlink(missing_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download CIC-IDS2017 automatically ({exc}).\n"
            f"Download MachineLearningCSV.zip manually from\n  {cfg.url}\n"
            f"and extract the CSVs into: {raw.resolve()}"
        ) from exc
    if not find_csvs(raw):
        raise RuntimeError(f"Archive extracted but no CSVs found under {raw.resolve()}")
    return raw

def _download(url: str, dest: Path, timeout: int = 30) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_download.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/data/download.py tests/data/test_download.py
git commit -m "feat: CIC-IDS2017 download with manual fallback"
```

---

### Task 3: Preprocessing

This is the task where the dataset's known defects (spec §3) get handled. Every defect gets its own test.

**Files:**
- Create: `src/edge_ids/data/preprocess.py`
- Test: `tests/data/test_preprocess.py`

**Interfaces:**
- Consumes: `find_csvs` from Task 2, `DataConfig`
- Produces:
  - `normalize_columns(df) -> DataFrame`
  - `drop_leakage_columns(df, cols) -> DataFrame`
  - `clean_numeric(df) -> DataFrame`
  - `binarize_labels(series) -> Series[int]`
  - `drop_constant_columns(df) -> DataFrame`
  - `load_raw(paths, sample_frac, seed) -> DataFrame`
  - `build_splits(df, cfg) -> Splits` where `Splits` is a dataclass with `X_train, y_train, X_val, y_val, X_test, y_test` (all `np.ndarray`, float32/int64), `feature_names: list[str]`, `scaler: StandardScaler`
  - `save_splits(splits, out_dir)` / `load_splits(out_dir) -> Splits`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_preprocess.py
import numpy as np
import pandas as pd
import pytest
from edge_ids.config import DataConfig
from edge_ids.data import preprocess as pp

def test_strips_leading_spaces_from_column_names():
    df = pd.DataFrame({" Flow Duration": [1], "Label ": ["BENIGN"]})
    out = pp.normalize_columns(df)
    assert list(out.columns) == ["Flow Duration", "Label"]

def test_drops_leakage_columns_that_are_present():
    df = pd.DataFrame({"Source IP": ["1.1.1.1"], "Destination Port": [80], "Flow Duration": [5]})
    out = pp.drop_leakage_columns(df, ["Source IP", "Destination Port", "Timestamp"])
    assert list(out.columns) == ["Flow Duration"]

def test_replaces_infinities_and_drops_nan_rows():
    df = pd.DataFrame({"Flow Bytes/s": [1.0, np.inf, -np.inf, np.nan], "x": [1.0, 2.0, 3.0, 4.0]})
    out = pp.clean_numeric(df)
    assert len(out) == 1
    assert np.isfinite(out.to_numpy()).all()

def test_binarizes_labels_benign_to_zero_attacks_to_one():
    s = pd.Series(["BENIGN", " BENIGN ", "Bot", "DDoS", "Web Attack \x96 Brute Force"])
    out = pp.binarize_labels(s)
    assert out.tolist() == [0, 0, 1, 1, 1]

def test_drops_zero_variance_columns():
    df = pd.DataFrame({"constant": [3, 3, 3], "varying": [1, 2, 3]})
    out = pp.drop_constant_columns(df)
    assert list(out.columns) == ["varying"]

def _synthetic_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "Flow Duration": rng.normal(y * 3, 1, n),
        "Total Fwd Packets": rng.normal(y * -2, 1, n),
        "Source IP": ["10.0.0.1"] * n,
        "Label": np.where(y == 1, "Bot", "BENIGN"),
    })

def test_scaler_is_fitted_on_training_split_only():
    splits = pp.build_splits(_synthetic_frame(), DataConfig())
    # A scaler fitted on train alone will not centre the *val* split exactly on zero.
    assert abs(splits.X_train.mean()) < 1e-6
    assert abs(splits.X_val.mean()) > 1e-9

def test_splits_contain_no_leakage_columns_and_are_finite():
    splits = pp.build_splits(_synthetic_frame(), DataConfig())
    assert "Source IP" not in splits.feature_names
    assert "Label" not in splits.feature_names
    for arr in (splits.X_train, splits.X_val, splits.X_test):
        assert np.isfinite(arr).all()

def test_features_are_clipped_to_configured_range():
    df = _synthetic_frame()
    df.loc[0, "Flow Duration"] = 1e9
    splits = pp.build_splits(df, DataConfig(clip=5.0))
    assert splits.X_train.max() <= 5.0 and splits.X_train.min() >= -5.0

def test_split_proportions_and_stratification():
    splits = pp.build_splits(_synthetic_frame(n=1000), DataConfig())
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert abs(len(splits.y_test) / total - 0.2) < 0.02
    rates = [s.mean() for s in (splits.y_train, splits.y_val, splits.y_test)]
    assert max(rates) - min(rates) < 0.05

def test_roundtrip_save_and_load(tmp_path):
    splits = pp.build_splits(_synthetic_frame(), DataConfig())
    pp.save_splits(splits, tmp_path)
    back = pp.load_splits(tmp_path)
    assert np.allclose(back.X_train, splits.X_train)
    assert back.feature_names == splits.feature_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_preprocess.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Turn raw CIC-IDS2017 CSVs into scaled, split, leakage-free arrays."""
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_COLUMN = "Label"
BENIGN = "BENIGN"

@dataclass
class Splits:
    X_train: np.ndarray; y_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def drop_leakage_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def binarize_labels(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() != BENIGN).astype(np.int64)

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.nunique(dropna=False) > 1]

def load_raw(paths, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    frames = []
    for p in paths:
        # latin1: the "Web Attack - Brute Force" label carries a non-UTF8 byte.
        frame = pd.read_csv(p, encoding="latin1", low_memory=False)
        frames.append(normalize_columns(frame))
    df = pd.concat(frames, ignore_index=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df

def build_splits(df: pd.DataFrame, cfg) -> Splits:
    df = normalize_columns(df)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"No '{LABEL_COLUMN}' column; found {list(df.columns)[:10]}")
    y = binarize_labels(df[LABEL_COLUMN])
    X = drop_leakage_columns(df.drop(columns=[LABEL_COLUMN]), cfg.drop_columns)
    X = X.select_dtypes(include=[np.number])

    keep = clean_numeric(X).index
    X, y = X.loc[keep], y.loc[keep]
    dedup = ~X.duplicated()
    X, y = X[dedup], y[dedup]
    X = drop_constant_columns(X)

    feature_names = list(X.columns)
    Xv, yv = X.to_numpy(np.float64), y.to_numpy(np.int64)

    X_tr, X_hold, y_tr, y_hold = train_test_split(
        Xv, yv, test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.seed, stratify=yv)
    rel = cfg.test_size / (cfg.val_size + cfg.test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_hold, y_hold, test_size=rel, random_state=cfg.seed, stratify=y_hold)

    scaler = StandardScaler().fit(X_tr)          # train split only — never full data
    clip = cfg.clip
    tf = lambda a: np.clip(scaler.transform(a), -clip, clip).astype(np.float32)
    return Splits(tf(X_tr), y_tr, tf(X_val), y_val, tf(X_te), y_te, feature_names, scaler)

# Feature names go to JSON rather than a pickled object array, so loading never
# needs allow_pickle=True. joblib is used only for the sklearn scaler, which this
# pipeline both writes and reads - it never loads a scaler from an outside source.
def save_splits(splits: Splits, out_dir) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "splits.npz",
        X_train=splits.X_train, y_train=splits.y_train,
        X_val=splits.X_val, y_val=splits.y_val,
        X_test=splits.X_test, y_test=splits.y_test)
    (out / "feature_names.json").write_text(json.dumps(splits.feature_names))
    joblib.dump(splits.scaler, out / "scaler.joblib")

def load_splits(out_dir) -> Splits:
    out = Path(out_dir)
    d = np.load(out / "splits.npz")
    return Splits(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                  d["X_test"], d["y_test"],
                  json.loads((out / "feature_names.json").read_text()),
                  joblib.load(out / "scaler.joblib"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_preprocess.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/data/preprocess.py tests/data/test_preprocess.py
git commit -m "feat: CIC-IDS2017 preprocessing with leakage and defect handling"
```

---

### Task 4: Torch datasets and loaders

**Files:**
- Create: `src/edge_ids/data/dataset.py`
- Test: `tests/data/test_dataset.py`

**Interfaces:**
- Consumes: `Splits` from Task 3
- Produces: `make_loaders(splits, batch_size, seed) -> tuple[DataLoader, DataLoader, DataLoader]` (train shuffled, val/test not) and `class_weights(y) -> torch.Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py
import numpy as np, torch
from edge_ids.data.dataset import make_loaders, class_weights
from edge_ids.data.preprocess import Splits

def _splits(n=100, d=4):
    rng = np.random.default_rng(0)
    f = lambda k: rng.normal(size=(k, d)).astype(np.float32)
    g = lambda k: rng.integers(0, 2, k).astype(np.int64)
    return Splits(f(n), g(n), f(n // 2), g(n // 2), f(n // 2), g(n // 2),
                  [f"f{i}" for i in range(d)], None)

def test_loaders_yield_correct_dtypes_and_shapes():
    tr, va, te = make_loaders(_splits(), batch_size=16, seed=42)
    x, y = next(iter(tr))
    assert x.dtype == torch.float32 and y.dtype == torch.int64
    assert x.shape[1] == 4

def test_class_weights_upweight_the_minority_class():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y)
    assert w[1] > w[0]

def test_validation_loader_preserves_order():
    s = _splits()
    _, va, _ = make_loaders(s, batch_size=1000, seed=42)
    x, _ = next(iter(va))
    assert np.allclose(x.numpy(), s.X_val)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_dataset.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Wrap numpy splits into torch DataLoaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def _ds(X, y) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)).float(),
                         torch.from_numpy(np.ascontiguousarray(y)).long())

def make_loaders(splits, batch_size: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(_ds(splits.X_train, splits.y_train),
                       batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    val = DataLoader(_ds(splits.X_val, splits.y_val), batch_size=batch_size, shuffle=False)
    test = DataLoader(_ds(splits.X_test, splits.y_test), batch_size=batch_size, shuffle=False)
    return train, val, test

def class_weights(y) -> torch.Tensor:
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_dataset.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/data/dataset.py tests/data/test_dataset.py
git commit -m "feat: torch dataloaders and class weighting"
```

---

### Task 5: Teacher and student models

Both model files land together — they are siblings and the tests compare them.

**Files:**
- Create: `src/edge_ids/models/teacher.py`, `src/edge_ids/models/student.py`
- Test: `tests/models/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `TeacherMLP(in_features, hidden=(512,256,128,64), num_classes=2, dropout=0.3)`
  - `StudentMLP(in_features, hidden=(32,16), num_classes=2, quantizable=False)` with `.hidden_sizes() -> tuple[int, ...]`, `.linear_layers() -> list[nn.Linear]`, and `.quantizable` attribute
  - `count_parameters(model) -> int` in `models/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_models.py
import torch
from edge_ids.models import count_parameters
from edge_ids.models.teacher import TeacherMLP
from edge_ids.models.student import StudentMLP

def test_teacher_forward_shape():
    m = TeacherMLP(10, hidden=(8, 4)).eval()
    assert m(torch.randn(5, 10)).shape == (5, 2)

def test_student_forward_shape():
    assert StudentMLP(10, hidden=(6, 3)).eval()(torch.randn(5, 10)).shape == (5, 2)

def test_teacher_is_larger_than_student():
    assert count_parameters(TeacherMLP(70)) > 20 * count_parameters(StudentMLP(70))

def test_student_has_no_batchnorm():
    # BatchNorm complicates quantization fusion and the pruning rebuild (spec 4.2).
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in StudentMLP(70).modules())

def test_student_reports_hidden_sizes():
    assert StudentMLP(10, hidden=(6, 3)).hidden_sizes() == (6, 3)

def test_student_linear_layers_ordered_input_to_output():
    layers = StudentMLP(10, hidden=(6, 3)).linear_layers()
    assert [l.out_features for l in layers] == [6, 3, 2]

def test_quantizable_student_runs_in_float_before_conversion():
    m = StudentMLP(10, hidden=(6, 3), quantizable=True).eval()
    assert m(torch.randn(4, 10)).shape == (4, 2)

def test_student_accepts_batch_of_one():
    # Single-sample inference is the deployment case; must not need batch statistics.
    assert StudentMLP(10, hidden=(6, 3)).eval()(torch.randn(1, 10)).shape == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/models/test_models.py -v`
Expected: FAIL — modules do not exist

- [ ] **Step 3: Write `src/edge_ids/models/teacher.py`**

```python
"""Over-parameterized teacher. Not deployable by design - it sets the accuracy ceiling."""
import torch.nn as nn

class TeacherMLP(nn.Module):
    def __init__(self, in_features: int, hidden=(512, 256, 128, 64),
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.hidden = tuple(hidden)
        layers, prev = [], in_features
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

- [ ] **Step 4: Write `src/edge_ids/models/student.py`**

```python
"""Compact student. Plain Linear+ReLU only: BatchNorm would need fusion before
static quantization and would have to be sliced in step with the pruning rebuild."""
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

class StudentMLP(nn.Module):
    def __init__(self, in_features: int, hidden=(32, 16),
                 num_classes: int = 2, quantizable: bool = False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.quantizable = quantizable
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.quantizable:
            x = self.quant(x)
        x = self.net(x)
        if self.quantizable:
            x = self.dequant(x)
        return x

    def linear_layers(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def hidden_sizes(self):
        return tuple(l.out_features for l in self.linear_layers()[:-1])
```

- [ ] **Step 5: Write `src/edge_ids/models/__init__.py`**

```python
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/models/test_models.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add src/edge_ids/models/ tests/models/
git commit -m "feat: teacher and student MLP architectures"
```

---

### Task 6: Shared training loops

Written once here and reused by the teacher, both students, and the pruning fine-tune — rather than four near-copies.

**Files:**
- Create: `src/edge_ids/training/loops.py`
- Test: `tests/training/test_loops.py`

**Interfaces:**
- Consumes: models from Task 5, loaders from Task 4
- Produces:
  - `train_model(model, train_loader, val_loader, *, loss_fn, max_epochs, patience, lr, log_prefix="") -> dict` history; restores best-val-F1 weights in place before returning
  - `evaluate(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(y_true, y_pred, y_prob_positive)`
  - `set_seed(seed)`

`loss_fn` has signature `(logits, targets, inputs) -> Tensor`, so plain CE and the distillation loss are interchangeable.

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_loops.py
import numpy as np, torch, torch.nn.functional as F
from edge_ids.data.dataset import make_loaders
from edge_ids.data.preprocess import Splits
from edge_ids.models.student import StudentMLP
from edge_ids.training.loops import train_model, evaluate, set_seed

def _separable_splits(n=400, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def _ce(logits, targets, inputs):
    return F.cross_entropy(logits, targets)

def test_training_reduces_loss_and_learns_separable_data():
    set_seed(42)
    s = _separable_splits()
    tr, va, te = make_loaders(s, batch_size=32, seed=42)
    m = StudentMLP(6, hidden=(8, 4))
    hist = train_model(m, tr, va, loss_fn=_ce, max_epochs=15, patience=5, lr=1e-2)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    y_true, y_pred, _ = evaluate(m, te)
    assert (y_true == y_pred).mean() > 0.85

def test_early_stopping_halts_before_max_epochs():
    set_seed(42)
    tr, va, _ = make_loaders(_separable_splits(), batch_size=32, seed=42)
    m = StudentMLP(6, hidden=(8, 4))
    hist = train_model(m, tr, va, loss_fn=_ce, max_epochs=100, patience=2, lr=1e-2)
    assert len(hist["train_loss"]) < 100

def test_evaluate_returns_probabilities_in_unit_range():
    tr, va, te = make_loaders(_separable_splits(), batch_size=32, seed=42)
    _, _, prob = evaluate(StudentMLP(6, hidden=(8, 4)), te)
    assert prob.min() >= 0.0 and prob.max() <= 1.0

def test_set_seed_makes_initialization_reproducible():
    set_seed(1); a = StudentMLP(6, hidden=(8, 4)).net[0].weight.clone()
    set_seed(1); b = StudentMLP(6, hidden=(8, 4)).net[0].weight.clone()
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_loops.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Training and evaluation loops shared by every model in the pipeline."""
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    trues, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x)
        p = F.softmax(logits, dim=1)[:, 1]
        trues.append(y.numpy())
        preds.append(logits.argmax(1).numpy())
        probs.append(p.numpy())
    return (np.concatenate(trues), np.concatenate(preds), np.concatenate(probs))

def train_model(model, train_loader, val_loader, *, loss_fn,
                max_epochs: int, patience: int, lr: float, log_prefix: str = "") -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_f1": []}
    best_f1, best_state, stale = -1.0, copy.deepcopy(model.state_dict()), 0

    for epoch in range(max_epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y, x)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        train_loss = running / max(seen, 1)

        y_true, y_pred, _ = evaluate(model, val_loader)
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        if log_prefix:
            print(f"{log_prefix} epoch {epoch+1:3d}  loss {train_loss:.4f}  val_f1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_state, stale = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)   # always return the best model, not the last
    history["best_val_f1"] = best_f1
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_loops.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/training/loops.py tests/training/test_loops.py
git commit -m "feat: shared training loop with early stopping on val F1"
```

---

### Task 7: Teacher training

**Files:**
- Create: `src/edge_ids/training/train_teacher.py`
- Test: `tests/training/test_train_teacher.py`

**Interfaces:**
- Consumes: `TeacherMLP`, `train_model`, `class_weights`
- Produces: `train_teacher(splits, cfg) -> tuple[TeacherMLP, dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_train_teacher.py
import numpy as np
from edge_ids.config import Config
from edge_ids.data.preprocess import Splits
from edge_ids.training.train_teacher import train_teacher

def _splits(n=400, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def test_teacher_trains_and_reaches_useful_val_f1():
    cfg = Config.default()
    cfg.teacher.hidden = [16, 8]
    cfg.teacher.max_epochs = 12
    cfg.teacher.batch_size = 32
    model, hist = train_teacher(_splits(), cfg)
    assert hist["best_val_f1"] > 0.8
    assert model.in_features == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_train_teacher.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Train the over-parameterized teacher with class-weighted cross-entropy."""
import torch.nn.functional as F
from ..data.dataset import class_weights, make_loaders
from ..models.teacher import TeacherMLP
from .loops import set_seed, train_model

def train_teacher(splits, cfg):
    set_seed(cfg.data.seed)
    tc = cfg.teacher
    train_loader, val_loader, _ = make_loaders(splits, tc.batch_size, cfg.data.seed)
    weights = class_weights(splits.y_train)
    model = TeacherMLP(splits.n_features, hidden=tuple(tc.hidden), dropout=tc.dropout)

    def loss_fn(logits, targets, _inputs):
        return F.cross_entropy(logits, targets, weight=weights)

    history = train_model(model, train_loader, val_loader, loss_fn=loss_fn,
                          max_epochs=tc.max_epochs, patience=tc.patience,
                          lr=tc.lr, log_prefix="[teacher]")
    return model, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_train_teacher.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/training/train_teacher.py tests/training/test_train_teacher.py
git commit -m "feat: teacher training"
```

---

### Task 8: Knowledge distillation

The `T²` correction is the subtle part and gets a dedicated numeric test.

**Files:**
- Create: `src/edge_ids/training/distill.py`
- Test: `tests/training/test_distill.py`

**Interfaces:**
- Consumes: `TeacherMLP`, `StudentMLP`, `train_model`, `class_weights`
- Produces:
  - `distillation_loss(student_logits, teacher_logits, targets, temperature, alpha, class_weights=None) -> Tensor`
  - `train_student(splits, cfg, teacher=None) -> tuple[StudentMLP, dict]` — `teacher=None` trains the plain ablation

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_distill.py
import numpy as np, torch, torch.nn.functional as F
from edge_ids.config import Config
from edge_ids.data.preprocess import Splits
from edge_ids.training.distill import distillation_loss, train_student
from edge_ids.training.train_teacher import train_teacher

def _splits(n=400, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def test_alpha_zero_reduces_exactly_to_cross_entropy():
    s, t = torch.randn(8, 2), torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    assert torch.allclose(distillation_loss(s, t, y, 4.0, 0.0),
                          F.cross_entropy(s, y), atol=1e-6)

def test_alpha_one_reduces_exactly_to_scaled_kl_term():
    s, t = torch.randn(8, 2), torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    T = 3.0
    expected = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                        reduction="batchmean") * T * T
    assert torch.allclose(distillation_loss(s, t, y, T, 1.0), expected, atol=1e-6)

def test_temperature_squared_factor_is_present():
    # Without the T^2 correction the soft term collapses as T grows and
    # distillation silently degrades into plain cross-entropy (spec 5).
    s, t = torch.randn(16, 2) * 2, torch.randn(16, 2) * 2
    y = torch.randint(0, 2, (16,))
    raw_kl = lambda T: F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                                reduction="batchmean")
    for T in (2.0, 4.0, 8.0):
        assert torch.allclose(distillation_loss(s, t, y, T, 1.0), raw_kl(T) * T * T, atol=1e-6)
    # The corrected loss must not vanish as T rises.
    assert distillation_loss(s, t, y, 8.0, 1.0) > 0.5 * distillation_loss(s, t, y, 2.0, 1.0)

def test_loss_is_zero_when_student_matches_teacher_and_labels():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    y = torch.tensor([0, 1])
    assert distillation_loss(logits, logits, y, 4.0, 0.7).item() < 1e-3

def test_distilled_and_plain_students_both_train():
    cfg = Config.default()
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [16, 8], 10, 32
    cfg.student.hidden, cfg.student.max_epochs, cfg.student.batch_size = [8, 4], 12, 32
    s = _splits()
    teacher, _ = train_teacher(s, cfg)
    kd_model, kd_hist = train_student(s, cfg, teacher=teacher)
    plain_model, plain_hist = train_student(s, cfg, teacher=None)
    assert kd_hist["best_val_f1"] > 0.8 and plain_hist["best_val_f1"] > 0.8
    assert kd_model.hidden_sizes() == plain_model.hidden_sizes() == (8, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_distill.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Knowledge distillation: soft-target KL against the teacher, blended with hard CE."""
import torch
import torch.nn.functional as F
from ..data.dataset import class_weights, make_loaders
from ..models.student import StudentMLP
from .loops import set_seed, train_model

def distillation_loss(student_logits, teacher_logits, targets,
                      temperature: float, alpha: float, class_weights=None):
    """alpha * T^2 * KL(student_T || teacher_T) + (1 - alpha) * CE(student, y).

    The T^2 factor is mandatory: soft-target gradients scale as 1/T^2, so without
    it raising the temperature silently turns this back into plain cross-entropy.
    """
    T = temperature
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, targets, weight=class_weights)
    return alpha * soft + (1.0 - alpha) * hard

def train_student(splits, cfg, teacher=None):
    """teacher=None trains the no-distillation ablation under identical settings."""
    set_seed(cfg.data.seed)
    sc = cfg.student
    train_loader, val_loader, _ = make_loaders(splits, sc.batch_size, cfg.data.seed)
    weights = class_weights(splits.y_train)
    model = StudentMLP(splits.n_features, hidden=tuple(sc.hidden))

    if teacher is None:
        def loss_fn(logits, targets, _inputs):
            return F.cross_entropy(logits, targets, weight=weights)
        prefix = "[student-plain]"
    else:
        teacher.eval()
        def loss_fn(logits, targets, inputs):
            with torch.no_grad():
                t_logits = teacher(inputs)
            return distillation_loss(logits, t_logits, targets,
                                     cfg.distill.temperature, cfg.distill.alpha, weights)
        prefix = "[student-kd]"

    history = train_model(model, train_loader, val_loader, loss_fn=loss_fn,
                          max_epochs=sc.max_epochs, patience=sc.patience,
                          lr=sc.lr, log_prefix=prefix)
    return model, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_distill.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/training/distill.py tests/training/test_distill.py
git commit -m "feat: knowledge distillation with T-squared correction"
```

---

### Task 9: Structured pruning with physical rebuild

**Files:**
- Create: `src/edge_ids/compression/prune.py`
- Test: `tests/compression/test_prune.py`

**Interfaces:**
- Consumes: `StudentMLP`, `train_model`, `distillation_loss`
- Produces:
  - `neuron_scores(layer) -> Tensor` (L2 norm per output neuron)
  - `structured_prune(model, amount, min_neurons=4) -> StudentMLP` — a genuinely smaller model
  - `iterative_prune(model, splits, cfg, teacher=None) -> tuple[StudentMLP, dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/compression/test_prune.py
import numpy as np, torch
from edge_ids.config import Config
from edge_ids.compression.prune import neuron_scores, structured_prune, iterative_prune
from edge_ids.data.preprocess import Splits
from edge_ids.models import count_parameters
from edge_ids.models.student import StudentMLP

def _splits(n=400, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def test_pruning_zero_percent_is_numerically_identical():
    # Guards against row/column transposition in the weight-copy logic.
    torch.manual_seed(0)
    m = StudentMLP(6, hidden=(8, 4)).eval()
    x = torch.randn(5, 6)
    out = structured_prune(m, amount=0.0).eval()
    assert torch.allclose(m(x), out(x), atol=1e-6)

def test_pruning_actually_reduces_parameter_count():
    # A masked implementation would leave this unchanged (spec 6).
    m = StudentMLP(6, hidden=(16, 8))
    pruned = structured_prune(m, amount=0.5)
    assert count_parameters(pruned) < count_parameters(m)
    assert pruned.hidden_sizes() == (8, 4)

def test_pruned_model_still_produces_correct_output_shape():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.5).eval()
    assert pruned(torch.randn(3, 6)).shape == (3, 2)

def test_minimum_neurons_floor_prevents_layer_collapse():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.99, min_neurons=4)
    assert all(h >= 4 for h in pruned.hidden_sizes())

def test_output_layer_is_never_pruned():
    pruned = structured_prune(StudentMLP(6, hidden=(16, 8)), amount=0.5)
    assert pruned.linear_layers()[-1].out_features == 2

def test_highest_norm_neurons_are_the_ones_kept():
    torch.manual_seed(0)
    m = StudentMLP(4, hidden=(4, 4))
    with torch.no_grad():                      # make neuron 2 dominant, neuron 0 dead
        m.net[0].weight.zero_()
        m.net[0].weight[2] = 10.0
        m.net[0].weight[1] = 1.0
        m.net[0].weight[3] = 0.5
    scores = neuron_scores(m.net[0])
    assert int(scores.argmax()) == 2
    pruned = structured_prune(m, amount=0.5, min_neurons=1)
    assert pruned.hidden_sizes()[0] == 2

def test_iterative_pruning_reaches_target_and_keeps_accuracy():
    cfg = Config.default()
    cfg.student.hidden, cfg.student.batch_size = [16, 8], 32
    cfg.prune.total_amount, cfg.prune.iterations, cfg.prune.finetune_epochs = 0.5, 2, 3
    s = _splits()
    m = StudentMLP(6, hidden=(16, 8))
    pruned, hist = iterative_prune(m, s, cfg, teacher=None)
    assert count_parameters(pruned) < count_parameters(m)
    assert hist["final_val_f1"] > 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/compression/test_prune.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Structured pruning that physically rebuilds the network.

torch.nn.utils.prune applies a mask: zeroed weights still occupy full FP32
tensors, so parameter count and file size are unchanged. To report a real
compression number the surviving neurons must be copied into smaller layers.
"""
import copy
import torch
import torch.nn.functional as F
from ..data.dataset import class_weights, make_loaders
from ..models.student import StudentMLP
from ..training.distill import distillation_loss
from ..training.loops import evaluate, train_model
from sklearn.metrics import f1_score

def neuron_scores(layer) -> torch.Tensor:
    """L2 norm of each output neuron's incoming weight vector."""
    return layer.weight.data.norm(p=2, dim=1)

def structured_prune(model: StudentMLP, amount: float, min_neurons: int = 4) -> StudentMLP:
    layers = model.linear_layers()
    hidden_layers = layers[:-1]                  # output layer is never pruned

    keep = []
    for layer in hidden_layers:
        n_keep = max(min_neurons, int(round(layer.out_features * (1.0 - amount))))
        n_keep = min(n_keep, layer.out_features)
        idx = torch.topk(neuron_scores(layer), n_keep).indices.sort().values
        keep.append(idx)

    new_model = StudentMLP(model.in_features,
                           hidden=tuple(len(i) for i in keep),
                           num_classes=model.num_classes,
                           quantizable=model.quantizable)

    prev = None
    for i, (old, new) in enumerate(zip(layers, new_model.linear_layers())):
        w, b = old.weight.data.clone(), old.bias.data.clone()
        if prev is not None:
            w = w[:, prev]                       # drop inputs killed upstream
        if i < len(keep):
            w, b = w[keep[i], :], b[keep[i]]     # drop this layer's weak neurons
            prev = keep[i]
        new.weight.data.copy_(w)
        new.bias.data.copy_(b)
    return new_model

def iterative_prune(model: StudentMLP, splits, cfg, teacher=None):
    """Prune a fraction, fine-tune, repeat - far gentler than one-shot pruning."""
    pc = cfg.prune
    train_loader, val_loader, _ = make_loaders(splits, cfg.student.batch_size, cfg.data.seed)
    weights = class_weights(splits.y_train)
    per_step = 1.0 - (1.0 - pc.total_amount) ** (1.0 / max(pc.iterations, 1))

    if teacher is not None:
        teacher.eval()
        def loss_fn(logits, targets, inputs):
            with torch.no_grad():
                t_logits = teacher(inputs)
            return distillation_loss(logits, t_logits, targets,
                                     cfg.distill.temperature, cfg.distill.alpha, weights)
    else:
        def loss_fn(logits, targets, _inputs):
            return F.cross_entropy(logits, targets, weight=weights)

    current = copy.deepcopy(model)
    history = {"hidden_sizes": [current.hidden_sizes()], "val_f1": []}
    for step in range(pc.iterations):
        current = structured_prune(current, per_step, pc.min_neurons)
        h = train_model(current, train_loader, val_loader, loss_fn=loss_fn,
                        max_epochs=pc.finetune_epochs, patience=pc.finetune_epochs,
                        lr=cfg.student.lr, log_prefix=f"[prune {step+1}/{pc.iterations}]")
        history["hidden_sizes"].append(current.hidden_sizes())
        history["val_f1"].append(h["best_val_f1"])

    y_true, y_pred, _ = evaluate(current, val_loader)
    history["final_val_f1"] = f1_score(y_true, y_pred, zero_division=0)
    return current, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/compression/test_prune.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/compression/prune.py tests/compression/test_prune.py
git commit -m "feat: structured pruning with physical layer rebuild"
```

---

### Task 10: INT8 quantization

**Files:**
- Create: `src/edge_ids/compression/quantize.py`
- Test: `tests/compression/test_quantize.py`

**Interfaces:**
- Consumes: `StudentMLP`, loaders
- Produces:
  - `resolve_backend(preferred="auto") -> str`
  - `quantize_static(model, calib_loader, cfg) -> nn.Module`
  - `quantize_dynamic_model(model) -> nn.Module`
  - `saved_size_kb(model, path) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/compression/test_quantize.py
import numpy as np, pytest, torch
from edge_ids.config import Config
from edge_ids.compression.quantize import (quantize_dynamic_model, quantize_static,
                                           resolve_backend, saved_size_kb)
from edge_ids.data.dataset import make_loaders
from edge_ids.data.preprocess import Splits
from edge_ids.models.student import StudentMLP

def _splits(n=200, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = rng.normal(size=(n, d)).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def test_resolve_backend_returns_a_supported_engine():
    # This build reports only ['onednn']; a hardcoded fbgemm would fail at convert time.
    assert resolve_backend("auto") in torch.backends.quantized.supported_engines

def test_resolve_backend_honours_an_explicit_supported_choice():
    engine = torch.backends.quantized.supported_engines[0]
    assert resolve_backend(engine) == engine

def test_resolve_backend_falls_back_when_preference_is_unavailable():
    assert resolve_backend("definitely-not-a-backend") in torch.backends.quantized.supported_engines

def test_dynamic_quantization_shrinks_the_saved_model(tmp_path):
    m = StudentMLP(64, hidden=(128, 64)).eval()
    q = quantize_dynamic_model(m)
    assert saved_size_kb(q, tmp_path / "q.pt") < saved_size_kb(m, tmp_path / "f.pt")

def test_dynamic_quantized_model_still_predicts():
    m = StudentMLP(6, hidden=(16, 8)).eval()
    assert quantize_dynamic_model(m)(torch.randn(4, 6)).shape == (4, 2)

def test_static_quantization_shrinks_and_predicts(tmp_path):
    cfg = Config.default()
    cfg.quantize.calibration_batches = 2
    tr, _, _ = make_loaders(_splits(), batch_size=32, seed=42)
    m = StudentMLP(6, hidden=(16, 8), quantizable=True).eval()
    try:
        q = quantize_static(m, tr, cfg)
    except RuntimeError as exc:
        pytest.skip(f"quantization backend unavailable: {exc}")
    assert q(torch.randn(4, 6)).shape == (4, 2)
    assert saved_size_kb(q, tmp_path / "q.pt") < saved_size_kb(m, tmp_path / "f.pt")

def test_static_quantization_does_not_mutate_the_source_model():
    cfg = Config.default(); cfg.quantize.calibration_batches = 1
    tr, _, _ = make_loaders(_splits(), batch_size=32, seed=42)
    m = StudentMLP(6, hidden=(16, 8), quantizable=True).eval()
    before = m.net[0].weight.detach().clone()
    try:
        quantize_static(m, tr, cfg)
    except RuntimeError:
        pytest.skip("quantization backend unavailable")
    assert torch.allclose(before, m.net[0].weight)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/compression/test_quantize.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Post-training INT8 quantization.

Static PTQ quantizes weights *and* activations, which is what the design calls
for. Dynamic PTQ quantizes weights only, needs no calibration, and serves as a
robust fallback when the fbgemm backend is unavailable.
"""
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.ao.quantization as tq

def saved_size_kb(model, path) -> float:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)     # measured bytes, never computed
    return path.stat().st_size / 1024.0

BACKEND_PREFERENCE = ("fbgemm", "onednn", "qnnpack")

def resolve_backend(preferred: str = "auto") -> str:
    """Pick a quantization engine this torch build actually ships.

    Builds vary: fbgemm is common on x86 Linux but absent from some Windows
    wheels, which expose only onednn. Hardcoding one fails at convert() time.
    """
    supported = list(torch.backends.quantized.supported_engines)
    supported = [e for e in supported if e != "none"]
    if preferred != "auto" and preferred in supported:
        return preferred
    for candidate in BACKEND_PREFERENCE:
        if candidate in supported:
            return candidate
    if not supported:
        raise RuntimeError("This torch build supports no quantization engine.")
    return supported[0]

def _fuse_linear_relu(model):
    """Fuse each Linear+ReLU pair so the quantized graph has fewer requant steps."""
    pairs = []
    modules = list(model.net)
    for i in range(len(modules) - 1):
        if isinstance(modules[i], nn.Linear) and isinstance(modules[i + 1], nn.ReLU):
            pairs.append([f"net.{i}", f"net.{i + 1}"])
    return tq.fuse_modules(model, pairs, inplace=True) if pairs else model

def quantize_static(model, calib_loader, cfg):
    backend = resolve_backend(cfg.quantize.backend)
    m = copy.deepcopy(model)
    m.quantizable = True
    m.eval()
    torch.backends.quantized.engine = backend
    _fuse_linear_relu(m)
    m.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= cfg.quantize.calibration_batches:
                break
            m(x)                                  # calibration uses TRAINING data only
    tq.convert(m, inplace=True)
    return m

def quantize_dynamic_model(model):
    m = copy.deepcopy(model).eval()
    return tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/compression/test_quantize.py -v`
Expected: 7 passed (static tests skip only if no engine supports the fused graph)

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/compression/quantize.py tests/compression/test_quantize.py
git commit -m "feat: static and dynamic INT8 post-training quantization"
```

---

### Task 11: Classification metrics

**Files:**
- Create: `src/edge_ids/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: outputs of `evaluate`
- Produces: `classification_metrics(y_true, y_pred, y_prob) -> dict` with keys `accuracy, precision, recall, f1, roc_auc, fpr, tn, fp, fn, tp`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_metrics.py
import numpy as np
from edge_ids.evaluation.metrics import classification_metrics

def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 0, 1])
    m = classification_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["fpr"] == 0.0

def test_false_positive_rate_is_computed_from_benign_flows():
    # 2 benign, both wrongly flagged as attacks -> every legitimate device blocked.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    m = classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["fpr"] == 1.0 and m["fp"] == 2 and m["tn"] == 0

def test_confusion_counts_sum_to_sample_count():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    m = classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 50

def test_single_class_input_does_not_crash_roc_auc():
    y = np.zeros(6, dtype=int)
    m = classification_metrics(y, y, np.zeros(6))
    assert np.isnan(m["roc_auc"]) or 0.0 <= m["roc_auc"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Classification metrics. FPR is first-class: a false positive here means a
legitimate device gets cut off the network."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

def classification_metrics(y_true, y_pred, y_prob) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:                    # only one class present in y_true
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: classification metrics with false-positive rate"
```

---

### Task 12: Size and latency benchmarking

**Files:**
- Create: `src/edge_ids/evaluation/benchmark.py`
- Test: `tests/evaluation/test_benchmark.py`

**Interfaces:**
- Consumes: any `nn.Module`, `saved_size_kb`, `classification_metrics`, `evaluate`
- Produces:
  - `measure_latency(model, sample, cfg) -> dict` with `median_us`, `p95_us`
  - `benchmark_model(name, model, test_loader, sample, cfg, artifacts_dir) -> dict` — one full result row

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_benchmark.py
import numpy as np, torch
from edge_ids.config import Config
from edge_ids.data.dataset import make_loaders
from edge_ids.data.preprocess import Splits
from edge_ids.evaluation.benchmark import benchmark_model, measure_latency
from edge_ids.models.student import StudentMLP

def _splits(n=120, d=6):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    X = (rng.normal(size=(n, d)) + y[:, None] * 3.0).astype(np.float32)
    h = n // 4
    return Splits(X[:n-2*h], y[:n-2*h], X[n-2*h:n-h], y[n-2*h:n-h],
                  X[n-h:], y[n-h:], [f"f{i}" for i in range(d)], None)

def _fast_cfg():
    cfg = Config.default()
    cfg.benchmark.warmup, cfg.benchmark.iterations = 5, 20
    return cfg

def test_latency_returns_positive_median_and_p95():
    m = StudentMLP(6, hidden=(8, 4)).eval()
    r = measure_latency(m, torch.randn(1, 6), _fast_cfg())
    assert r["median_us"] > 0 and r["p95_us"] >= r["median_us"]

def test_benchmark_row_has_every_reported_field(tmp_path):
    s = _splits()
    _, _, te = make_loaders(s, batch_size=32, seed=42)
    row = benchmark_model("student", StudentMLP(6, hidden=(8, 4)).eval(),
                          te, torch.randn(1, 6), _fast_cfg(), tmp_path)
    for key in ("model", "accuracy", "f1", "roc_auc", "fpr",
                "params", "size_kb", "median_us", "p95_us"):
        assert key in row
    assert row["size_kb"] > 0 and row["params"] > 0

def test_bigger_model_reports_bigger_size(tmp_path):
    s = _splits()
    _, _, te = make_loaders(s, batch_size=32, seed=42)
    cfg, x = _fast_cfg(), torch.randn(1, 6)
    big = benchmark_model("big", StudentMLP(6, hidden=(64, 32)).eval(), te, x, cfg, tmp_path)
    small = benchmark_model("small", StudentMLP(6, hidden=(4, 2)).eval(), te, x, cfg, tmp_path)
    assert big["size_kb"] > small["size_kb"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_benchmark.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Measure what a model actually costs: bytes on disk and single-thread latency."""
import time
from pathlib import Path
import numpy as np
import torch
from ..compression.quantize import saved_size_kb
from ..models import count_parameters
from ..training.loops import evaluate
from .metrics import classification_metrics

def measure_latency(model, sample, cfg) -> dict:
    """Single-sample latency on one thread - a router core, not a laptop's parallelism."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.benchmark.threads)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(cfg.benchmark.warmup):      # discard warmup
                model(sample)
            times = []
            for _ in range(cfg.benchmark.iterations):
                t0 = time.perf_counter()
                model(sample)
                times.append((time.perf_counter() - t0) * 1e6)
    finally:
        torch.set_num_threads(prev_threads)
    arr = np.asarray(times)
    return {"median_us": float(np.median(arr)), "p95_us": float(np.percentile(arr, 95))}

def benchmark_model(name, model, test_loader, sample, cfg, artifacts_dir) -> dict:
    y_true, y_pred, y_prob = evaluate(model, test_loader)
    row = {"model": name}
    row.update(classification_metrics(y_true, y_pred, y_prob))
    row["params"] = count_parameters(model)
    row["size_kb"] = saved_size_kb(model, Path(artifacts_dir) / "models" / f"{name}.pt")
    row.update(measure_latency(model, sample, cfg))
    return row
```

Note: `count_parameters` returns 0 for converted quantized models whose weights are packed; fall back to the pre-quantization count when reporting, and rely on `size_kb` as the authoritative compression number.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_benchmark.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat: size and single-thread latency benchmarking"
```

---

### Task 13: Results reporting

**Files:**
- Create: `src/edge_ids/evaluation/report.py`
- Test: `tests/evaluation/test_report.py`

**Interfaces:**
- Consumes: list of benchmark rows from Task 12
- Produces:
  - `write_metrics_csv(rows, path)`
  - `write_markdown_table(rows, path)`
  - `plot_confusion_matrices(rows, path)`, `plot_size_vs_f1(rows, path)`, `plot_latency(rows, path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_report.py
import csv
from edge_ids.evaluation.report import (plot_latency, plot_size_vs_f1,
                                        write_markdown_table, write_metrics_csv)

ROWS = [
    {"model": "teacher", "accuracy": .99, "precision": .98, "recall": .97, "f1": .975,
     "roc_auc": .99, "fpr": .01, "tn": 90, "fp": 1, "fn": 2, "tp": 7,
     "params": 211000, "size_kb": 824.0, "median_us": 210.0, "p95_us": 260.0},
    {"model": "student-int8", "accuracy": .97, "precision": .95, "recall": .95, "f1": .95,
     "roc_auc": .97, "fpr": .03, "tn": 88, "fp": 3, "fn": 3, "tp": 6,
     "params": 1400, "size_kb": 3.1, "median_us": 40.0, "p95_us": 55.0},
]

def test_csv_has_one_row_per_model(tmp_path):
    p = tmp_path / "metrics.csv"
    write_metrics_csv(ROWS, p)
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2 and rows[0]["model"] == "teacher"

def test_markdown_table_includes_headers_and_models(tmp_path):
    p = tmp_path / "comparison.md"
    write_markdown_table(ROWS, p)
    text = p.read_text()
    assert "| model" in text and "teacher" in text and "student-int8" in text

def test_markdown_table_reports_compression_versus_teacher(tmp_path):
    p = tmp_path / "comparison.md"
    write_markdown_table(ROWS, p)
    assert "99.6" in p.read_text()      # 824.0 -> 3.1 KB

def test_plots_are_written(tmp_path):
    plot_size_vs_f1(ROWS, tmp_path / "size_vs_f1.png")
    plot_latency(ROWS, tmp_path / "latency.png")
    assert (tmp_path / "size_vs_f1.png").stat().st_size > 0
    assert (tmp_path / "latency.png").stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_report.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
"""Turn benchmark rows into the comparison table and plots."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMNS = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr",
           "params", "size_kb", "median_us", "p95_us", "tn", "fp", "fn", "tp"]

def _prepare(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); return path

def write_metrics_csv(rows, path) -> None:
    path = _prepare(path)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def write_markdown_table(rows, path) -> None:
    path = _prepare(path)
    baseline = rows[0]["size_kb"] if rows else 0.0
    head = ["model", "F1", "accuracy", "recall", "FPR", "ROC-AUC",
            "size (KB)", "vs teacher", "median (us)", "p95 (us)"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for r in rows:
        shrink = (1 - r["size_kb"] / baseline) * 100 if baseline else 0.0
        lines.append("| " + " | ".join([
            str(r["model"]), f"{r['f1']:.4f}", f"{r['accuracy']:.4f}",
            f"{r['recall']:.4f}", f"{r['fpr']:.4f}", f"{r['roc_auc']:.4f}",
            f"{r['size_kb']:.1f}", f"{shrink:.1f}%",
            f"{r['median_us']:.1f}", f"{r['p95_us']:.1f}"]) + " |")
    path.write_text("\n".join(lines) + "\n")

def plot_size_vs_f1(rows, path) -> None:
    path = _prepare(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["size_kb"], r["f1"], s=90)
        ax.annotate(r["model"], (r["size_kb"], r["f1"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("model size on disk (KB, log scale)")
    ax.set_ylabel("F1 on test set")
    ax.set_title("Compression vs. detection quality")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_latency(rows, path) -> None:
    path = _prepare(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [r["model"] for r in rows]
    ax.bar(names, [r["median_us"] for r in rows])
    ax.set_ylabel("median single-sample latency (us, 1 thread)")
    ax.set_title("Inference latency")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_confusion_matrices(rows, path) -> None:
    path = _prepare(path)
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.4))
    axes = [axes] if len(rows) == 1 else list(axes)
    for ax, r in zip(axes, rows):
        cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", fontsize=9)
        ax.set_title(r["model"], fontsize=10)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_report.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/edge_ids/evaluation/report.py tests/evaluation/test_report.py
git commit -m "feat: comparison table and result plots"
```

---

### Task 14: Pipeline orchestrator and README

**Files:**
- Create: `scripts/run_pipeline.py`, `README.md`
- Test: `tests/test_pipeline_end_to_end.py`

**Interfaces:**
- Consumes: every module above
- Produces: `run(cfg, sample_frac=None, use_synthetic=False) -> list[dict]` (the benchmark rows) plus a CLI

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_pipeline_end_to_end.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pipeline import run
from edge_ids.config import Config

def test_pipeline_produces_five_models_and_writes_results(tmp_path):
    cfg = Config.default()
    cfg.artifacts_dir = str(tmp_path)
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [32, 16], 8, 64
    cfg.student.hidden, cfg.student.max_epochs, cfg.student.batch_size = [16, 8], 8, 64
    cfg.prune.iterations, cfg.prune.finetune_epochs = 2, 2
    cfg.quantize.calibration_batches = 2
    cfg.benchmark.warmup, cfg.benchmark.iterations = 5, 20

    rows = run(cfg, use_synthetic=True)

    names = [r["model"] for r in rows]
    assert names[:4] == ["teacher", "student-plain", "student-kd", "student-pruned"]
    assert any("int8" in n for n in names)
    assert (tmp_path / "results" / "metrics.csv").exists()
    assert (tmp_path / "results" / "comparison.md").exists()

def test_compression_actually_shrinks_the_model(tmp_path):
    cfg = Config.default()
    cfg.artifacts_dir = str(tmp_path)
    cfg.teacher.hidden, cfg.teacher.max_epochs, cfg.teacher.batch_size = [32, 16], 5, 64
    cfg.student.hidden, cfg.student.max_epochs, cfg.student.batch_size = [16, 8], 5, 64
    cfg.prune.iterations, cfg.prune.finetune_epochs = 2, 2
    cfg.quantize.calibration_batches = 2
    cfg.benchmark.warmup, cfg.benchmark.iterations = 5, 20

    rows = {r["model"]: r for r in run(cfg, use_synthetic=True)}
    assert rows["student-kd"]["size_kb"] < rows["teacher"]["size_kb"]
    assert rows["student-pruned"]["size_kb"] < rows["student-kd"]["size_kb"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_end_to_end.py -v`
Expected: FAIL — `run_pipeline` does not exist

- [ ] **Step 3: Write `scripts/run_pipeline.py`**

```python
"""Run the whole pipeline: data -> teacher -> students -> prune -> quantize -> benchmark."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_ids.compression.prune import iterative_prune
from edge_ids.compression.quantize import quantize_dynamic_model, quantize_static
from edge_ids.config import Config
from edge_ids.data.dataset import make_loaders
from edge_ids.data.download import ensure_dataset, find_csvs
from edge_ids.data.preprocess import build_splits, load_raw
from edge_ids.evaluation.benchmark import benchmark_model
from edge_ids.evaluation.report import (plot_confusion_matrices, plot_latency,
                                        plot_size_vs_f1, write_markdown_table,
                                        write_metrics_csv)
from edge_ids.models import count_parameters
from edge_ids.training.distill import train_student
from edge_ids.training.loops import set_seed
from edge_ids.training.train_teacher import train_teacher

def _synthetic_frame(n=6000, d=20, seed=42) -> pd.DataFrame:
    """Stand-in traffic for smoke tests: two overlapping Gaussian populations."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)          # imbalanced, like real traffic
    X = rng.normal(size=(n, d)) + y[:, None] * rng.normal(1.2, 0.4, d)
    df = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(d)])
    df["Label"] = np.where(y == 1, "Bot", "BENIGN")
    return df

def run(cfg: Config, sample_frac=None, use_synthetic: bool = False) -> list:
    set_seed(cfg.data.seed)
    if sample_frac is not None:
        cfg.data.sample_frac = sample_frac

    if use_synthetic:
        df = _synthetic_frame()
    else:
        raw_dir = ensure_dataset(cfg.data)
        df = load_raw(find_csvs(raw_dir), cfg.data.sample_frac, cfg.data.seed)

    splits = build_splits(df, cfg.data)
    print(f"features: {splits.n_features}  train: {len(splits.y_train):,}  "
          f"attack rate: {splits.y_train.mean():.3f}")

    teacher, _ = train_teacher(splits, cfg)
    student_plain, _ = train_student(splits, cfg, teacher=None)
    student_kd, _ = train_student(splits, cfg, teacher=teacher)
    pruned, prune_hist = iterative_prune(student_kd, splits, cfg, teacher=teacher)
    print(f"pruned hidden sizes: {prune_hist['hidden_sizes']}")

    train_loader, _, test_loader = make_loaders(splits, cfg.student.batch_size, cfg.data.seed)
    sample = torch.from_numpy(splits.X_test[:1]).float()

    candidates = [("teacher", teacher), ("student-plain", student_plain),
                  ("student-kd", student_kd), ("student-pruned", pruned)]

    pruned_params = count_parameters(pruned)
    try:
        static_q = quantize_static(pruned, train_loader, cfg)
        candidates.append(("student-pruned-int8-static", static_q))
    except Exception as exc:                       # fbgemm unavailable on this box
        print(f"static quantization unavailable ({exc}); using dynamic only")
    candidates.append(("student-pruned-int8-dynamic", quantize_dynamic_model(pruned)))

    artifacts = Path(cfg.artifacts_dir)
    rows = []
    for name, model in candidates:
        row = benchmark_model(name, model, test_loader, sample, cfg, artifacts)
        if row["params"] == 0:                     # quantized weights are packed
            row["params"] = pruned_params
        rows.append(row)
        print(f"{name:30s} F1 {row['f1']:.4f}  FPR {row['fpr']:.4f}  "
              f"{row['size_kb']:8.1f} KB  {row['median_us']:7.1f} us")

    results = artifacts / "results"
    write_metrics_csv(rows, results / "metrics.csv")
    write_markdown_table(rows, results / "comparison.md")
    plot_size_vs_f1(rows, results / "size_vs_f1.png")
    plot_latency(rows, results / "latency.png")
    plot_confusion_matrices(rows, results / "confusion_matrices.png")
    print(f"\nresults written to {results.resolve()}")
    return rows

def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-optimized IDS pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="fraction of flows to use, e.g. 0.05 for a quick run")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data instead of CIC-IDS2017")
    args = ap.parse_args()
    cfg = Config.load(args.config) if Path(args.config).exists() else Config.default()
    run(cfg, sample_frac=args.sample_frac, use_synthetic=args.synthetic)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the end-to-end tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pipeline_end_to_end.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests pass

- [ ] **Step 6: Write `README.md`**

Cover: what the project does, the five models and why the plain student matters, setup with `uv`, how to get the dataset (auto and manual), how to run (`--synthetic` smoke test, `--sample-frac 0.05` quick run, full run), where results land, and a short "how to read the results" section including the honest note from spec §9 about INT8 latency.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_pipeline.py README.md tests/test_pipeline_end_to_end.py
git commit -m "feat: end-to-end pipeline orchestrator and documentation"
```

---

## Verification

After Task 14, before reporting completion:

1. `.venv\Scripts\python.exe -m pytest -v` — full suite green, with actual output shown.
2. `.venv\Scripts\python.exe scripts/run_pipeline.py --synthetic` — completes and writes all five result files.
3. Open `artifacts/results/comparison.md` and confirm `student-pruned` is genuinely smaller than `student-kd` in KB. If it is not, pruning is masking rather than rebuilding — spec §6 — and Task 9 is wrong.
4. Only then run against real CIC-IDS2017 data, starting with `--sample-frac 0.05`.

## Self-Review Notes

**Spec coverage:** §3 defects → Task 3 (one test per defect); §3 leakage → Tasks 1, 3; §4 models → Task 5; §4.3 five models → Task 14; §5 KD and T² → Task 8; §6 physical rebuild → Task 9; §7 static + dynamic PTQ → Task 10; §8 metrics/size/latency → Tasks 11–13; §9 honest latency note → README, Task 14 Step 6; §13 defaults → Task 1.

**Type consistency:** `Splits` fields are constructed identically in every test helper. `loss_fn(logits, targets, inputs)` is the single signature used by `train_model`, `train_teacher`, `train_student`, and `iterative_prune`. `saved_size_kb` is defined in `quantize.py` and imported by `benchmark.py` — one definition, one name.

**Known follow-up:** `count_parameters` returns 0 on converted quantized models; Task 14 substitutes the pre-quantization count and the plan notes that `size_kb` is the authoritative compression figure.
