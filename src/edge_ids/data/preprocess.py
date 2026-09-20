"""Turn raw CIC-IDS2017 CSVs into scaled, split, leakage-free arrays.

CIC-IDS2017 is widely used and widely mishandled. Each defect handled here
(leading spaces in headers, non-UTF8 label bytes, infinities from zero-duration
flows, duplicate rows, constant columns) silently corrupts a naive read_csv
pipeline rather than raising.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_COLUMN = "Label"
BENIGN = "BENIGN"


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers - CIC-IDS2017 ships ' Flow Duration', not 'Flow Duration'."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_leakage_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Drop identifier columns if present, so either CSV distribution works."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Infinities come from flows of zero duration; the rows carrying them are dropped."""
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def binarize_labels(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper() != BENIGN).astype(np.int64)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Several flag counters are constant across the dataset; they carry no signal."""
    return df.loc[:, df.nunique(dropna=False) > 1]


def read_one(path) -> pd.DataFrame:
    """Read a single CSV or Parquet file of labelled flows.

    latin1 for CSV: the 'Web Attack - Brute Force' label carries a non-UTF8 byte
    that raises UnicodeDecodeError under the default encoding.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return normalize_columns(pd.read_parquet(path))
    return normalize_columns(pd.read_csv(path, encoding="latin1"))


def load_raw(paths, sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Read and concatenate the labelled flow files."""
    frames = [read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df


def build_splits(df: pd.DataFrame, cfg) -> Splits:
    df = normalize_columns(df)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"No '{LABEL_COLUMN}' column found; got {list(df.columns)[:10]}"
        )

    y = binarize_labels(df[LABEL_COLUMN])
    X = drop_leakage_columns(df.drop(columns=[LABEL_COLUMN]), cfg.drop_columns)

    # Coerce first, then keep numerics: real CSVs carry stray non-numeric tokens
    # in otherwise numeric columns, which would silently make the whole column object.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number])

    keep = clean_numeric(X).index
    X, y = X.loc[keep], y.loc[keep]

    unique = ~X.duplicated()
    X, y = X[unique], y[unique]

    X = drop_constant_columns(X)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    feature_names = list(X.columns)
    X_all = X.to_numpy(dtype=np.float64)
    y_all = y.to_numpy(dtype=np.int64)

    holdout = cfg.val_size + cfg.test_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_all, y_all, test_size=holdout, random_state=cfg.seed, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=cfg.test_size / holdout,
        random_state=cfg.seed,
        stratify=y_hold,
    )

    # Fitted on the training split ONLY. Fitting on the full frame first would
    # leak the test distribution into training.
    scaler = StandardScaler().fit(X_train)
    clip = cfg.clip

    def transform(a: np.ndarray) -> np.ndarray:
        return np.clip(scaler.transform(a), -clip, clip).astype(np.float32)

    return Splits(
        transform(X_train),
        y_train,
        transform(X_val),
        y_val,
        transform(X_test),
        y_test,
        feature_names,
        scaler,
    )


# Feature names go to JSON rather than a pickled object array, so loading never
# needs allow_pickle=True. joblib holds only the sklearn scaler, which this
# pipeline both writes and reads - it never loads one from an outside source.
def save_splits(splits: Splits, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "splits.npz",
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
    )
    (out / "feature_names.json").write_text(json.dumps(splits.feature_names))
    joblib.dump(splits.scaler, out / "scaler.joblib")


def load_splits(out_dir) -> Splits:
    out = Path(out_dir)
    d = np.load(out / "splits.npz")
    return Splits(
        d["X_train"],
        d["y_train"],
        d["X_val"],
        d["y_val"],
        d["X_test"],
        d["y_test"],
        json.loads((out / "feature_names.json").read_text()),
        joblib.load(out / "scaler.joblib"),
    )
