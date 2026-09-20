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
    df = pd.DataFrame(
        {"Source IP": ["1.1.1.1"], "Destination Port": [80], "Flow Duration": [5]}
    )
    out = pp.drop_leakage_columns(df, ["Source IP", "Destination Port", "Timestamp"])
    assert list(out.columns) == ["Flow Duration"]


def test_replaces_infinities_and_drops_nan_rows():
    df = pd.DataFrame(
        {"Flow Bytes/s": [1.0, np.inf, -np.inf, np.nan], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    out = pp.clean_numeric(df)
    assert len(out) == 1
    assert np.isfinite(out.to_numpy()).all()


def test_binarizes_labels_benign_to_zero_attacks_to_one():
    s = pd.Series(["BENIGN", " BENIGN ", "Bot", "DDoS", "Web Attack \x96 Brute Force"])
    assert pp.binarize_labels(s).tolist() == [0, 0, 1, 1, 1]


def test_drops_zero_variance_columns():
    df = pd.DataFrame({"constant": [3, 3, 3], "varying": [1, 2, 3]})
    assert list(pp.drop_constant_columns(df).columns) == ["varying"]


def _synthetic_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "Flow Duration": rng.normal(y * 3, 1, n),
            "Total Fwd Packets": rng.normal(y * -2, 1, n),
            "Source IP": ["10.0.0.1"] * n,
            "Label": np.where(y == 1, "Bot", "BENIGN"),
        }
    )


def test_scaler_is_fitted_on_training_split_only():
    splits = pp.build_splits(_synthetic_frame(), DataConfig())
    # Standardised by its own statistics, the training split centres on zero...
    assert splits.X_train.mean() == pytest.approx(0.0, abs=1e-4)
    # ...while the validation split, standardised by *train* statistics, does not.
    # Fitting on the full frame before splitting would centre both.
    assert abs(splits.X_val.mean()) > 1e-3


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
    every = np.concatenate([splits.X_train, splits.X_val, splits.X_test])
    # The outlier is clipped rather than passed through to dominate the
    # quantization calibration range.
    assert every.max() == pytest.approx(5.0)
    assert every.min() >= -5.0


def test_coerces_junk_values_to_nan_and_drops_those_rows():
    df = _synthetic_frame(n=200)
    df["Flow Duration"] = df["Flow Duration"].astype(object)
    df.loc[3, "Flow Duration"] = "not-a-number"
    splits = pp.build_splits(df, DataConfig())
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total == 199
    assert "Flow Duration" in splits.feature_names


def test_embedded_header_rows_are_dropped_not_scored_as_attacks():
    # IDS2018 CSVs carry repeated header rows mid-file (25 of them in the
    # Infiltration capture). Their Label cell reads "Label", which is != BENIGN
    # and would therefore be scored as an attack. They are caught only because
    # their feature cells are column names rather than numbers - so this pins
    # that behaviour against a refactor of the coercion step.
    df = _synthetic_frame(n=200)
    header_row = pd.DataFrame([{c: c for c in df.columns}])
    polluted = pd.concat([df, header_row], ignore_index=True)
    assert pp.binarize_labels(header_row["Label"]).tolist() == [1]  # naive path mislabels it

    splits = pp.build_splits(polluted, DataConfig())
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total == 200


def test_duplicate_rows_are_dropped_before_splitting():
    df = _synthetic_frame(n=200)
    doubled = pd.concat([df, df], ignore_index=True)
    once = pp.build_splits(df, DataConfig())
    twice = pp.build_splits(doubled, DataConfig())
    n = lambda s: len(s.y_train) + len(s.y_val) + len(s.y_test)
    assert n(twice) == n(once)


def test_split_proportions_and_stratification():
    splits = pp.build_splits(_synthetic_frame(n=1000), DataConfig())
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert abs(len(splits.y_test) / total - 0.2) < 0.02
    rates = [s.mean() for s in (splits.y_train, splits.y_val, splits.y_test)]
    assert max(rates) - min(rates) < 0.05


def test_missing_label_column_raises_a_clear_error():
    with pytest.raises(ValueError, match="Label"):
        pp.build_splits(pd.DataFrame({"Flow Duration": [1.0, 2.0]}), DataConfig())


def test_roundtrip_save_and_load(tmp_path):
    splits = pp.build_splits(_synthetic_frame(), DataConfig())
    pp.save_splits(splits, tmp_path)
    back = pp.load_splits(tmp_path)
    assert np.allclose(back.X_train, splits.X_train)
    assert back.feature_names == splits.feature_names
    assert back.scaler is not None
