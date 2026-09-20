import pytest

from edge_ids.config import DataConfig
from edge_ids.data.download import (
    IDS2018_DAYS,
    ensure_dataset,
    find_data_files,
    ids2018_url,
)


def test_find_data_files_locates_nested_csvs(tmp_path):
    nested = tmp_path / "MachineLearningCVE"
    nested.mkdir()
    (nested / "Monday.pcap_ISCX.csv").write_text("a,b\n1,2\n")
    (nested / "notes.txt").write_text("ignore me")
    found = find_data_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "Monday.pcap_ISCX.csv"


def test_find_data_files_also_locates_parquet(tmp_path):
    (tmp_path / "flows.parquet").write_bytes(b"PAR1")
    (tmp_path / "flows.csv").write_text("a\n1\n")
    assert {p.suffix for p in find_data_files(tmp_path)} == {".csv", ".parquet"}


def test_ensure_dataset_skips_download_when_data_present(tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n")
    cfg = DataConfig(raw_dir=str(tmp_path))
    assert ensure_dataset(cfg) == tmp_path


def test_ids2018_url_percent_encodes_the_spaces_in_the_prefix():
    url = ids2018_url("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
    assert url.startswith("https://cse-cic-ids2018.s3.amazonaws.com/")
    assert " " not in url
    assert "Processed%20Traffic%20Data%20for%20ML%20Algorithms" in url
    assert url.endswith("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")


def test_the_botnet_day_is_documented():
    # The whole project is about botnet detection; this is the file that carries
    # the Bot label, so a reader should not have to guess which day to fetch.
    assert IDS2018_DAYS["Friday-02-03-2018"] == "Bot"


def test_ensure_dataset_raises_actionable_error_when_unavailable(tmp_path):
    cfg = DataConfig(
        raw_dir=str(tmp_path),
        source="ids2018",
        files=["nope.csv"],
        url="http://127.0.0.1:1/",
    )
    with pytest.raises(RuntimeError, match="manually"):
        ensure_dataset(cfg)


def test_ids2017_zip_path_still_raises_actionably(tmp_path):
    cfg = DataConfig(
        raw_dir=str(tmp_path), source="ids2017", url="http://127.0.0.1:1/nope.zip"
    )
    with pytest.raises(RuntimeError, match="manually"):
        ensure_dataset(cfg)


def test_unknown_source_is_rejected(tmp_path):
    cfg = DataConfig(raw_dir=str(tmp_path), source="not-a-dataset")
    with pytest.raises(ValueError, match="not-a-dataset"):
        ensure_dataset(cfg)
