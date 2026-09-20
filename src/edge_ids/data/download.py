"""Fetch the labelled flow data.

Two sources are supported:

- **ids2018** (default) - CSE-CIC-IDS2018, whose processed CSVs live in a public
  AWS S3 bucket that can be fetched directly over HTTPS.
- **ids2017** - CIC-IDS2017's MachineLearningCSV.zip. As of this writing the CIC
  host answers every path with an HTML form page rather than the archive, so this
  route generally requires a manual download; the error says so explicitly.
"""

import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

DATA_SUFFIXES = (".csv", ".parquet")
ZIP_MAGIC = b"PK\x03\x04"

IDS2018_BASE = "https://cse-cic-ids2018.s3.amazonaws.com"
IDS2018_PREFIX = "Processed Traffic Data for ML Algorithms"

# Which attack each capture day carries. Friday-02-03-2018 is the botnet day and
# the one this project cares about most.
IDS2018_DAYS = {
    "Wednesday-14-02-2018": "FTP-BruteForce, SSH-Bruteforce",
    "Thursday-15-02-2018": "DoS-GoldenEye, DoS-Slowloris",
    "Friday-16-02-2018": "DoS-SlowHTTPTest, DoS-Hulk",
    "Thuesday-20-02-2018": "DDoS-LOIC-HTTP",
    "Wednesday-21-02-2018": "DDoS-LOIC-UDP, DDoS-HOIC",
    "Thursday-22-02-2018": "BruteForce-Web, BruteForce-XSS, SQL-Injection",
    "Friday-23-02-2018": "BruteForce-Web, BruteForce-XSS, SQL-Injection",
    "Wednesday-28-02-2018": "Infiltration",
    "Thursday-01-03-2018": "Infiltration",
    "Friday-02-03-2018": "Bot",
}


def find_data_files(root) -> list[Path]:
    """Every CSV or Parquet file under `root`, at any depth."""
    root = Path(root)
    found: list[Path] = []
    for suffix in DATA_SUFFIXES:
        found.extend(root.rglob(f"*{suffix}"))
    return sorted(found)


def find_csvs(root) -> list[Path]:
    """Backwards-compatible alias for find_data_files."""
    return find_data_files(root)


def ids2018_url(filename: str) -> str:
    """Public S3 URL for one processed-traffic CSV.

    The bucket prefix contains spaces, which must be percent-encoded or S3
    answers with a signature error rather than the file.
    """
    return f"{IDS2018_BASE}/{quote(IDS2018_PREFIX)}/{quote(filename)}"


def ensure_dataset(cfg) -> Path:
    """Return the directory holding the flow files, downloading them if needed."""
    raw = Path(cfg.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    if find_data_files(raw):
        return raw

    source = getattr(cfg, "source", "ids2018")
    if source == "ids2018":
        return _ensure_ids2018(cfg, raw)
    if source == "ids2017":
        return _ensure_ids2017(cfg, raw)
    raise ValueError(
        f"Unknown data source '{source}'. Use 'ids2018' or 'ids2017'."
    )


def _ensure_ids2018(cfg, raw: Path) -> Path:
    for filename in cfg.files:
        destination = raw / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        url = ids2018_url(filename)
        try:
            _download(url, destination)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download '{filename}' from CSE-CIC-IDS2018 ({exc}).\n"
                f"Download it manually from\n  {url}\n"
                f"and place it in: {raw.resolve()}"
            ) from exc

    if not find_data_files(raw):
        raise RuntimeError(f"No flow files found under {raw.resolve()}")
    return raw


def _ensure_ids2017(cfg, raw: Path) -> Path:
    archive = raw / "MachineLearningCSV.zip"
    try:
        _download(cfg.url, archive, expect_zip=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)
        archive.unlink(missing_ok=True)
    except Exception as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download CIC-IDS2017 automatically ({exc}).\n"
            f"The CIC host now serves a download form rather than the archive, so\n"
            f"this usually has to be done manually: fetch MachineLearningCSV.zip from\n"
            f"  {cfg.url}\n"
            f"and extract the CSVs into: {raw.resolve()}"
        ) from exc

    if not find_data_files(raw):
        raise RuntimeError(
            f"Archive extracted but no CSVs were found under {raw.resolve()}."
        )
    return raw


def _download(url: str, dest: Path, timeout: int = 60, expect_zip: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        # A restructured host answers 200 with an HTML landing page rather than a
        # 404, so status alone does not mean the data arrived. Without this check
        # the failure surfaces later as an opaque parse error.
        content_type = response.headers.get("content-type", "")
        if "html" in content_type.lower():
            raise RuntimeError(
                f"server returned an HTML page, not data "
                f"(content-type: {content_type}) - the URL is probably stale"
            )

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name[:40]
        ) as bar:
            first = True
            for chunk in response.iter_content(chunk_size=1 << 20):
                if first and expect_zip and chunk[:4] != ZIP_MAGIC:
                    raise RuntimeError("downloaded content is not a zip archive")
                first = False
                fh.write(chunk)
                bar.update(len(chunk))
