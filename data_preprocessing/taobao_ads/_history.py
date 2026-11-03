# ruff: noqa: INP001

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import httpx
import polars as pl
from ._download import RAW_FILES

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/zerotrustai/taobao-ad-display-click?datasetVersionNumber=1"
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())


def collect(args: argparse.Namespace) -> None:
    raw = args.output / "raw"
    tables = args.output / "tables"
    raw.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    archive = raw / "taobao-ad-display-click-v1.zip"
    temporary = archive.with_suffix(".partial")
    with (
        httpx.Client(follow_redirects=True, timeout=300) as client,
        client.stream("GET", SOURCE) as response,
    ):
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(
                f"Expected the full archive, received HTTP {response.status_code}"
            )
        with temporary.open("wb") as stream:
            written = 0
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                stream.write(chunk)
                written += len(chunk)
                if written % (512 * 1024 * 1024) < len(chunk):
                    logger.info("Downloaded %s bytes", written)
        if response.headers.get("Content-Length") and written != int(
            response.headers["Content-Length"]
        ):
            raise ValueError("Incomplete archive download")
    temporary.replace(archive)
    expected = {*RAW_FILES, "behavior_log.csv"}
    found = set()
    with (
        tempfile.TemporaryDirectory(dir=args.output.parent) as directory,
        ZipFile(archive) as zipped,
    ):
        workdir = Path(directory)
        for member in zipped.infolist():
            filename = Path(member.filename).name
            if filename not in expected or member.is_dir():
                continue
            if filename in found:
                raise ValueError(f"Duplicate archive member: {filename}")
            destination = workdir / filename
            # ZipFile validates each member's CRC during extraction.
            with zipped.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if destination.stat().st_size == 0:
                raise ValueError(f"Empty archive member: {filename}")
            found.add(filename)
        if found != expected:
            raise ValueError(f"Archive missing required CSV tables: {sorted(expected - found)}")
        history = pl.scan_csv(
            workdir / "behavior_log.csv",
            schema_overrides={
                "user": pl.UInt64,
                "time_stamp": pl.Int64,
                "btag": pl.String,
                "cate": pl.String,
                "brand": pl.String,
            },
        )
        history.sink_parquet(
            tables / "behavior_log.pq", compression="zstd", maintain_order=True
        )
    history = pl.scan_parquet(tables / "behavior_log.pq")
    stats = (
        history.select(
            rows=pl.len(),
            first_timestamp=pl.col("time_stamp").min(),
            last_timestamp=pl.col("time_stamp").max(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    actions = history.group_by("btag").len().collect(engine="streaming").to_dicts()
    report = {
        "status": "full_raw_archive_and_behavior_history_collected",
        "official_dataset": "https://tianchi.aliyun.com/dataset/56",
        "mirror_url": SOURCE,
        "mirror_version": 1,
        "history": stats,
        "actions": actions,
        "lcn_paper_protocol_reproduced": False,
    }
    (args.output / "collection-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("Complete raw Taobao report: %s", report)


if __name__ == "__main__":
    main()
