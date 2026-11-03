# ruff: noqa: INP001

import argparse
import json
import logging
from pathlib import Path

import polars as pl
# Pin the source revision without a separate source digest list.
REPO_ID = "Angrybird12/ad-display_click-data_taobao.com"
REVISION = "29af49a45b296680c66304a2a1c597cae3eb81f7"
RAW_FILES = ("ad_feature.csv", "raw_sample.csv", "user_profile.csv")

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
    from huggingface_hub import hf_hub_download

    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename in RAW_FILES:
        path = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
            filename=filename, local_dir=raw, token=False,
        ))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty downloaded source: {filename}")
        logger.info("Downloaded: %s", filename)
    samples = pl.scan_csv(raw / "raw_sample.csv")
    nonclick = "nonclk" if "nonclk" in samples.collect_schema() else "noclk"
    stats = (
        samples.select(
            rows=pl.len(),
            clicks=pl.col("clk").sum(),
            invalid_labels=(
                pl.col("clk").is_null() | ~pl.col("clk").is_in([0, 1])
            ).sum(),
            inconsistent_labels=(pl.col("clk") + pl.col(nonclick) != 1).sum(),
            first_timestamp=pl.col("time_stamp").min(),
            last_timestamp=pl.col("time_stamp").max(),
            users=pl.col("user").n_unique(),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if stats["invalid_labels"] or stats["inconsistent_labels"]:
        raise ValueError(f"Invalid observed click labels: {stats}")
    ads = pl.scan_csv(raw / "ad_feature.csv")
    profiles = pl.scan_csv(raw / "user_profile.csv")
    ads_stats = (
        ads.select(rows=pl.len(), unique_ids=pl.col("adgroup_id").n_unique())
        .collect()
        .row(0, named=True)
    )
    users_stats = (
        profiles.select(rows=pl.len(), unique_ids=pl.col("userid").n_unique())
        .collect()
        .row(0, named=True)
    )
    if (
        ads_stats["rows"] != ads_stats["unique_ids"]
        or users_stats["rows"] != users_stats["unique_ids"]
    ):
        raise ValueError("Duplicate identifiers in the raw feature tables")
    report = {
        "status": "three_raw_tables_collected; original_behavior_log_not_available",
        "source": {"repo_id": REPO_ID, "revision": REVISION},
        "samples": stats,
        "ads": ads_stats,
        "profiles": users_stats,
        "synthetic_negatives": False,
        "long_history_ready": False,
        "lcn_paper_protocol_reproduced": False,
    }
    (args.output / "collection-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("Taobao source report: %s", report)


if __name__ == "__main__":
    main()
