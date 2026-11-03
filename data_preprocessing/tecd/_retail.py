# ruff: noqa: INP001

import argparse
import json
import logging
from pathlib import Path

import polars as pl

TARGETS = {"click": "click", "cart": "added_to_cart", "order": "order"}
DAY_US = 86_400_000_000
GAP_US = DAY_US // 2
logger = logging.getLogger(__name__)


def prepare_samples(events: pl.DataFrame, day: int) -> pl.DataFrame:
    actions = set(events["action_type"].unique().to_list())
    if actions - {"view", *TARGETS.values()}:
        raise ValueError(f"Unexpected Retail actions: {actions}")
    if events.select(
        pl.any_horizontal(
            pl.col("user_id", "item_id", "timestamp", "action_type").is_null()
        ).any(),
    ).item():
        raise ValueError("Null identifiers, timestamps or actions")
    stamps = events["timestamp"].dt.total_microseconds()
    if len(events) and ((stamps // DAY_US) != day).any():
        raise ValueError(f"Events do not belong to day {day}")
    return events.with_row_index("source_row").select(
        pl.concat_str(pl.lit(f"retail/{day:05d}/"), pl.col("source_row")).alias(
            "sample_id"
        ),
        pl.col("user_id").cast(pl.String).alias("client_id"),
        (pl.col("timestamp").dt.total_microseconds() - GAP_US)
        .cast(pl.Datetime("us"))
        .alias("timestamp"),
        pl.col("timestamp")
        .dt.total_microseconds()
        .cast(pl.Datetime("us"))
        .alias("prediction_timestamp"),
        pl.struct(pl.col("item_id").cast(pl.String).alias("item")).alias("context"),
        pl.struct(
            (pl.col("action_type") == action).alias(name)
            for name, action in TARGETS.items()
        ).alias("target"),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


def prepare(args: argparse.Namespace) -> None:
    root = args.source / "raw/dataset/small"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "source-manifest.json").read_text())
    first, last = manifest["day_begin"], manifest["day_end"]
    days = last - first + 1
    valid_start, test_start = first + int(days * 0.8), first + int(days * 0.9)
    reports = []
    paths = sorted((root / "retail/events").glob("*.pq"))
    if not paths:
        raise ValueError("No Retail event files")
    for path in paths:
        day = int(path.stem)
        samples = prepare_samples(pl.read_parquet(path), day)
        if day < valid_start:
            split = "train"
        elif day < test_start:
            split = "valid"
        else:
            split = "test"
        before = samples.height
        if day in (valid_start, test_start):
            boundary = day * DAY_US + GAP_US
            samples = samples.filter(
                pl.col("prediction_timestamp").cast(pl.Int64) >= boundary
            )
        destination = args.output / "basis" / split / f"{day:05d}.pq"
        destination.parent.mkdir(parents=True, exist_ok=True)
        samples.write_parquet(destination, compression="zstd")
        report = {
            "day": day,
            "split": split,
            "rows": samples.height,
            "embargo_removed": before - samples.height,
            "positives": samples["target"].struct.unnest().sum().row(0, named=True),
        }
        reports.append(report)
        logger.info("Prepared: %s", report)
    report = {
        "task": "logged_event_action_prediction",
        "label_policy": {
            "view": [0, 0, 0],
            "click": [1, 0, 0],
            "added_to_cart": [0, 1, 0],
            "order": [0, 0, 1],
        },
        "label_order": list(TARGETS),
        "inferred_actions": False,
        "synthetic_negatives": False,
        "history_cutoff_field": "timestamp",
        "history_gap_hours": 12,
        "prediction_time_field": "prediction_timestamp",
        "train_days": [first, valid_start - 1],
        "valid_days": [valid_start, test_start - 1],
        "test_days": [test_start, last],
        "split_embargo_hours": 12,
        "complete_retail_days": [int(path.stem) for path in paths]
        == list(range(first, last + 1)),
        "model_encoders_fitted": False,
        "source_revision": manifest["revision"],
        "partitions": reports,
    }
    (args.output / "preparation-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
