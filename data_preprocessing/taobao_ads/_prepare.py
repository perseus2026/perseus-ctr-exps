# ruff: noqa: INP001, PLC0415

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import yaml
from ._download import RAW_FILES
from data_preprocessing._cross_domain import write_partitioned

logger = logging.getLogger(__name__)
PROFILE = [
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
]
FEATURES = [
    "item",
    "brand",
    "category",
    "user_id",
    *PROFILE,
    "customer",
    "campaign_id",
    "price_bucket",
    "pid",
]
EVENT_FIELDS = ["item", "brand", "category", "event"]


def make_configs(templates: Path, output: Path, *, train_test: bool = False) -> dict:
    configs = {}
    for model in ["perseus", "est", "hemix", "hyformer"]:
        for history in ["single", "cross"]:
            config = yaml.safe_load((templates / f"{model}_{history}.yaml").read_text())
            if train_test:
                config["training"]["num_epochs"] = 1
                config["training"].pop("early_stopping", None)
            else:
                config["training"]["num_epochs"] = 3
                config["training"]["early_stopping"] = {
                    "metric": "logloss/click",
                    "group": "overall",
                    "mode": "min",
                    "patience": 1,
                }
            (output / f"{model}_{history}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
            configs[f"{model}_{history}"] = config
    return configs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument(
        "--protocol",
        choices=["train-valid-test", "train-test-6-1"],
        default="train-test-6-1",
    )
    prepare(parser.parse_args())


def prepare(args: argparse.Namespace) -> None:
    from perseus import core

    train_test = args.protocol == "train-test-6-1"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configs").mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    raw = {p.name: p for p in args.raw.rglob("*.csv")}
    for filename in RAW_FILES:
        if filename not in raw or not raw[filename].is_file() or raw[filename].stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Taobao source: {filename}")
    if not args.history.is_file() or args.history.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty Taobao history: {args.history}")
    ads = (
        pl.read_csv(raw["ad_feature.csv"])
        .rename(str.strip)
        .with_columns(
            pl.col("brand").cast(pl.String),
            pl.col("cate_id").cast(pl.String).alias("category"),
            ((pl.col("price").clip(lower_bound=0) + 1).log() * 10)
            .floor()
            .cast(pl.Int64)
            .cast(pl.String)
            .alias("price_bucket"),
        )
    )
    samples = (
        pl.read_csv(raw["raw_sample.csv"], row_index_name="raw_row")
        .rename(str.strip)
        .join(ads, on="adgroup_id", how="left", validate="m:1", maintain_order="left")
        .filter(pl.col("cate_id") < 2000)
        .with_columns(
            pl.from_epoch("time_stamp", time_unit="s")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Shanghai")
            .dt.date()
            .cast(pl.String)
            .alias("day"),
        )
        .filter(
            pl.col("day").is_between(
                pl.lit("2017-05-07"), pl.lit("2017-05-13"), closed="both"
            )
        )
        .join(
            pl.read_csv(raw["user_profile.csv"]).rename(str.strip),
            left_on="user",
            right_on="userid",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .with_columns(
            pl.col("user").cast(pl.String).alias("client_id"),
            pl.col("user").cast(pl.String).alias("user_id"),
            pl.concat_str(pl.lit("ad:"), pl.col("adgroup_id").cast(pl.String)).alias(
                "item"
            ),
            pl.concat_str(pl.lit("taobao/"), pl.col("raw_row").cast(pl.String)).alias(
                "sample_id"
            ),
            pl.from_epoch("time_stamp", time_unit="s")
            .cast(pl.Datetime("ns"))
            .alias("timestamp"),
            pl.when(pl.col("day") == "2017-05-13")
            .then(pl.lit("test"))
            .when((pl.col("day") == "2017-05-12") & pl.lit(not train_test))
            .then(pl.lit("valid"))
            .otherwise(pl.lit("train"))
            .alias("split"),
            pl.col("clk").cast(pl.Int32).alias("click"),
            (pl.col("user") % args.shards).cast(pl.UInt16).alias("shard"),
        )
        .with_columns(pl.col(FEATURES).cast(pl.String))
        .select(
            "client_id", "timestamp", "sample_id", "split", *FEATURES, "click", "shard"
        )
    )
    if set(samples["click"].unique().to_list()) != {0, 1}:
        raise ValueError("Expected observed binary click labels")
    clients = samples.select("client_id").unique()
    members = clients.select(pl.col("client_id").cast(pl.UInt64).alias("user"))
    samples.write_parquet(args.scratch / "samples.pq")
    logger.info(
        "Target impressions: %s",
        samples.group_by("split").agg(pl.len(), pl.col("click").sum()).to_dicts(),
    )
    del samples

    def event_frames() -> Iterator[pl.DataFrame]:
        offset = 0
        for batch in pq.ParquetFile(args.history).iter_batches(batch_size=1_000_000):
            frame = pl.from_arrow(batch).with_row_index("source_row", offset=offset)
            offset += batch.num_rows
            frame = frame.join(members, on="user", how="semi").filter(
                pl.col("time_stamp").is_between(1492790400, 1494691200, closed="left"),
            )
            if frame.is_empty():
                continue
            frame = frame.select(
                pl.col("user").cast(pl.String).alias("client_id"),
                pl.from_epoch("time_stamp", time_unit="s")
                .cast(pl.Datetime("ns"))
                .alias("timestamp"),
                pl.lit(None, pl.String).alias("item"),
                pl.col("brand", "cate").cast(pl.String),
                pl.concat_str(
                    pl.when(pl.col("cate").cast(pl.Int64) < 2000)
                    .then(pl.lit("target-"))
                    .otherwise(pl.lit("source-")),
                    pl.col("btag"),
                ).alias("event"),
                "source_row",
                (pl.col("user") % args.shards).cast(pl.UInt16).alias("shard"),
            ).rename({"cate": "category"})
            yield frame
            if offset % 50_000_000 == 0:
                logger.info("Read %s behavior rows", offset)

    write_partitioned(event_frames(), args.scratch / "events")
    configs = make_configs(
        args.templates, args.output / "configs", train_test=train_test
    )
    config = core.Config(**configs["est_cross"])
    observers = {
        name: core.encoders.registry[desc.type].observer()
        for name, desc in config.encoders.items()
    }
    train = pl.read_parquet(args.scratch / "samples.pq").filter(
        pl.col("split") == "train"
    )
    bounds = train.group_by("client_id").agg(pl.col("timestamp").max().alias("cutoff"))
    for name in FEATURES:
        observers[name].observe(train[name])
    for shard in range(args.shards):
        events = pl.read_parquet(args.scratch / f"events/shard={shard}/*.parquet")
        past = events.join(bounds, on="client_id", how="inner").filter(
            pl.col("timestamp") < pl.col("cutoff")
        )
        for name in EVENT_FIELDS:
            observers[name].observe(past[name])
        logger.info("Observed vocabularies: %s/%s", shard + 1, args.shards)
    preprocessors = {
        name: core.encoders.registry[desc.type].preprocessor.fit(
            observers[name], **desc.preprocessor
        )
        for name, desc in config.encoders.items()
    }
    preprocessor = core.tasks.registry["classification"].preprocessor(
        {"click": ["0", "1"]}
    )
    for history in ["single", "cross"]:
        destination = args.output / history
        core.checkpoint.Prepared(
            core.Config(**configs[f"est_{history}"]), preprocessors
        ).save(
            destination / "checkpoint",
        )
        preprocessor.save(destination / "dataset/preprocessor")
    stores = {
        history: core.EventStore(
            args.output / f"{history}/dataset/events", backend="disk"
        )
        for history in ["single", "cross"]
    }
    counts = {}
    with (
        stores["single"].open_writer() as single,
        stores["cross"].open_writer() as cross,
    ):
        for shard in range(args.shards):
            events = pl.read_parquet(
                args.scratch / f"events/shard={shard}/*.parquet"
            ).select(
                "client_id",
                "timestamp",
                *EVENT_FIELDS,
                "source_row",
            )
            users = clients.filter(
                pl.col("client_id").cast(pl.UInt64) % args.shards == shard
            )
            for kind, frame, writer in [
                (
                    "single",
                    events.filter(pl.col("event").str.starts_with("target-")),
                    single,
                ),
                ("cross", events, cross),
            ]:
                missing = users.join(
                    frame.select("client_id").unique(), on="client_id", how="anti"
                )
                dummy = missing.with_columns(
                    timestamp=pl.lit(
                        8_000_000_000_000_000_000, dtype=pl.Datetime("ns")
                    ),
                    **{name: pl.lit(None, pl.String) for name in EVENT_FIELDS},
                    source_row=pl.lit(0, pl.UInt32),
                )
                frame = (
                    pl.concat(
                        [frame, dummy.select(frame.columns)], how="vertical_relaxed"
                    )
                    .sort("client_id", "timestamp", "source_row")
                    .drop("source_row")
                )
                frame = frame.with_columns(
                    position=pl.int_range(pl.len(), 0, -1).over("client_id")
                )
                for name in EVENT_FIELDS:
                    frame = frame.with_columns(
                        preprocessors[name].static_transform(frame[name]).alias(name)
                    )
                writer.write(frame)
                counts[kind] = counts.get(kind, 0) + frame.height
            logger.info("Encoded histories: %s/%s", shard + 1, args.shards)
    samples = pl.read_parquet(args.scratch / "samples.pq")
    for name in FEATURES:
        samples = samples.with_columns(
            preprocessors[name].static_transform(samples[name]).alias(name)
        )
    samples = samples.select(
        "client_id",
        "timestamp",
        "sample_id",
        "split",
        pl.struct(FEATURES).alias("context"),
        pl.struct("click").alias("target"),
    )
    for fold, frame in [
        ("train", samples.filter(pl.col("split") == "train")),
        ("test", samples.filter(pl.col("split") != "train")),
    ]:
        if train_test:
            frame = frame.drop("split")
        path = args.output / f"cross/dataset/{fold}/samples.pq"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        other = args.output / f"single/dataset/{fold}/samples.pq"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.hardlink_to(path)
    report = {
        "protocol": "taobao_lcn_train_test_6_1_v1"
        if train_test
        else "taobao_lcn_category_split_reconstruction_v1",
        "paper": "2312.06424v2",
        "exact_paper_replication": False,
        "dates": {
            "train": "2017-05-07 through 2017-05-12"
            if train_test
            else "2017-05-07 through 2017-05-11",
            "valid": None if train_test else "2017-05-12",
            "test": "2017-05-13",
            "timezone": "Asia/Shanghai",
        },
        "source_category_threshold": 2000,
        "history_item_ids_available": False,
        "history_identity_fields": ["category", "brand"],
        "history_features": EVENT_FIELDS,
        "context_features": FEATURES,
        "train_only_vocabularies": True,
        "synthetic_negative_sampling": False,
        "folds": samples.group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("target").struct.field("click").sum().alias("clicks"),
        )
        .to_dicts(),
        "target_users": clients.height,
        "event_rows": counts,
        "history_length": 500,
    }
    (args.output / "preparation-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("Taobao prepared: %s", report)


if __name__ == "__main__":
    main()
