# ruff: noqa: INP001, PLC0415

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import yaml
from data_preprocessing._cross_domain import checksum, write_partitioned
from ._events import FEATURES, normalize_events

logger = logging.getLogger(__name__)
LABELS = ["click", "cart", "order"]
GAP_US = 43_200_000_000


def chronological_cutoff(
    paths: list[Path], train_ratio: float = 0.8
) -> tuple[int, int]:
    sizes = [pq.ParquetFile(path).metadata.num_rows for path in paths]
    total = sum(sizes)
    if not total or not 0 < train_ratio < 1:
        raise ValueError(
            "Expected nonempty events and a train ratio between zero and one"
        )
    index = int(total * train_ratio)
    offset = 0
    for path, size in zip(paths, sizes, strict=True):
        if offset + size > index:
            timestamps = (
                pl.read_parquet(path, columns=["timestamp"])["timestamp"]
                .dt.total_microseconds()
                .sort()
            )
            return int(timestamps[index - offset]), total
        offset += size
    raise ValueError("Could not locate temporal split")


def configs_for(
    templates: Path, event_names: list[str], output: Path, batch: int
) -> dict:
    configs = {}
    for model in ["perseus", "est", "hemix", "hyformer"]:
        for history in ["single", "cross"]:
            config = yaml.safe_load((templates / f"{model}_{history}.yaml").read_text())
            names = (
                [name for name in event_names if name.startswith("retail-")]
                if history == "single"
                else event_names
            )
            config["events"] = {
                name: {"attributes": {"item": None, "brand": None}} for name in names
            }
            config["training"]["dataloader"]["batch_size"] = batch
            config["training"]["test_dataloader"]["batch_size"] = batch
            config["training"]["accumulation"] = 32768 // batch
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
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--payments", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--protocol",
        choices=["train-valid-test", "train-test-80-20"],
        default="train-test-80-20",
    )
    prepare(parser.parse_args())


def prepare(args: argparse.Namespace) -> None:
    from perseus import core

    if args.shards < 1 or args.shards > 128 or 32768 % args.batch_size:
        raise ValueError("Invalid shard count or global batch size")
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    (args.output / "configs").mkdir(exist_ok=True)
    raw = args.small / "raw/dataset/small"
    source_manifest = json.loads((args.small / "source-manifest.json").read_text())
    sources = {entry["path"]: entry for entry in source_manifest["files"]}
    retail_paths = sorted((raw / "retail/events").glob("*.pq"))
    train_test = args.protocol == "train-test-80-20"
    split_audit = {}
    if train_test:
        split_timestamp, raw_rows = chronological_cutoff(retail_paths)
        split_audit = {
            "train_ratio_requested": 0.8,
            "split_timestamp_us": split_timestamp,
            "raw_target_rows": raw_rows,
            "test_embargo_hours": 12,
            "embargo_removed": 0,
            "timestamp_ties_kept_together": True,
        }
    clients = (
        (
            pl.scan_parquet(retail_paths).select(
                pl.col("user_id").cast(pl.String).alias("client_id")
            )
            if train_test
            else pl.scan_parquet(args.small / "prepared/basis/*/*.pq").select(
                "client_id"
            )
        )
        .unique()
        .collect(
            engine="streaming",
        )
    )
    members = clients.select(pl.col("client_id").cast(pl.UInt64).alias("user_id"))
    clients.write_parquet(args.output / "target-clients.pq")
    profiles = pl.read_parquet(
        raw / "users.pq", columns=["user_id", "region", "socdem_cluster"]
    ).select(
        pl.col("user_id").cast(pl.String).alias("client_id"),
        pl.col("region", "socdem_cluster").cast(pl.String),
    )
    retail_items = pl.read_parquet(
        raw / "retail/items.pq", columns=["item_id", "brand_id"]
    ).select(
        pl.col("item_id").cast(pl.String).alias("candidate"),
        pl.col("brand_id").cast(pl.String).alias("brand"),
    )
    domain_counts, event_names, source_counts = {}, set(), {}

    def event_frames() -> Iterator[pl.DataFrame]:
        for domain in ["retail", "marketplace", "offers", "payments"]:
            items = None
            if domain == "payments":
                paths = sorted(
                    args.payments.rglob("filtered/small-payments/events/*.pq")
                )
            else:
                items = pl.read_parquet(
                    raw / domain / "items.pq", columns=["item_id", "brand_id"]
                ).with_columns(
                    pl.col("item_id").cast(pl.String),
                )
                paths = sorted((raw / domain / "events").glob("*.pq"))
            expected = 594 if domain == "payments" else 227
            if len(paths) != expected or len({p.name for p in paths}) != expected:
                raise ValueError(f"Missing or repeated dates: {domain}")
            domain_counts[domain] = 0
            for index, path in enumerate(paths, 1):
                expected_sha = (
                    json.loads(path.with_suffix(".json").read_text())["filtered_sha256"]
                    if domain == "payments"
                    else sources[f"dataset/small/{domain}/events/{path.name}"]["sha256"]
                )
                if checksum(path) != expected_sha:
                    raise ValueError(f"Stored source checksum mismatch: {path}")
                fields = (
                    ["user_id", "timestamp", "brand_id"]
                    if domain == "payments"
                    else [
                        "user_id",
                        "timestamp",
                        "item_id",
                        "action_type",
                    ]
                )
                frame = (
                    pl.scan_parquet(path)
                    .select(fields)
                    .join(members.lazy(), on="user_id", how="semi")
                    .collect(
                        engine="streaming",
                    )
                )
                frame = normalize_events(frame, domain, items)
                domain_counts[domain] += frame.height
                event_names.update(frame["event"].unique().to_list())
                if not frame.is_empty():
                    yield frame.with_columns(
                        (pl.col("client_id").cast(pl.UInt64) % args.shards)
                        .cast(pl.UInt16)
                        .alias("shard"),
                    )
                if index % 50 == 0:
                    logger.info(
                        "Normalized %s: %s/%s days, %s rows",
                        domain,
                        index,
                        len(paths),
                        domain_counts[domain],
                    )

    write_partitioned(event_frames(), args.scratch / "events")
    logger.info("Normalized histories complete: %s", domain_counts)

    def sample_frames() -> Iterator[pl.DataFrame]:
        for split in ["train", "valid", "test"]:
            source_counts[split] = {"rows": 0, **dict.fromkeys(LABELS, 0)}
        if train_test:
            from ._retail import prepare_samples

            inputs = [(None, path) for path in retail_paths]
        else:
            inputs = [
                (split, path)
                for split in ["train", "valid", "test"]
                for path in sorted((args.small / "prepared/basis" / split).glob("*.pq"))
            ]
        for split, path in inputs:
            frame = (
                prepare_samples(pl.read_parquet(path), int(path.stem))
                if train_test
                else pl.read_parquet(path)
            )
            if train_test:
                stamps = pl.col("prediction_timestamp").dt.epoch("us")
                before = frame.height
                frame = frame.filter(
                    (stamps < split_timestamp) | (stamps >= split_timestamp + GAP_US)
                )
                split_audit["embargo_removed"] += before - frame.height
                split_column = (
                    pl.when(stamps < split_timestamp)
                    .then(pl.lit("train"))
                    .otherwise(pl.lit("test"))
                )
            else:
                split_column = pl.lit(split)
            frame = (
                frame.with_columns(
                    pl.col("context").struct.field("item").alias("candidate")
                )
                .join(
                    retail_items,
                    on="candidate",
                    how="left",
                    validate="m:1",
                    maintain_order="left",
                )
                .join(
                    profiles,
                    on="client_id",
                    how="left",
                    validate="m:1",
                    maintain_order="left",
                )
                .with_columns(
                    pl.col("client_id").alias("user_id"),
                    pl.concat_str(pl.lit("retail:"), pl.col("candidate")).alias("item"),
                    pl.col("timestamp", "prediction_timestamp").cast(pl.Datetime("ns")),
                )
                .unnest("target")
                .select(
                    "client_id",
                    "timestamp",
                    "prediction_timestamp",
                    "sample_id",
                    *FEATURES,
                    *LABELS,
                )
                .with_columns(
                    split=split_column,
                    shard=(pl.col("client_id").cast(pl.UInt64) % args.shards).cast(
                        pl.UInt16
                    ),
                )
            )
            for kind, part in frame.partition_by("split", as_dict=True).items():
                source_counts[kind[0]]["rows"] += part.height
                for name, value in part.select(LABELS).sum().row(0, named=True).items():
                    source_counts[kind[0]][name] += value
            yield frame

    write_partitioned(sample_frames(), args.scratch / "samples")
    logger.info("Partitioned target examples: %s", source_counts)
    configs = configs_for(
        args.templates, sorted(event_names), args.output / "configs", args.batch_size
    )
    if train_test:
        for name, value in configs.items():
            value["training"].pop("early_stopping", None)
            value["training"]["save_epoch_states"] = False
            (args.output / f"configs/{name}.yaml").write_text(
                yaml.safe_dump(value, sort_keys=False)
            )
    config = core.Config(**configs["est_cross"])
    observers = {
        name: core.encoders.registry[desc.type].observer()
        for name, desc in config.encoders.items()
    }
    for shard in range(args.shards):
        events = (
            pl.read_parquet(args.scratch / f"events/shard={shard}/*.parquet")
            .select(
                "client_id",
                "timestamp",
                "item",
                "brand",
                "event",
            )
            .unique()
        )
        samples = (
            pl.scan_parquet(args.scratch / f"samples/shard={shard}/*.parquet")
            .filter(
                pl.col("split") == "train",
            )
            .collect(engine="streaming")
        )
        boundary = samples.group_by("client_id").agg(
            pl.col("timestamp").max().alias("last_train_cutoff")
        )
        train_events = events.join(boundary, on="client_id", how="inner").filter(
            pl.col("timestamp") < pl.col("last_train_cutoff"),
        )
        for feature in ["item", "brand", "event"]:
            observers[feature].observe(train_events[feature])
        for feature in FEATURES:
            observers[feature].observe(samples[feature])
        logger.info("Observed train vocabularies: shard %s/%s", shard + 1, args.shards)
    preprocessors = {
        name: core.encoders.registry[desc.type].preprocessor.fit(
            observers[name], **desc.preprocessor
        )
        for name, desc in config.encoders.items()
    }
    task = core.tasks.registry["classification"]
    target_preprocessor = task.preprocessor({name: ["0", "1"] for name in LABELS})
    for history in ["single", "cross"]:
        destination = args.output / history
        core.checkpoint.Prepared(
            core.Config(**configs[f"est_{history}"]), preprocessors
        ).save(
            destination / "checkpoint",
        )
        target_preprocessor.save(destination / "dataset/preprocessor")
    counts = {
        name: {"rows": 0, **dict.fromkeys(LABELS, 0)}
        for name in ["train", "valid", "test"]
    }
    writers, event_rows = {}, {"single": 0, "cross": 0}
    cross_store = core.EventStore(args.output / "cross/dataset/events", backend="disk")
    single_store = core.EventStore(
        args.output / "single/dataset/events", backend="disk"
    )
    with (
        cross_store.open_writer() as cross_writer,
        single_store.open_writer() as single_writer,
    ):
        for shard in range(args.shards):
            events = (
                pl.read_parquet(args.scratch / f"events/shard={shard}/*.parquet")
                .select("client_id", "timestamp", "item", "brand", "event")
                .unique()
                .sort(
                    "client_id",
                    "timestamp",
                    "event",
                    "item",
                    "brand",
                )
            )
            retail = events.filter(pl.col("event").str.starts_with("retail-"))
            for history, frame, writer in [
                ("single", retail, single_writer),
                ("cross", events, cross_writer),
            ]:
                frame = frame.with_columns(
                    position=pl.int_range(pl.len(), 0, -1).over("client_id")
                )
                for name in ["item", "brand", "event"]:
                    frame = frame.with_columns(
                        preprocessors[name].static_transform(frame[name]).alias(name)
                    )
                writer.write(frame)
                event_rows[history] += frame.height
            samples = pl.read_parquet(args.scratch / f"samples/shard={shard}/*.parquet")
            for feature in FEATURES:
                samples = samples.with_columns(
                    preprocessors[feature]
                    .static_transform(samples[feature])
                    .alias(feature)
                )
            samples = samples.select(
                "client_id",
                "timestamp",
                "prediction_timestamp",
                "sample_id",
                "split",
                pl.struct(FEATURES).alias("context"),
                pl.struct(pl.col(LABELS).cast(pl.Int32)).alias("target"),
            )
            for split in ["train", "valid", "test"]:
                frame = samples.filter(pl.col("split") == split)
                counts[split]["rows"] += frame.height
                for name, value in (
                    frame["target"].struct.unnest().sum().row(0, named=True).items()
                ):
                    counts[split][name] += value
            for fold, frame in [
                ("train", samples.filter(pl.col("split") == "train")),
                ("test", samples.filter(pl.col("split") != "train")),
            ]:
                if train_test:
                    frame = frame.drop("split")
                table = frame.to_arrow()
                if fold not in writers:
                    path = args.output / f"cross/dataset/{fold}/samples.pq"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writers[fold] = pq.ParquetWriter(
                        path, table.schema, compression="zstd"
                    )
                writers[fold].write_table(table)
            logger.info(
                "Encoded events and examples: shard %s/%s", shard + 1, args.shards
            )
    for writer in writers.values():
        writer.close()
    if counts != source_counts:
        raise ValueError(f"Target rows or labels changed: {counts} != {source_counts}")
    for fold in ["train", "test"]:
        source = args.output / f"cross/dataset/{fold}/samples.pq"
        destination = args.output / f"single/dataset/{fold}/samples.pq"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.hardlink_to(source)
    for history in ["single", "cross"]:
        index = pl.read_parquet(args.output / f"{history}/dataset/events/index.pq")
        if (
            index["client_id"].n_unique() != clients.height
            or index.height != clients.height
        ):
            raise ValueError(f"Missing or repeated clients in {history} EventStore")
    report = {
        "protocol": args.protocol,
        "split_audit": split_audit,
        "status": "prepared_full_target_population",
        "source_revision": source_manifest["revision"],
        "target_users": clients.height,
        "cohort_sampling": False,
        "cold_histories_retained": True,
        "shared_train_only_vocabularies": True,
        "identical_samples_and_labels": True,
        "source_event_rows_for_target_users": domain_counts,
        "event_rows_after_deduplication": event_rows,
        "folds": counts,
        "history_length": 512,
        "history_gap_hours": 12,
        "vocabularies": {
            name: preprocessor.cardinality
            for name, preprocessor in preprocessors.items()
        },
        "sample_sha256": {
            fold: checksum(args.output / f"cross/dataset/{fold}/samples.pq")
            for fold in ["train", "test"]
        },
    }
    (args.output / "full-report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Full preparation complete: %s", report)


if __name__ == "__main__":
    main()
