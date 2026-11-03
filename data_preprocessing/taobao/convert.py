"""Convert a UniRank blocked Taobao root into the action-split hub and full basis.

Input (frozen UniRank layout):
    <raw_root>/meta_data.json
    <raw_root>/{train,valid,test}/{data,user_info,item_info}/part-<id>.parquet

Output (new action-format root, never over the frozen artifacts):
    <out_root>/hub/<action-combination>/<date>/<NNNN>.pq
    <out_root>/basis/train/samples.pq   # every source train row
    <out_root>/basis/test/samples.pq    # every source valid + test row, `split` preserved
    <out_root>/manifest.json, <out_root>/action-event-mapping.json

One impression row = one basis sample = one hub event. The hub event type is the
full action combination of the impression (including `exposure`) taken from
`full_action_seq[seq_len]`; the four label columns are checked against that code
independently. Labels never enter hub rows and the current outcome never enters
context. No train/valid/test truncation happens here.

The output root must be absent or empty. Hub/basis fragments are written to a
staging directory per source block and merged once per final output file, so a
later block cannot overwrite an earlier fragment of the same output partition.
"""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import polars as pl


from data_preprocessing._common import (
    BASIS_EXTRA_COLUMNS,
    CLIENT_COLUMN,
    CONTEXT_COLUMNS,
    EXPOSURE_ACTION,
    ITEM_COLUMNS,
    LABEL_COLUMNS,
    MAX_POSITION,
    POSITION_COLUMN,
    POSITION_OFFSET_SCALE,
    SPLITS,
    TIMESTAMP_UNIT_SCALE,
    USER_STATIC_COLUMNS,
    ActionVocab,
    BlockPaths,
    DataContractError,
    block_counts,
    discover_blocks,
    event_names,
    load_block_triple,
    read_block_manifest,
    read_meta_data,
)

EVENT_ATTRIBUTES = ("item", *ITEM_COLUMNS[1:])
EVENT_ROOT = "hub"
BASIS_ROOT = "basis"
STAGING_ROOT = ".staging"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="UniRank blocked dataset root (read-only input)")
    parser.add_argument("--out-root", type=Path, required=True, help="new action-format output root")
    parser.add_argument("--dataset-version", default="taobao-action-v1", help="dataset version recorded in manifests")
    parser.add_argument("--splits", nargs="*", default=None, help="subset of splits to convert (default: all)")
    parser.add_argument(
        "--block-manifest",
        type=Path,
        default=None,
        help="completeness manifest (default: <raw-root>/block_manifest.json when present)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow whole missing blocks for an explicitly local subset; corrupted present blocks still fail",
    )
    parser.add_argument(
        "--emit-config",
        type=Path,
        default=None,
        help="write the action-event prepare config generated from the actual action_vocab (no conversion run)",
    )
    return parser.parse_args()


def emit_config(vocab: ActionVocab, path: Path, /) -> None:
    import yaml

    attributes = dict.fromkeys(EVENT_ATTRIBUTES)
    config = {
        "task": {
            "type": "classification",
            "metrics": {"roc_auc": {"type": "roc_auc"}, "logloss": {"type": "logloss"}},
            "preprocessor": {},
            "layer": {},
            "criterion": {},
            "head": {},
        },
        "events": {name: {"attributes": dict(attributes)} for name in event_names(vocab)},
        "encoders": {
            name: {"type": "id", "preprocessor": {"min_count": 1}}
            for name in (
                "item",
                *ITEM_COLUMNS[1:],
                *USER_STATIC_COLUMNS,
                *CONTEXT_COLUMNS,
                "user_id",
                "candidate_action",
            )
        },
        "features": {
            "item": {"located_in": {"event": True, "context": True}, "encoder": "item"},
            **{
                name: {"located_in": {"event": True, "context": True}, "encoder": name}
                for name in ITEM_COLUMNS[1:]
            },
            **{
                name: {"located_in": {"context": True}, "encoder": name}
                for name in (*USER_STATIC_COLUMNS, *CONTEXT_COLUMNS, "user_id", "candidate_action")
            },
        },
        "max_events_per_sequence": 100,
        "embedding_dim": 16,
        "backbone": {
            "dim": 256,
            "event_aggregator": {"type": "sum", "params": {}},
            "context_aggregator": {"type": "identity", "params": {}},
            "history_aggregator": {"sequence_encoder": {"type": "modern_bert", "params": {}}},
        },
        "training": {
            "num_epochs": 1,
            "seed": 42,
            "event_store_backend": "cached",
        },
        "prepare": {"allow_empty_sequences": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def sha256_file(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(Path(__file__).resolve().parent)], capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "data_scripts_dirty": None}
    return {"commit": commit, "data_scripts_dirty": bool(dirty)}


def alias_of(label: str, /) -> str:
    return label.removeprefix("is_")


def canonical_mask_to_name(vocab: ActionVocab, /) -> dict[int, str]:
    mask_to_name = {}
    for name in vocab.name_to_code:
        aliases = set() if name == EXPOSURE_ACTION else {alias_of(part) for part in name.split("|")}
        mask = 0
        for index, label in enumerate(vocab.label_columns):
            if alias_of(label) in aliases:
                mask |= 1 << index
        if mask in mask_to_name:
            raise DataContractError(f"action_vocab names {mask_to_name[mask]!r} and {name!r} share mask {mask}")
        mask_to_name[mask] = name
    if set(mask_to_name) != set(range(1 << len(vocab.label_columns))):
        missing = sorted(set(range(1 << len(vocab.label_columns))) - set(mask_to_name))
        raise DataContractError(f"action_vocab is incomplete: label masks {missing} have no name")
    return mask_to_name


def check_binary_labels(data: pl.DataFrame, path: Path, vocab: ActionVocab, /) -> None:
    for label in vocab.label_columns:
        column = data[label]
        if column.null_count():
            raise DataContractError(f"{path}: column {label} has {column.null_count()} nulls")
        bad = ((column != 0.0) & (column != 1.0)).sum()
        if bad:
            raise DataContractError(f"{path}: column {label} has {bad} non-binary values; refusing silent bool cast")


def label_mask(vocab: ActionVocab, /) -> pl.Expr:
    mask = pl.lit(0, pl.Int32())
    for index, label in enumerate(vocab.label_columns):
        mask = mask + (pl.col(label) != 0.0).cast(pl.Int32()) * (1 << index)
    return mask


def _users_long(users: pl.DataFrame, path: Path, /) -> pl.DataFrame:
    lengths = users.select(
        pl.col("full_item_seq").list.len().alias("items"),
        pl.col("full_action_seq").list.len().alias("actions"),
        pl.col("full_timestamp_seq").list.len().alias("timestamps"),
    )
    bad_lengths = lengths.filter((pl.col("items") != pl.col("actions")) | (pl.col("items") != pl.col("timestamps")))
    if bad_lengths.height:
        raise DataContractError(f"{path}: full_item_seq/full_action_seq/full_timestamp_seq lengths disagree")
    long = (
        users.with_columns(position=pl.int_ranges(pl.col("full_item_seq").list.len()))
        .explode("position", "full_item_seq", "full_action_seq", "full_timestamp_seq")
        .rename(
            {
                "full_item_seq": "seq_item_index",
                "full_action_seq": "seq_action_code",
                "full_timestamp_seq": "ts_sec",
            },
        )
    )
    if long.select(pl.struct("user_index", "position").n_unique()).item() != long.height:
        raise DataContractError(f"{path}: (user_index, position) is not unique in user_info")
    return long.select("user_index", "position", "seq_item_index", "seq_action_code", "ts_sec")


def convert_block(
    raw_root: Path,
    block: BlockPaths,
    vocab: ActionVocab,
    mask_to_name: dict[int, str],
    item_id_to_attrs: dict[int, tuple],
    /,
) -> tuple[pl.DataFrame, list[tuple[str, pl.DataFrame]], dict]:
    split = block.split
    part_id = block.part_id
    data, users, items = load_block_triple(raw_root, split, part_id)
    if data.height == 0:
        return pl.DataFrame(), [], {"rows": 0}

    check_binary_labels(data, block.data, vocab)
    if data["item_index"].eq(0).any():
        raise DataContractError(f"{block.data}: item_index 0 is the padding row and cannot be an impression")
    if data[CLIENT_COLUMN].null_count():
        raise DataContractError(f"{block.data}: {CLIENT_COLUMN} has nulls; refusing to invent a client identity")

    users_long = _users_long(users, block.user_info)
    joined = data.with_row_index("source_row_ordinal").join(
        users_long,
        left_on=["user_index", POSITION_COLUMN],
        right_on=["user_index", "position"],
        how="left",
        coalesce=True,
    )
    if joined.height != data.height:
        raise DataContractError(
            f"{block.data}: join with user_info multiplied rows ({data.height} -> {joined.height}); "
            "one impression must stay one sample",
        )
    if joined.select(pl.col("ts_sec").is_null().any()).item():
        raise DataContractError(f"{block.data}: some (user_index, {POSITION_COLUMN}) pairs are absent from user_info")

    mismatch = joined.filter(pl.col("seq_item_index") != pl.col("item_index"))
    if not mismatch.is_empty():
        raise DataContractError(
            f"{block.data}: full_item_seq[{POSITION_COLUMN}] != item_index for {mismatch.height} rows "
            f"(first ordinals {mismatch['source_row_ordinal'].head(5).to_list()})",
        )

    code_to_name = vocab.code_to_name
    if joined.select(pl.col("seq_action_code").is_null().any()).item():
        raise DataContractError(f"{block.data}: full_action_seq has null action codes")
    unknown_codes = sorted(set(joined["seq_action_code"].to_list()) - set(code_to_name))
    if unknown_codes:
        raise DataContractError(f"{block.data}: full_action_seq contains codes outside action_vocab: {unknown_codes}")
    joined = joined.with_columns(
        source_action_name=pl.col("seq_action_code").replace_strict(code_to_name, return_dtype=pl.String),
        label_action_name=label_mask(vocab).replace_strict(mask_to_name, return_dtype=pl.String),
    )
    action_mismatch = joined.filter(pl.col("source_action_name") != pl.col("label_action_name"))
    if not action_mismatch.is_empty():
        sample = action_mismatch.select(
            "source_row_ordinal", "seq_action_code", "source_action_name", "label_action_name",
        ).head(5).to_dicts()
        raise DataContractError(
            f"{block.data}: full_action_seq[seq_len] disagrees with the label combination for "
            f"{action_mismatch.height} rows: {sample}",
        )

    item_side = items.select(
        "item_index",
        pl.col("item_id").cast(pl.Int64),
        *[pl.col(name) for name in ITEM_COLUMNS[1:]],
    ).unique(subset="item_index")
    if item_side.height != items.height:
        raise DataContractError(f"{raw_root}/{split}: item_info part {part_id} has duplicate item_index")
    joined = joined.join(item_side, on="item_index", how="left", coalesce=True)
    if joined.height != data.height:
        raise DataContractError(f"{block.data}: join with item_info multiplied rows")
    if joined.select(pl.col("item_id").is_null().any()).item():
        raise DataContractError(f"{block.data}: some item_index values are absent from item_info part {part_id}")

    for row in item_side.iter_rows(named=True):
        if row["item_index"] == 0:
            continue
        attrs = tuple(row[name] for name in ITEM_COLUMNS[1:])
        known = item_id_to_attrs.setdefault(row["item_id"], attrs)
        if known != attrs:
            raise DataContractError(
                f"{raw_root}/{split}: item_id {row['item_id']} has conflicting attributes across blocks: {known} vs {attrs}",
            )

    max_position = joined[POSITION_COLUMN].max()
    if max_position >= MAX_POSITION:
        raise DataContractError(f"{block.data}: position {max_position} exceeds the µs-offset capacity {MAX_POSITION}")
    max_ts_sec = joined["ts_sec"].max()
    if max_ts_sec >= (2**63) // TIMESTAMP_UNIT_SCALE:
        raise DataContractError(f"{block.data}: source second {max_ts_sec} overflows the ns timestamp range")
    if joined["ts_sec"].min() < 0:
        raise DataContractError(f"{block.data}: negative source timestamp seconds")

    timestamp = (
        pl.col("ts_sec").cast(pl.Int64) * TIMESTAMP_UNIT_SCALE
        + pl.col(POSITION_COLUMN).cast(pl.Int64) * POSITION_OFFSET_SCALE
    ).cast(pl.Datetime("ns"))

    samples = joined.select(
        pl.col(CLIENT_COLUMN).cast(pl.Int64).cast(pl.String).alias("client_id"),
        timestamp.alias("timestamp"),
        pl.struct(*(pl.col(label).cast(pl.Int8).alias(label) for label in vocab.label_columns)).alias("target"),
        pl.struct(
            pl.col(CLIENT_COLUMN).cast(pl.Int64).cast(pl.String).alias("user_id"),
            pl.lit(0, dtype=pl.Int64).alias("candidate_action"),
            *(pl.col(name).cast(pl.Int64).cast(pl.String) for name in USER_STATIC_COLUMNS),
            *(pl.col(name).cast(pl.Int64).cast(pl.String) for name in CONTEXT_COLUMNS),
            pl.col("item_id").cast(pl.String).alias("item"),
            *(pl.col(name).cast(pl.Int64).cast(pl.String) for name in ITEM_COLUMNS[1:]),
        ).alias("context"),
        pl.lit(split).alias("split"),
        pl.lit(split).alias("source_split"),
        pl.lit(part_id, pl.UInt32).alias("source_part_id"),
        pl.col("source_row_ordinal").cast(pl.UInt32),
        pl.col(POSITION_COLUMN).cast(pl.UInt32).alias("source_position"),
        pl.col("ts_sec").cast(pl.Int64).alias("source_ts_sec"),
    )

    events = joined.select(
        pl.col(CLIENT_COLUMN).cast(pl.Int64).cast(pl.String).alias("client_id"),
        timestamp.alias("timestamp"),
        pl.col("item_id").cast(pl.String).alias("item"),
        *(pl.col(name).cast(pl.Int64).cast(pl.String) for name in EVENT_ATTRIBUTES[1:]),
        pl.col("source_action_name").alias("_action"),
    )
    event_frames = [
        (name[0], frame.drop("_action"))
        for name, frame in events.partition_by("_action", as_dict=True).items()
    ]
    timestamp_ns = samples["timestamp"].dt.epoch("ns")
    stats = {
        "rows": data.height,
        "actions": dict(sorted(Counter(events["_action"].to_list()).items())),
        "timestamp_ns_min": int(timestamp_ns.min()),
        "timestamp_ns_max": int(timestamp_ns.max()),
        "source_ts_sec_min": int(joined["ts_sec"].min()),
        "source_ts_sec_max": int(joined["ts_sec"].max()),
    }
    return samples, event_frames, stats


def compare_with_block_manifest(
    raw_root: Path,
    manifest_path: Path | None,
    split_blocks: dict[str, dict[int, BlockPaths]],
    selected_splits: tuple[str, ...],
    allow_partial: bool,
    /,
) -> dict:
    """Require the manifest; --allow-partial may omit whole blocks, never corrupt them."""
    path = manifest_path or (raw_root / "block_manifest.json")
    if not path.is_file():
        raise DataContractError(
            f"block manifest is required for conversion: {path} is missing; "
            "fetch the complete block_manifest.json (no manifest-less full-run claims)",
        )
    expected = read_block_manifest(path)
    if expected is None:
        raise DataContractError(f"could not parse {path}")
    report_splits: dict[str, dict] = {}
    missing_messages: list[str] = []
    corruption_messages: list[str] = []
    any_missing = False
    for split in selected_splits:
        actual = split_blocks[split]
        expected_blocks = expected[split]
        missing = sorted(set(expected_blocks) - set(actual))
        extra = sorted(set(actual) - set(expected_blocks))
        row_mismatches = []
        users_mismatches = []
        items_mismatches = []
        for block_id in sorted(set(actual) & set(expected_blocks)):
            counts = block_counts(actual[block_id])
            declared = expected_blocks[block_id]
            if int(declared.get("rows", -1)) != counts["rows"]:
                row_mismatches.append({"block_id": block_id, "manifest": declared.get("rows"), "actual": counts["rows"]})
            if "users" in declared and int(declared["users"]) != counts["users"]:
                users_mismatches.append({"block_id": block_id, "manifest": declared["users"], "actual": counts["users"]})
            if "items" in declared and int(declared["items"]) != counts["items"] - 1:
                items_mismatches.append({"block_id": block_id, "manifest": declared["items"], "actual": counts["items"] - 1})
        split_missing = bool(missing)
        split_corrupt = bool(extra or row_mismatches or users_mismatches or items_mismatches)
        any_missing = any_missing or split_missing
        report_splits[split] = {
            "declared_blocks": len(expected_blocks),
            "present_blocks": len(actual),
            "missing_block_ids": missing,
            "unexpected_block_ids": extra,
            "row_count_mismatches": row_mismatches,
            "user_count_mismatches": users_mismatches,
            "item_count_mismatches": items_mismatches,
            "declared_rows": sum(int(block.get("rows", 0)) for block in expected_blocks.values()),
            "present_rows": sum(block_counts(actual[block_id])["rows"] for block_id in actual),
            "partial": split_missing,
        }
        if missing:
            missing_messages.append(f"{split}: missing whole manifest block ids {missing}")
        if extra:
            corruption_messages.append(f"{split}: unexpected block ids {extra}")
        if row_mismatches:
            corruption_messages.append(f"{split}: row counts differ from manifest {row_mismatches[:5]}")
        if users_mismatches:
            corruption_messages.append(f"{split}: user counts differ from manifest {users_mismatches[:5]}")
        if items_mismatches:
            corruption_messages.append(f"{split}: item counts (excluding padding) differ from manifest {items_mismatches[:5]}")
    if corruption_messages:
        raise DataContractError(
            "present input blocks are corrupted relative to the manifest; --allow-partial only permits "
            "omitting whole blocks and cannot bypass these failures:\n"
            + "\n".join(f"- {message}" for message in corruption_messages),
        )
    if missing_messages and not allow_partial:
        raise DataContractError(
            "input is not the complete manifest dataset; pass --allow-partial only for an explicitly local subset:\n"
            + "\n".join(f"- {message}" for message in missing_messages),
        )
    any_partial = any_missing or len(selected_splits) != len(SPLITS)
    return {
        "path": str(path),
        "status": "partial" if any_partial else "complete",
        "partial": any_partial,
        "missing_blocks": missing_messages,
        "corruption": [],
        "splits": report_splits,
    }


def ensure_empty_output(out_root: Path, /) -> None:
    if out_root.exists():
        if not out_root.is_dir():
            raise DataContractError(f"output root {out_root} exists and is not a directory")
        entries = list(out_root.iterdir())
        if entries:
            raise DataContractError(
                f"refusing to write into non-empty output root {out_root}; choose a new prefix (no resume/overwrite protocol)",
            )
    out_root.mkdir(parents=True, exist_ok=True)


def stage_events(staging: Path, action_name: str, split: str, part_id: int, frame: pl.DataFrame, /) -> None:
    from perseus import core

    distributed = (
        core.event_hub.distribute_to_partitions(frame.lazy())
        .with_columns(date=pl.col("timestamp").dt.date())
        .collect()
    )
    for (date,), date_frame in distributed.partition_by("date", as_dict=True).items():
        date_dir = staging / "events" / action_name / str(date)
        date_dir.mkdir(parents=True, exist_ok=True)
        for (partition_id,), partition_frame in date_frame.partition_by("partition_id", as_dict=True).items():
            path = date_dir / f"part-{int(partition_id):04d}-{split}-{part_id:05d}.pq"
            if path.exists():
                raise DataContractError(f"staging fragment collision at {path}")
            partition_frame.drop("partition_id").write_parquet(path)


def merge_fragments(files: list[Path], destination: Path, /) -> None:
    if not files:
        raise DataContractError(f"no fragments for {destination}")
    if destination.exists():
        raise DataContractError(f"refusing to overwrite existing output file {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(files).sink_parquet(destination)


def finalize_hub(staging: Path, hub_root: Path, vocab: ActionVocab, /) -> dict:
    event_schema = {
        "client_id": pl.String,
        "timestamp": pl.Datetime("ns"),
        **dict.fromkeys(EVENT_ATTRIBUTES, pl.String),
        "date": pl.Date,
    }
    staged = staging / "events"
    counts: dict[str, int] = {}
    for action_name in event_names(vocab):
        type_dir = hub_root / action_name
        type_dir.mkdir(parents=True, exist_ok=True)
        action_staging = staged / action_name
        files: list[Path] = []
        if action_staging.is_dir():
            for date_dir in sorted(path for path in action_staging.iterdir() if path.is_dir()):
                by_partition: dict[int, list[Path]] = {}
                for fragment in sorted(date_dir.glob("part-*.pq")):
                    partition_id = int(fragment.name.split("-", maxsplit=2)[1])
                    by_partition.setdefault(partition_id, []).append(fragment)
                for partition_id, fragments in sorted(by_partition.items()):
                    final = type_dir / date_dir.name / f"{partition_id:04d}.pq"
                    merge_fragments(fragments, final)
                    files.append(final)
        if not files:
            empty_dir = type_dir / "1970-01-01"
            empty_dir.mkdir(exist_ok=True)
            pl.DataFrame(schema=event_schema).write_parquet(empty_dir / "0001.pq")
        else:
            counts[action_name] = pl.scan_parquet(files).select(pl.len()).collect().item()
    return counts


def finalize_basis(staging: Path, basis_root: Path, /) -> dict:
    fold_sources = {"train": ("train",), "test": ("valid", "test")}
    report: dict[str, dict] = {}
    for fold, source_splits in fold_sources.items():
        files = sorted(
            staging / "basis" / split / fragment.name
            for split in source_splits
            for fragment in (staging / "basis" / split).glob("part-*.pq")
        )
        destination = basis_root / fold / "samples.pq"
        if files:
            merge_fragments(files, destination)
            split_counts = dict(
                pl.scan_parquet(files)
                .group_by("split")
                .len()
                .collect()
                .sort("split")
                .iter_rows(),
            )
            rows = int(sum(split_counts.values()))
            missing_defaults = [
                field
                for field in ("client_id", "timestamp", "target", "context", "split", *BASIS_EXTRA_COLUMNS)
                if field not in pl.scan_parquet(files).collect_schema()
            ]
            if missing_defaults:
                raise DataContractError(f"basis {fold} lacks required columns {missing_defaults}")
        else:
            split_counts = {}
            rows = 0
            raise DataContractError(f"no basis fragments for fold {fold}")
        report[fold] = {"rows": rows, "split_counts": split_counts, "sha256": sha256_file(destination)}
    return report


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root
    out_root: Path = args.out_root
    splits = tuple(args.splits) if args.splits else SPLITS
    unknown_splits = [split for split in splits if split not in SPLITS]
    if unknown_splits:
        raise SystemExit(f"unknown splits {unknown_splits}")

    meta_data = read_meta_data(raw_root)
    vocab = ActionVocab.from_meta_data(meta_data)
    if tuple(vocab.label_columns) != LABEL_COLUMNS:
        raise DataContractError(
            f"unexpected Taobao label columns {vocab.label_columns}; this converter is Taobao-specific",
        )

    if args.emit_config:
        emit_config(vocab, args.emit_config)
        return

    ensure_empty_output(out_root)
    split_blocks = {split: discover_blocks(raw_root, split) for split in splits}
    completeness = compare_with_block_manifest(
        raw_root,
        args.block_manifest,
        split_blocks,
        splits,
        args.allow_partial,
    )
    mask_to_name = canonical_mask_to_name(vocab)

    staging = out_root / STAGING_ROOT
    hub_root = out_root / EVENT_ROOT
    basis_root = out_root / BASIS_ROOT
    for directory in (staging, hub_root, basis_root):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_version": args.dataset_version,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "output_root": str(out_root),
        "meta_data_sha256": sha256_file(raw_root / "meta_data.json"),
        "code": code_revision(),
        "requested_splits": list(splits),
        "partial": bool(completeness.get("partial")),
        "completeness": completeness,
        "timestamp_formula": "ts_sec * 1e9 + source_position * 1000 (ns); position < 1e6 and ns range enforced",
        "sample_cardinality": "one raw impression row -> one basis sample -> one hub event; join multiplication is an error",
        "label_columns": list(vocab.label_columns),
        "action_source": "hub event name = action_vocab[full_action_seq[seq_len]], independently checked against labels",
        "basis_folds": {"train": ["train"], "test": ["valid", "test"]},
        "splits": {},
    }

    item_id_to_attrs: dict[int, tuple] = {}
    for split in splits:
        split_stats = {"parts": {}, "rows": 0}
        for part_id, block in split_blocks[split].items():
            samples, event_frames, stats = convert_block(
                raw_root, block, vocab, mask_to_name, item_id_to_attrs,
            )
            if samples.height:
                basis_dir = staging / "basis" / split
                basis_dir.mkdir(parents=True, exist_ok=True)
                fragment = basis_dir / f"part-{part_id:05d}.pq"
                if fragment.exists():
                    raise DataContractError(f"staging fragment collision at {fragment}")
                samples.write_parquet(fragment)
            for action_name, frame in event_frames:
                stage_events(staging, action_name, split, part_id, frame)
            split_stats["parts"][str(part_id)] = stats
            split_stats["rows"] += stats["rows"]
        split_stats["timestamp_ns_min"] = min(
            (part["timestamp_ns_min"] for part in split_stats["parts"].values() if "timestamp_ns_min" in part),
            default=None,
        )
        split_stats["timestamp_ns_max"] = max(
            (part["timestamp_ns_max"] for part in split_stats["parts"].values() if "timestamp_ns_max" in part),
            default=None,
        )
        manifest["splits"][split] = split_stats

    manifest["hub_event_rows"] = finalize_hub(staging, hub_root, vocab)
    basis_report = finalize_basis(staging, basis_root)
    manifest["basis"] = basis_report
    manifest["basis_rows_total"] = sum(entry["rows"] for entry in basis_report.values())
    if manifest["basis_rows_total"] != sum(entry["rows"] for entry in manifest["splits"].values()):
        raise DataContractError("basis row total differs from converted source row total")

    mapping = {
        "event_naming": "hub event type == action_vocab name from meta_data.json, identity mapping",
        "source": "full_action_seq[seq_len]",
        "aliasing": {"rule": "strip 'is_' prefix maps a label column to its action token"},
        "actions": [
            {
                "event_name": name,
                "action_code": vocab.name_to_code[name],
                "active_labels": [
                    label
                    for label in vocab.label_columns
                    if alias_of(label) in (set() if name == EXPOSURE_ACTION else {alias_of(part) for part in name.split("|")})
                ],
                "exposure": name == EXPOSURE_ACTION,
            }
            for name in event_names(vocab)
        ],
        "item_id_count": len(item_id_to_attrs),
        "partial": manifest["partial"],
    }
    (out_root / "action-event-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(staging)
    print(
        f"converted {manifest['basis_rows_total']} rows into "
        f"{', '.join(f'{fold}={entry['rows']}' for fold, entry in basis_report.items())}; "
        f"partial={manifest['partial']}",
    )


if __name__ == "__main__":
    main()
