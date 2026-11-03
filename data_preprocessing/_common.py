"""Shared contracts for the Taobao UniRank → Perseus action-format conversion.

One raw impression row of `<raw_root>/{split}/data/part-<id>.parquet` maps to
exactly one basis sample and exactly one hub event. Block filenames are used
verbatim: the real UniRank root calls the first block `part-00000.parquet`, not
`part-0.parquet`. The source key is `(split, block_id, row_ordinal)`; `(client_id,
timestamp)` is not unique in Taobao (impressions of one user may share a second),
so ordinals are mandatory.

Timestamps reuse the frozen-artifact formula `ts_sec * 1e9 + position * 1000`
(ns), where `position` is the impression index in the user's full history
(`seq_len`). Same-second impressions keep UniRank's `(time_stamp, _row_id)` order
because `position` increases along that order.

Event type names are the `action_vocab` names of `meta_data.json` verbatim: one
source action combination = one hub event type. The event name is taken from
`full_action_seq[position]`; the label combination is checked independently.
"""

from __future__ import annotations
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

DATA_DIR = "data"
USER_INFO_DIR = "user_info"
ITEM_INFO_DIR = "item_info"
SPLITS = ("train", "valid", "test")

LABEL_COLUMNS = ("is_click", "cart", "fav", "buy")
USER_STATIC_COLUMNS = (
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
)
CONTEXT_COLUMNS = ("pid", "is_weekend", "hour")
ITEM_COLUMNS = ("item_id", "cate_id", "campaign_id", "customer_id", "brand", "price_bucket")

CLIENT_COLUMN = "user_id"
POSITION_COLUMN = "seq_len"
TIMESTAMP_UNIT_SCALE = 1_000_000_000
POSITION_OFFSET_SCALE = 1_000
MAX_POSITION = TIMESTAMP_UNIT_SCALE // POSITION_OFFSET_SCALE

SOURCE_KEY_COLUMNS = ("source_split", "source_part_id", "source_row_ordinal")
BASIS_EXTRA_COLUMNS = (*SOURCE_KEY_COLUMNS, "source_position", "source_ts_sec")

EXPOSURE_ACTION = "exposure"
PART_FILE_RE = re.compile(r"^part-(\d+)\.parquet$")
BLOCK_DIRS = (DATA_DIR, USER_INFO_DIR, ITEM_INFO_DIR)


class DataContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlockPaths:
    split: str
    part_id: int
    data: Path
    user_info: Path
    item_info: Path


@dataclass(frozen=True)
class ActionVocab:
    name_to_code: dict[str, int]
    label_columns: tuple[str, ...]

    @classmethod
    def from_meta_data(cls, meta_data: dict, /) -> ActionVocab:
        raw_vocab = meta_data.get("action_vocab")
        if not isinstance(raw_vocab, dict) or not raw_vocab:
            raise DataContractError("meta_data.json has no usable action_vocab mapping")
        labels = meta_data.get("label")
        if not labels:
            raise DataContractError("meta_data.json has no `label` list")
        vocab = {str(name): int(code) for name, code in raw_vocab.items()}
        codes = sorted(vocab.values())
        if len(set(codes)) != len(codes):
            raise DataContractError(f"action_vocab codes are not unique: {codes}")
        if codes != list(range(1, len(vocab) + 1)):
            raise DataContractError(f"action_vocab codes must be 1..N contiguous, got {codes}")
        label_columns = tuple(str(name) for name in labels)
        if len(set(label_columns)) != len(label_columns):
            raise DataContractError(f"meta_data.json has duplicate label columns: {label_columns}")
        return cls(name_to_code=vocab, label_columns=label_columns)

    @property
    def code_to_name(self) -> dict[int, str]:
        return {code: name for name, code in self.name_to_code.items()}

    def name_of(self, row_labels: dict[str, int], /) -> str:
        active = {alias_of(label) for label, value in row_labels.items() if value != 0}

        def aliases(name: str) -> set[str]:
            return set() if name == EXPOSURE_ACTION else {alias_of(part) for part in name.split("|")}

        matches = [name for name in self.name_to_code if aliases(name) == active]
        if len(matches) != 1:
            raise DataContractError(f"labels {row_labels} match action_vocab names {matches}, expected exactly one")
        return matches[0]


def alias_of(label: str, /) -> str:
    return label.removeprefix("is_")


def event_names(vocab: ActionVocab, /) -> list[str]:
    return sorted(vocab.name_to_code, key=lambda name: vocab.name_to_code[name])


def timestamp_ns(ts_sec: int, position: int, /) -> int:
    if ts_sec < 0:
        raise DataContractError(f"negative source second {ts_sec}")
    if position < 0:
        raise DataContractError(f"negative source position {position}")
    if position >= MAX_POSITION:
        raise DataContractError(
            f"position {position} exceeds the µs-offset capacity {MAX_POSITION}; timestamp ordering would be undefined",
        )
    value = ts_sec * TIMESTAMP_UNIT_SCALE + position * POSITION_OFFSET_SCALE
    if value >= 2**63:
        raise DataContractError(f"timestamp {value} overflows ns int64")
    return value


def multiset_digest(keys: list[tuple], /) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(repr(key).encode())
    return digest.hexdigest()


def require_columns(schema: pl.Schema, columns: tuple[str, ...] | list[str], path: Path, /) -> None:
    missing = [name for name in columns if name not in schema]
    if missing:
        raise DataContractError(f"{path} is missing required columns {missing}; found {list(schema.names())}")


def discover_blocks(raw_root: Path, split: str, /) -> dict[int, BlockPaths]:
    """Return block-id -> actual paths, coordinating data/user_info/item_info.

    The numeric block id is parsed only for coordination; all reads use the real
    filename. Duplicate ids, missing counterparts and unknown files fail loudly.
    """
    paths_by_kind: dict[str, dict[int, Path]] = {}
    source_names: dict[str, dict[int, str]] = {}
    for kind in BLOCK_DIRS:
        directory = raw_root / split / kind
        if not directory.is_dir():
            raise DataContractError(f"missing block directory {directory}")
        found: dict[int, Path] = {}
        names: dict[int, str] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            match = PART_FILE_RE.match(path.name)
            if match is None:
                raise DataContractError(f"unexpected file in {directory}: {path.name}")
            block_id = int(match.group(1))
            if block_id in found:
                raise DataContractError(
                    f"duplicate block id {block_id} in {directory}: {names[block_id]} and {path.name}",
                )
            found[block_id] = path
            names[block_id] = path.name
        paths_by_kind[kind] = found
        source_names[kind] = names

    data_ids = set(paths_by_kind[DATA_DIR])
    for kind in (USER_INFO_DIR, ITEM_INFO_DIR):
        other_ids = set(paths_by_kind[kind])
        if data_ids != other_ids:
            missing = sorted(data_ids - other_ids)
            extra = sorted(other_ids - data_ids)
            raise DataContractError(
                f"{raw_root}/{split}: data/{kind} block ids disagree; missing {missing}, extra {extra}",
            )
    return {
        block_id: BlockPaths(
            split=split,
            part_id=block_id,
            data=paths_by_kind[DATA_DIR][block_id],
            user_info=paths_by_kind[USER_INFO_DIR][block_id],
            item_info=paths_by_kind[ITEM_INFO_DIR][block_id],
        )
        for block_id in sorted(data_ids)
    }


def part_ids(raw_root: Path, split: str, /) -> list[int]:
    return sorted(discover_blocks(raw_root, split))


def read_meta_data(raw_root: Path, /) -> dict:
    meta_path = raw_root / "meta_data.json"
    if not meta_path.is_file():
        raise DataContractError(f"missing {meta_path}")
    return json.loads(meta_path.read_text())


def read_block_manifest(path: Path, /) -> dict[str, dict[int, dict]] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    result: dict[str, dict[int, dict]] = {}
    for split in SPLITS:
        entry = raw.get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("blocks"), list):
            raise DataContractError(f"{path}: split {split} has no `blocks` list")
        blocks: dict[int, dict] = {}
        for block in entry["blocks"]:
            block_id = int(block["block_id"])
            if block_id in blocks:
                raise DataContractError(f"{path}: duplicate {split} block_id {block_id}")
            blocks[block_id] = block
        expected_count = entry.get("num_blocks_written", entry.get("num_blocks_configured"))
        if expected_count is not None and int(expected_count) != len(blocks):
            raise DataContractError(
                f"{path}: {split} declares {expected_count} written blocks but lists {len(blocks)}",
            )
        result[split] = blocks
    return result


def block_counts(block: BlockPaths, /) -> dict[str, int]:
    data = pl.scan_parquet(block.data).select(pl.len()).collect().item()
    users = pl.scan_parquet(block.user_info).select(pl.len()).collect().item()
    items = pl.scan_parquet(block.item_info).select(pl.len()).collect().item()
    return {"rows": int(data), "users": int(users), "items": int(items)}


def load_block_triple(raw_root: Path, split: str, part_id: int, /) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    blocks = discover_blocks(raw_root, split)
    if part_id not in blocks:
        raise DataContractError(f"{raw_root}/{split}: block id {part_id} not found")
    paths = blocks[part_id]
    data = pl.read_parquet(paths.data)
    users = pl.read_parquet(paths.user_info)
    items = pl.read_parquet(paths.item_info)
    require_columns(
        data.schema,
        (CLIENT_COLUMN, POSITION_COLUMN, "item_index", *LABEL_COLUMNS, *USER_STATIC_COLUMNS, *CONTEXT_COLUMNS),
        paths.data,
    )
    require_columns(users.schema, ("user_index", "full_item_seq", "full_action_seq", "full_timestamp_seq"), paths.user_info)
    require_columns(items.schema, ("item_index", "item_id", *ITEM_COLUMNS[1:]), paths.item_info)
    if items.schema["item_index"] == pl.Int64() and (items["item_index"] == 0).sum() != 1:
        raise DataContractError(f"{paths.item_info} must contain the single 0 padding row")
    return data, users, items


def fail(messages: list[str], /) -> None:
    if messages:
        for message in messages:
            print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)
