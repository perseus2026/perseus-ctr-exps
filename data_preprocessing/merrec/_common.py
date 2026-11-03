"""Shared contracts for the MerRec UniRank -> Perseus action-format conversion.

One raw sample of `<raw_root>/{split}/data/part-<id>.parquet` (one user-item
view anchor) maps to exactly one basis sample and exactly one hub event. Block
filenames, the user_info/item_info layouts and the source key contract match
the Taobao chain and stay in unirank_common; this module only pins what MerRec
freezes differently (frozen reference: UniRank
`data/MerRec/preprocess_Merrec_seq_action.py`, identical at both spellings,
sha256 dee9ed50...; `config/dataset_config.yaml` MerRec_Action, sha256 40fb015c...):

- label columns and their order: Like, Cart, Offer, Checkout, Purchase;
- action names ARE the label combinations: parts of a "|"-joined name equal the
  label column names verbatim (Taobao strips an `is_` prefix, MerRec has none);
- `full_timestamp_seq` carries source SECONDS, so the ns timestamp formula is
  ts_sec * 1e9 + position * 1e3 with 0 <= position < 1_000_000;
- meta_data.json pins `feature_groups`; converter and gates require the frozen
  field lists below;
- the MerRec action vocab has 32 names (31 non-empty label combinations plus
  "exposure"); codes are 1..32;
- blocked-root categoricals encode missing/OOV values as 0 because
  `__MISSING__` is excluded from the reference vocabularies; this chain
  passes source values through and does not realign that semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path


from data_preprocessing._common import (  # noqa: F401
    DATA_DIR,
    ITEM_INFO_DIR,
    USER_INFO_DIR,
    SPLITS,
    ActionVocab,
    BlockPaths,
    DataContractError,
    block_counts,
    discover_blocks,
    event_names,
    multiset_digest,
    part_ids,
    read_block_manifest,
    read_meta_data,
    require_columns,
)

import polars as pl

LABEL_COLUMNS = ("Like", "Cart", "Offer", "Checkout", "Purchase")
USER_STATIC_COLUMNS: tuple[str, ...] = ()
CONTEXT_COLUMNS = ("session_id", "day_of_week", "is_weekend", "hour")
ITEM_COLUMNS = (
    "item_id",
    "product_id",
    "price_bucket",
    "c0_id",
    "c1_id",
    "c2_id",
    "brand_id",
    "item_condition_id",
    "size_id",
    "shipper_id",
    "color",
)

CLIENT_COLUMN = "user_id"
POSITION_COLUMN = "seq_len"
TIMESTAMP_UNIT_SCALE = 1_000_000_000
POSITION_OFFSET_SCALE = 1_000
MAX_POSITION = TIMESTAMP_UNIT_SCALE // POSITION_OFFSET_SCALE

SOURCE_KEY_COLUMNS = ("source_split", "source_part_id", "source_row_ordinal")
BASIS_EXTRA_COLUMNS = (*SOURCE_KEY_COLUMNS, "source_position", "source_ts_sec")

EXPOSURE_ACTION = "exposure"
EVENT_ATTRIBUTES = ("item", *ITEM_COLUMNS[1:])
CONTEXT_FIELDS = ("user_id", *CONTEXT_COLUMNS, "item", *ITEM_COLUMNS[1:])
EXPECTED_META_FEATURE_GROUPS = {
    "context_features": list(CONTEXT_COLUMNS),
    "item_static_features": list(ITEM_COLUMNS[1:]),
}


def alias_of(label: str, /) -> str:
    return label


def mask_to_action_name(label_columns: tuple[str, ...], mask: int, /) -> str:
    active = [label for index, label in enumerate(label_columns) if mask & (1 << index)]
    return "|".join(active) if active else EXPOSURE_ACTION


def canonical_action_names(label_columns: tuple[str, ...], /) -> set[str]:
    return {
        mask_to_action_name(label_columns, mask)
        for mask in range(1 << len(label_columns))
    }


def canonical_mask_to_name(vocab: ActionVocab, /) -> dict[int, str]:
    expected_names = canonical_action_names(vocab.label_columns)
    missing = sorted(expected_names - set(vocab.name_to_code))
    if missing:
        raise DataContractError(f"action_vocab misses canonical MerRec names: {missing}")
    extra = sorted(set(vocab.name_to_code) - expected_names)
    if extra:
        raise DataContractError(f"action_vocab carries non-canonical MerRec names: {extra}")
    mask_to_name: dict[int, str] = {}
    for name, code in vocab.name_to_code.items():
        if name == EXPOSURE_ACTION:
            mask = 0
        else:
            parts = name.split("|")
            mask = 0
            for part in parts:
                if part not in vocab.label_columns:
                    raise DataContractError(f"action name {name!r} has non-label part {part!r}")
                if parts.count(part) > 1:
                    raise DataContractError(f"action name {name!r} repeats label {part!r}")
                mask |= 1 << vocab.label_columns.index(part)
            joined = "|".join(
                vocab.label_columns[index]
                for index in range(len(vocab.label_columns))
                if mask & (1 << index)
            )
            if joined != name:
                raise DataContractError(f"action name {name!r} is not in label-column order")
        if mask in mask_to_name:
            raise DataContractError(f"action names {mask_to_name[mask]!r} and {name!r} share label mask {mask}")
        mask_to_name[mask] = name
    if set(mask_to_name) != set(range(1 << len(vocab.label_columns))):
        missing_masks = sorted(set(range(1 << len(vocab.label_columns))) - set(mask_to_name))
        raise DataContractError(f"action_vocab does not cover label masks {missing_masks}")
    return mask_to_name


def load_verified_meta_data(raw_root: Path, /) -> ActionVocab:
    meta_data = read_meta_data(raw_root)
    vocab = ActionVocab.from_meta_data(meta_data)
    if tuple(vocab.label_columns) != LABEL_COLUMNS:
        raise DataContractError(
            f"unexpected MerRec label columns {vocab.label_columns}; "
            "this chain is MerRec-specific and refuses any other order",
        )
    canonical_mask_to_name(vocab)
    groups = meta_data.get("feature_groups")
    if not isinstance(groups, dict):
        raise DataContractError("meta_data.json has no feature_groups block")
    for key, expected in EXPECTED_META_FEATURE_GROUPS.items():
        if list(groups.get(key, [])) != expected:
            raise DataContractError(
                f"meta_data.json feature_groups.{key} = {groups.get(key)} != frozen {expected}",
            )
    return vocab


def timestamp_ns(ts_sec: int, position: int, /) -> int:
    if ts_sec < 0:
        raise DataContractError(f"negative source second {ts_sec}")
    if position < 0:
        raise DataContractError(f"negative source position {position}")
    if position >= MAX_POSITION:
        raise DataContractError(
            f"position {position} exceeds the sub-second offset capacity {MAX_POSITION}; "
            "timestamp ordering would be undefined",
        )
    value = ts_sec * TIMESTAMP_UNIT_SCALE + position * POSITION_OFFSET_SCALE
    if value >= 2**63:
        raise DataContractError(f"timestamp {value} overflows ns int64")
    return value


def load_block_triple(raw_root: Path, split: str, part_id: int, /) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    paths = discover_blocks(raw_root, split)
    if part_id not in paths:
        raise DataContractError(f"{raw_root}/{split}: block id {part_id} not found")
    block = paths[part_id]
    data = pl.read_parquet(block.data)
    users = pl.read_parquet(block.user_info)
    items = pl.read_parquet(block.item_info)
    require_columns(
        data.schema,
        (CLIENT_COLUMN, POSITION_COLUMN, "item_index", *LABEL_COLUMNS, *USER_STATIC_COLUMNS, *CONTEXT_COLUMNS),
        block.data,
    )
    require_columns(
        users.schema,
        ("user_index", "full_item_seq", "full_action_seq", "full_timestamp_seq"),
        block.user_info,
    )
    require_columns(items.schema, ("item_index", "item_id", *ITEM_COLUMNS[1:]), block.item_info)
    if items.schema["item_index"] == pl.Int64() and (items["item_index"] == 0).sum() != 1:
        raise DataContractError(f"{block.item_info} must contain the single 0 padding row")
    return data, users, items
