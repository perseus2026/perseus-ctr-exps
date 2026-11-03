# ruff: noqa: INP001, PLC0415

import argparse
import contextlib
import json
import logging
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)


def write_stores(source_path: Path, output: Path, target_ids: list[int]) -> dict:
    source = pa.ipc.open_file(source_path / "data.arrow")
    index = pl.read_parquet(source_path / "index.pq")
    if source.num_record_batches != index.height:
        raise ValueError("Event batches and client index differ")
    values = pa.array(target_ids, type=source.schema.field("event").type)
    counts = {"target": 0, "source": 0}
    with contextlib.ExitStack() as stack:
        writers = {}
        for name in counts:
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            writers[name] = stack.enter_context(
                pa.ipc.new_file(
                    str(folder / "data.arrow"),
                    source.schema,
                    options=pa.ipc.IpcWriteOptions(compression="zstd"),
                ),
            )
        for i in range(source.num_record_batches):
            batch = source.get_batch(i)
            mask = pc.is_in(batch.column("event"), value_set=values)
            for name, selected in [
                ("target", batch.filter(mask)),
                ("source", batch.filter(pc.invert(mask))),
            ]:
                counts[name] += selected.num_rows
                if selected.num_rows == 0:
                    selected = batch.slice(0, 1)
                    position = selected.schema.get_field_index("timestamp")
                    field = selected.schema.field(position)
                    selected = selected.set_column(
                        position,
                        field,
                        pa.array([8_000_000_000_000_000_000], field.type),
                    )
                position = selected.schema.get_field_index("position")
                field = selected.schema.field(position)
                selected = selected.set_column(
                    position,
                    field,
                    pa.array(range(selected.num_rows, 0, -1), field.type),
                )
                writers[name].write_batch(selected)
            if (i + 1) % 50000 == 0:
                logger.info(
                    "Prepared domain histories: %s/%s users",
                    i + 1,
                    source.num_record_batches,
                )
    for name in counts:
        shutil.copyfile(source_path / "index.pq", output / name / "index.pq")
    return {"users": source.num_record_batches, "real_event_rows": counts}


def main() -> None:
    from perseus import core

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prefix", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    checkpoint = core.checkpoint.Prepared.load(args.workdir / "checkpoint")
    values = checkpoint.encoder_to_preprocessor["event"].values
    target_ids = values.filter(pl.col("value").str.starts_with(args.target_prefix))[
        "index"
    ].to_list()
    report = write_stores(args.workdir / "dataset/events", args.output, target_ids)
    (args.output / "lcn-stores.json").write_text(
        json.dumps(
            {
                "status": "prepared",
                "target_prefix": args.target_prefix,
                **report,
                "future_dummy_for_missing_domain": True,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
