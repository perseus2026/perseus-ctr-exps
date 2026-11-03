import hashlib
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(manifest: dict, entry: dict, directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=manifest["repo_id"],
            repo_type="dataset",
            revision=manifest["revision"],
            filename=entry["path"],
            local_dir=directory,
            token=False,
        ),
    )
    if path.stat().st_size != entry["size_bytes"] or checksum(path) != entry["sha256"]:
        raise ValueError(f"Source checksum mismatch: {entry['path']}")
    return path


def write_partitioned(frames: Iterator[pl.DataFrame], destination: Path) -> None:
    first = next(frames)
    schema = first.to_arrow().schema

    def batches() -> Iterator[pa.RecordBatch]:
        yield from first.to_arrow().to_batches()
        for frame in frames:
            yield from frame.to_arrow().cast(schema).to_batches()

    ds.write_dataset(
        batches(),
        destination,
        schema=schema,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([schema.field("shard")]), flavor="hive"),
        max_open_files=128,
        min_rows_per_group=131_072,
        max_rows_per_group=262_144,
        max_rows_per_file=1_048_576,
        use_threads=False,
    )
