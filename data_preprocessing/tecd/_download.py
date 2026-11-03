# ruff: noqa: INP001

import argparse
import json
import logging
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from data_preprocessing._cross_domain import checksum

logger = logging.getLogger(__name__)

# Frozen public snapshot. Only the ranges are stored in Git; file hashes are
# recorded in the generated local source-manifest.json after downloading.
REPO_ID = "t-tech/T-ECD"
REVISION = "79bcf78cd4699267f561ad3fd51fe7b52af53083"
SMALL_DAYS = range(1082, 1309)
PAYMENT_DAYS = range(715, 1309)
DOMAINS = ("retail", "marketplace", "offers", "payments")


def source_manifest() -> dict:
    paths = ["dataset/small/users.pq", "dataset/small/brands.pq"]
    for domain in DOMAINS[:-1]:
        paths.append(f"dataset/small/{domain}/items.pq")
        paths.extend(f"dataset/small/{domain}/events/{day:05d}.pq" for day in SMALL_DAYS)
    paths.extend(f"dataset/full/payments/events/{day:05d}.pq" for day in PAYMENT_DAYS)
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "domains": list(DOMAINS),
        "day_begin": SMALL_DAYS.start,
        "day_end": SMALL_DAYS.stop - 1,
        "files": [{"path": path} for path in paths],
    }


def load_manifest(path: Path) -> dict:
    expected = source_manifest()
    if not path.exists():
        return expected
    recorded = json.loads(path.read_text())
    if (recorded.get("repo_id") != REPO_ID or recorded.get("revision") != REVISION
            or {f["path"] for f in recorded["files"]} != {f["path"] for f in expected["files"]}
            or len(recorded["files"]) != len(expected["files"])):
        raise ValueError("Existing source-manifest.json belongs to a different dataset snapshot")
    return recorded


def save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def download_file(manifest: dict, entry: dict, directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(
        repo_id=manifest["repo_id"], repo_type="dataset",
        revision=manifest["revision"], filename=entry["path"],
        local_dir=directory, token=False,
    ))
    digest = checksum(downloaded)
    if "sha256" in entry and entry["sha256"] != digest:
        raise ValueError(f"Source checksum mismatch: {entry['path']}")
    if "size_bytes" in entry and entry["size_bytes"] != downloaded.stat().st_size:
        raise ValueError(f"Source size mismatch: {entry['path']}")
    entry["sha256"] = digest
    entry["size_bytes"] = downloaded.stat().st_size
    return downloaded



def filter_payments(
    source: Path, destination: Path, users: pl.DataFrame, brands: pl.DataFrame
) -> int:
    frame = pl.scan_parquet(source)
    schema = frame.collect_schema()
    frame = frame.join(
        users.select(pl.col("user_id").cast(schema["user_id"])).lazy(),
        on="user_id",
        how="semi",
        maintain_order="left",
    ).join(
        brands.select(pl.col("brand_id").cast(schema["brand_id"])).lazy(),
        on="brand_id",
        how="semi",
        maintain_order="left",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.pq")
    frame.sink_parquet(temporary, compression="zstd", maintain_order=True)
    rows = pq.ParquetFile(temporary).metadata.num_rows
    temporary.replace(destination)
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("retail", "marketplace", "offers", "payments"),
        default=["retail", "marketplace", "offers", "payments"],
    )
    parser.add_argument("--payments-days", nargs="+", type=int)
    parser.add_argument("--small-days", nargs="+", type=int)
    collect(parser.parse_args())


def collect(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = args.output / "source-manifest.json"
    manifest = load_manifest(source_manifest_path)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    expected_domains = {"retail", "marketplace", "offers", "payments"}
    if set(manifest["domains"]) != expected_domains:
        raise ValueError("Expected the four-domain source manifest")
    if len(entries) != len(manifest["files"]):
        raise ValueError("Duplicate source paths")
    for path in entries:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"Invalid source path: {path}")
    domains = set(args.domains)
    if args.payments_days and "payments" not in domains:
        raise ValueError("--payments-days requires the payments domain")
    available_days = {
        int(Path(path).stem)
        for path in entries
        if path.startswith("dataset/full/payments/events/")
    }
    if args.payments_days and not set(args.payments_days) <= available_days:
        raise ValueError("Payment days are outside the frozen manifest")

    raw = args.output / "raw"
    references = {"dataset/small/users.pq", "dataset/small/brands.pq"}
    small = [
        entry
        for path, entry in entries.items()
        if path in references
        or any(path.startswith(f"dataset/small/{domain}/") for domain in domains)
    ]
    if args.small_days:
        if not set(args.small_days) <= set(
            range(manifest["day_begin"], manifest["day_end"] + 1)
        ):
            raise ValueError("Small days are outside the frozen manifest")
        small = [
            entry
            for entry in small
            if "/events/" not in entry["path"]
            or int(Path(entry["path"]).stem) in args.small_days
        ]
    for index, entry in enumerate(small, start=1):
        download_file(manifest, entry, raw)
        logger.info("Small %s/%s: %s", index, len(small), entry["path"])

    # Keep the large inventory in the local data directory, never in Git.
    save_manifest(source_manifest_path, manifest)

    payment_reports = []
    if "payments" in domains:
        users = pl.read_parquet(raw / "dataset/small/users.pq", columns=["user_id"])
        brands = pl.read_parquet(raw / "dataset/small/brands.pq", columns=["brand_id"])
        payments = [
            entry
            for path, entry in sorted(entries.items())
            if path.startswith("dataset/full/payments/events/")
            and (not args.payments_days or int(Path(path).stem) in args.payments_days)
        ]
        spool = args.output / ".payment-downloads"
        spool.mkdir(exist_ok=True)
        for index, entry in enumerate(payments, start=1):
            destination = (
                args.output
                / "filtered/small-payments/events"
                / Path(entry["path"]).name
            )
            receipt = destination.with_suffix(".json")
            if receipt.exists() and destination.exists():
                report = json.loads(receipt.read_text())
                if ("sha256" in entry and report["source_sha256"] != entry["sha256"]) or (
                    len(report.get("source_sha256", "")) != 64
                ) or report["filtered_sha256"] != checksum(destination):
                    raise ValueError(f"Filtered checksum mismatch: {destination}")
            else:
                with tempfile.TemporaryDirectory(dir=spool) as temporary:
                    source = download_file(manifest, entry, Path(temporary))
                    rows = filter_payments(source, destination, users, brands)
                report = {
                    "source": entry["path"],
                    "source_sha256": entry["sha256"],
                    "filtered_rows": rows,
                    "filtered_sha256": checksum(destination),
                }
                receipt.write_text(json.dumps(report, indent=2) + "\n")
            # Payment hashes live in per-day local receipts. Avoid writing a huge
            # version-controlled inventory, or duplicating them in the manifest.
            entry.pop("sha256", None)
            entry.pop("size_bytes", None)
            payment_reports.append(report)
            logger.info(
                "Payments %s/%s: %s, retained %s rows",
                index,
                len(payments),
                entry["path"],
                report["filtered_rows"],
            )

    full = (
        domains == expected_domains
        and not args.small_days
        and len(payment_reports) == len(available_days)
    )
    report = {
        "status": "sources_collected" if full else "partial_sources_collected",
        "revision": manifest["revision"],
        "source_manifest_sha256": checksum(source_manifest_path),
        "domains": sorted(domains),
        "small_files": len(small),
        "small_days": args.small_days,
        "payment_files": len(payment_reports),
        "filtered_payment_rows": sum(item["filtered_rows"] for item in payment_reports),
        "prepared_model_dataset": False,
    }
    (args.output / "collection-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("Collection report: %s", report)


if __name__ == "__main__":
    main()
