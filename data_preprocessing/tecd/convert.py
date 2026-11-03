"""Collect verified T-ECD sources and prepare single/cross-domain datasets."""

import argparse
import logging
from pathlib import Path

from . import _download, _prepare, _retail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--payments", type=Path, help="Separate collected Payments shards"
    )
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/cross_domain/tecd",
    )
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--protocol",
        choices=["train-test-80-20", "train-valid-test"],
        default="train-test-80-20",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.download:
        _download.collect(
            argparse.Namespace(
                output=args.raw_root,
                domains=["retail", "marketplace", "offers", "payments"],
                small_days=None,
                payments_days=None,
            )
        )
    if args.protocol == "train-valid-test":
        _retail.prepare(
            argparse.Namespace(source=args.raw_root, output=args.raw_root / "prepared")
        )
    _prepare.prepare(
        argparse.Namespace(
            small=args.raw_root,
            payments=args.payments or args.raw_root,
            templates=args.templates,
            output=args.out_root,
            scratch=args.scratch
            or args.out_root.with_name(args.out_root.name + "-scratch"),
            shards=args.shards,
            batch_size=args.batch_size,
            protocol=args.protocol,
        )
    )


if __name__ == "__main__":
    main()
