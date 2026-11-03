"""Collect Taobao Ads sources and prepare category-domain datasets."""

import argparse
import logging
from pathlib import Path

from . import _download, _history, _prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "configs/cross_domain/taobao_ads",
    )
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument(
        "--protocol",
        choices=["train-test-6-1", "train-valid-test"],
        default="train-test-6-1",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    impressions = args.raw_root / "impressions"
    history = args.raw_root / "history"
    if args.download:
        _download.collect(argparse.Namespace(output=impressions))
        _history.collect(argparse.Namespace(output=history))
    _prepare.prepare(
        argparse.Namespace(
            raw=impressions / "raw",
            history=history / "tables/behavior_log.pq",
            templates=args.templates,
            output=args.out_root,
            scratch=args.scratch
            or args.out_root.with_name(args.out_root.name + "-scratch"),
            shards=args.shards,
            protocol=args.protocol,
        )
    )


if __name__ == "__main__":
    main()
