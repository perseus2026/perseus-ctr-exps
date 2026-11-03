"""One CLI for the two final cross-domain studies and all five recipes.

Run from the experiment repository root after preparing the corresponding data.
"""

import argparse
import sys
from pathlib import Path

from ._config import STUDIES


MODELS = ("perseus", "est", "hemix", "hyformer", "lcn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tecd", "taobao_ads"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--history", choices=("single", "cross"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing single/ and cross/ from data conversion")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--config", type=Path, help="Override the model recipe")
    parser.add_argument("--output", type=Path, help="Override the complete run directory")
    parser.add_argument("--stores", type=Path,
                        help="Domain-separated stores prepared once for Taobao LCN")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    study = STUDIES[args.dataset]
    if args.seed not in study["seeds"]:
        parser.error(f"Seed must be one of {study['seeds']}")
    data = args.data_root / args.history
    if not (data / "dataset").is_dir() or not (data / "checkpoint").is_dir():
        parser.error(f"Prepared data missing: {data} (run data conversion first)")
    config = args.config or repo / "configs/cross_domain" / args.dataset / f"{args.model}_{args.history}.yaml"
    if not config.is_file():
        parser.error(f"Configuration missing: {config}")
    output = args.output or args.runs_dir / args.dataset / f"{args.model}-{args.history}-s{args.seed}"
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "lcn":
        if args.dataset == "taobao_ads":
            if args.stores is None or not (args.stores / "target/data.arrow").is_file() or not (args.stores / "source/data.arrow").is_file():
                parser.error("Taobao LCN needs --stores PATH; prepare once with prepare_lcn_stores")
        elif args.stores is not None:
            parser.error("--stores is only used for Taobao LCN")
        from baselines.lcn import run as lcn
        command = ["lcn", "--data", str(data), "--config", str(config),
                   "--seed", str(args.seed), "--output", str(output)]
        if args.stores:
            command.extend(["--stores", str(args.stores)])
        runner = lcn.main
    else:
        if args.stores is not None:
            parser.error("--stores is only used for Taobao LCN")
        command = ["train", "--data", str(data), "--config", str(config),
                   "--seed", str(args.seed), "--output", str(output),
                   "--global-batch", str(study["global_batch"])]
        if args.dataset == "tecd":
            from . import _train
            command.append("--compact-samples")
            runner = _train.main
        else:
            from . import taobao
            command += ["--history", args.history]
            runner = taobao.main

    previous_argv = sys.argv
    try:
        sys.argv = command
        runner()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
