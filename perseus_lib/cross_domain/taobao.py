"""Run unchanged SeqCTR architectures with 50 target + 0/500 auxiliary events."""

import argparse
import hashlib
import importlib
import json
import multiprocessing
import sys
from pathlib import Path

import polars as pl
import torch
from perseus import core

from perseus_lib.cross_domain import _train as base

from .taobao_history import AdditiveStore


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history", choices=["single", "cross"], required=True)
    args, remainder = parser.parse_known_args()
    output = Path(remainder[remainder.index("--output") + 1])
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    original_load = core.Dataset.load
    fit = importlib.import_module("perseus.api.fit_model")
    original_optimizer = fit._create_optimizer

    @classmethod
    def load(cls, path, config):
        dataset = original_load(path, config)
        if config.max_events_per_sequence != 550:
            raise ValueError("Model capacity must be 550 in both conditions")
        cp = core.checkpoint.Prepared.load(Path(path).parent / "checkpoint")
        ids = (
            cp.encoder_to_preprocessor["event"]
            .values.filter(pl.col("value").str.starts_with("target-"))["index"]
            .to_list()
        )
        if not ids:
            raise ValueError("Target events not found")
        counts = {
            "train_rows": dataset.train_samples.height,
            "test_rows": dataset.test_samples.height,
            "test_positives": dataset.test_samples.select(
                pl.col("target").struct.unnest()
            )
            .sum()
            .row(0),
        }
        if counts != {
            "train_rows": 2797061,
            "test_rows": 470803,
            "test_positives": (24550,),
        }:
            raise ValueError(f"Unexpected full data: {counts}")
        dataset.event_store = AdditiveStore(
            Path(path) / "events", ids, 500 if args.history == "cross" else 0
        )
        (reports / "history-protocol.json").write_text(
            json.dumps(
                {
                    **counts,
                    "history": args.history,
                    "target_budget": 50,
                    "source_budget": 500 if args.history == "cross" else 0,
                    "model_capacity": 550,
                    "ordering": "chronological; original event-store order breaks equal-timestamp ties",
                },
                indent=2,
            )
            + "\n"
        )
        return dataset

    def optimizer(accelerator, embedders, backbone, task_layer, criterion, checkpoint):
        digest = hashlib.sha256()
        modules = {
            **{f"encoder/{n}": m for n, m in embedders.items()},
            "backbone": backbone,
            "task": task_layer,
        }
        for prefix, module in sorted(modules.items()):
            for name, value in sorted(module.state_dict().items()):
                digest.update((prefix + "/" + name).encode())
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(
                    value.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        (reports / "initial-state.json").write_text(
            json.dumps(
                {
                    "sha256": digest.hexdigest(),
                    "seed": checkpoint.config.training.seed,
                    "parameters": sum(
                        p.numel() for m in modules.values() for p in m.parameters()
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return original_optimizer(
            accelerator, embedders, backbone, task_layer, criterion, checkpoint
        )

    core.Dataset.load = load
    fit._create_optimizer = optimizer
    sys.argv = [sys.argv[0], *remainder]
    try:
        base.main()
    finally:
        core.Dataset.load = original_load
        fit._create_optimizer = original_optimizer


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
