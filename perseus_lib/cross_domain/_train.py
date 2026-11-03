import argparse
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import subprocess
import time
from pathlib import Path

from accelerate import PartialState
from perseus import api, core
from perseus.core.tasks.classification.evaluation import Evaluator

from ._config import load_recipe, save_recipe


def revision(path: Path) -> str | None:
    """Record an available Git revision without requiring a Git checkout."""
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-revision", help="Optional expected library revision")
    parser.add_argument("--experiments-revision", help="Optional expected experiment revision")
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--compact-samples", action="store_true")
    args = parser.parse_args()
    import perseus

    state = PartialState()
    repo = Path(__file__).resolve().parents[2]
    library = Path(perseus.__file__).resolve().parents[1]
    library_revision = revision(library)
    experiments_revision = revision(repo)
    if args.library_revision is not None and library_revision != args.library_revision:
        raise ValueError("Library revision differs from the explicitly requested revision")
    if args.experiments_revision is not None and experiments_revision != args.experiments_revision:
        raise ValueError("Experiment revision differs from the explicitly requested revision")
    recipe = load_recipe(args.config, args.seed)
    config = core.Config.model_validate(recipe)
    if config.training.num_epochs != 1 or config.training.early_stopping is not None:
        raise ValueError(
            "This protocol requires one fixed epoch without test-based selection"
        )
    global_batch = (
        config.training.dataloader.batch_size
        * (config.training.accumulation or 1)
        * state.num_processes
    )
    if global_batch != args.global_batch:
        raise ValueError(
            f"Effective batch {global_batch} differs from {args.global_batch}"
        )
    prepared = core.checkpoint.Prepared.load(args.data / "checkpoint")
    if prepared.config.feature_to_encoder(
        event=True, context=True, artifacts=True
    ) != config.feature_to_encoder(event=True, context=True, artifacts=True):
        raise ValueError("Prepared feature schema differs from the training config")
    prepared.config = config
    dataset = core.Dataset.load(args.data / "dataset", config)
    if args.compact_samples:
        dataset.train_samples = dataset.train_samples.rechunk()
        dataset.test_samples = dataset.test_samples.rechunk()
    if "split" in dataset.test_samples.columns or not dataset.test_samples.height:
        raise ValueError("Expected a standalone nonempty test without validation")
    reports = args.output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    effective_config = reports / "effective-config.yaml"
    if state.is_main_process:
        save_recipe(effective_config, recipe)
    original = core.tasks.registry["classification"]

    class ReportingEvaluator(Evaluator):
        def calculate_metrics(self, samples, artifacts, /):
            metrics = super().calculate_metrics(samples, artifacts)
            if state.is_main_process:
                columns = [name for name in ("sample_id", "client_id", "timestamp", "target", "prediction")
                           if name in samples.columns]
                samples.select(columns).write_parquet(reports / "test-predictions.pq")
                targets = {}
                for name, values in metrics.items():
                    metric, _, target = name.partition("/")
                    if metric in {"roc_auc", "logloss"} and "overall" in values:
                        targets.setdefault(target, {})[
                            "auc" if metric == "roc_auc" else metric
                        ] = values["overall"]
                (reports / "test-metrics.json").write_text(
                    json.dumps({"test": targets}, indent=2) + "\n"
                )
            return metrics

    core.tasks.registry["classification"] = dataclasses.replace(
        original, evaluator=ReportingEvaluator
    )
    started = time.monotonic()
    try:
        trained = api.fit_model(prepared, dataset, fixed_padding=True)
        if state.is_main_process:
            trained.save(args.output / "checkpoint")
            evidence = {
                "library_revision": library_revision,
                "experiments_revision": experiments_revision,
                "config_path": str(args.config),
                "config_sha256": hashlib.sha256(
                    effective_config.read_bytes()
                ).hexdigest(),
                "source_config_sha256": hashlib.sha256(
                    args.config.read_bytes()
                ).hexdigest(),
                "seed": config.training.seed,
                "compact_samples": args.compact_samples,
                "global_batch": global_batch,
                "world_size": state.num_processes,
                "device": str(state.device),
                "train_rows": dataset.train_samples.height,
                "test_rows": dataset.test_samples.height,
                "wall_seconds": time.monotonic() - started,
                "selection": "one fixed epoch; no validation",
                "data": str(args.data),
            }
            (reports / "run.json").write_text(json.dumps(evidence, indent=2) + "\n")
    finally:
        core.tasks.registry["classification"] = original
    state.wait_for_everyone()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    multiprocessing.set_start_method("spawn", force=True)
    main()
