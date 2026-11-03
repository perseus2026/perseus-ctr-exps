# ruff: noqa: INP001

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from perseus import core
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from .lcn import LCN
from .tecd import audit_dataset, audit_history, parameter_hash


class DomainReader:
    def __init__(self, target, source=None) -> None:
        self.target, self.source = target, source
        self.allow_empty_sequences = True
        self.max_events_per_sequence = target.max_events_per_sequence + (
            source.max_events_per_sequence if source else 0
        )

    def read(self, client_id, /, *, before=None):
        target = self.target.read(client_id, before=before)
        return (
            target
            if self.source is None
            else pl.concat([target, self.source.read(client_id, before=before)])
        )

    def read_batch(self, clients, boundaries, /, *, features=None):
        target, nt = self.target.read_batch(clients, boundaries, features=features)
        if self.source is None:
            return target, nt
        source, ns = self.source.read_batch(clients, boundaries, features=features)
        frames, lengths = [], []
        it = ins = 0
        for a, b in zip(nt, ns, strict=True):
            frames.extend([target.slice(it, a), source.slice(ins, b)])
            lengths.append(a + b)
            it += a
            ins += b
        return pl.concat(frames), lengths


def evaluate(
    model: LCN,
    loader: DataLoader,
    names: list[str],
    accelerator: Accelerator,
    output: Path,
    split: str,
    epoch: int,
) -> dict:
    model.eval()
    scores, targets = [], []
    prediction_writer = None
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{split} epoch={epoch}"):
            users = torch.tensor(
                [int(user) for user in batch.extras["_client_id"]],
                device=accelerator.device,
            )
            with accelerator.autocast():
                logits, _ = model(
                    batch.events_features,
                    batch.events_positions,
                    batch.context_features,
                    users,
                )
            prediction = logits.float().sigmoid().cpu().numpy()
            target = batch.target.cpu().numpy()
            scores.append(prediction)
            targets.append(target)
            if split == "test":
                table = pl.DataFrame(
                    {
                        "sample_id": batch.extras["sample_id"],
                        "client_id": batch.extras["_client_id"],
                        **{
                            f"prediction_{name}": prediction[:, i]
                            for i, name in enumerate(names)
                        },
                        **{
                            f"target_{name}": target[:, i].astype(np.uint8)
                            for i, name in enumerate(names)
                        },
                    },
                ).to_arrow()
                if prediction_writer is None:
                    prediction_writer = pq.ParquetWriter(
                        output / "test-predictions.pq",
                        table.schema,
                        compression="zstd",
                    )
                prediction_writer.write_table(table)
    if prediction_writer is not None:
        prediction_writer.close()
    scores, targets = np.concatenate(scores), np.concatenate(targets)
    metrics = {}
    for i, name in enumerate(names):
        metrics[name] = {
            "auc": float(roc_auc_score(targets[:, i], scores[:, i]))
            if np.unique(targets[:, i]).size == 2
            else None,
            "logloss": float(log_loss(targets[:, i], scores[:, i], labels=[0, 1])),
            "rows": int(targets.shape[0]),
            "positives": int(targets[:, i].sum()),
        }
    metrics["mean_logloss"] = float(
        np.mean([metrics[name]["logloss"] for name in names])
    )
    (output / f"{split}-epoch-{epoch}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return metrics


def main() -> None:
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--stores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--identity-fields", nargs="+", required=True)
    parser.add_argument("--history", choices=["single", "cross"], required=True)
    parser.add_argument(
        "--model-mode", choices=["auto", "source_lap", "target_lap"], default="auto"
    )
    parser.add_argument("--history-limit", type=int)
    parser.add_argument("--fallback-to-target", action="store_true")
    parser.add_argument("--compact-samples", action="store_true")
    parser.add_argument("--expected-tecd-counts", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--eval-rows", type=int)
    parser.add_argument("--target-budget", type=int)
    parser.add_argument("--source-budget", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crp-weight", type=float, default=0.01)
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    accelerator = Accelerator(
        mixed_precision="bf16", gradient_accumulation_steps=args.accumulation
    )
    if accelerator.num_processes != 1:
        raise ValueError("The experimental LCN runner uses one GPU")
    if accelerator.gradient_accumulation_steps != args.accumulation:
        raise ValueError("Accelerate changed the requested effective batch")
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = core.checkpoint.Prepared.load(args.workdir / "checkpoint")
    dataset = core.Dataset.load(args.workdir / "dataset", checkpoint.config)
    if args.compact_samples:
        dataset.train_samples = dataset.train_samples.rechunk()
        dataset.test_samples = dataset.test_samples.rechunk()
    audit = (
        audit_dataset(checkpoint, dataset, args) if args.expected_tecd_counts else None
    )
    if args.max_train_batches:
        dataset.train_samples = dataset.train_samples.head(
            args.max_train_batches * args.batch_size
        )
    if args.eval_rows:
        dataset.test_samples = dataset.test_samples.head(args.eval_rows)
    cross = args.history == "cross"
    target_budget = (
        args.target_budget if args.target_budget is not None else (50 if cross else 500)
    )
    source_budget = (
        args.source_budget if args.source_budget is not None else (500 if cross else 0)
    )
    if args.history_limit is not None:
        if args.history_limit <= 0 or args.stores is not None:
            raise ValueError(
                "Use a positive common history limit with the prepared event store"
            )
        reader = core.EventStore(
            args.workdir / "dataset/events",
            backend="cached",
            allow_empty_sequences=True,
        ).open_reader(args.history_limit)
        target_budget = source_budget = None
    else:
        if args.stores is None:
            raise ValueError("Domain-budget mode requires --stores")
        target = core.EventStore(args.stores / "target", backend="cached").open_reader(
            target_budget
        )
        source = (
            core.EventStore(args.stores / "source", backend="cached").open_reader(
                source_budget
            )
            if source_budget
            else None
        )
        reader = DomainReader(target, source)
    model_cross = args.model_mode == "source_lap" or (
        args.model_mode == "auto" and cross
    )
    if args.model_mode == "target_lap" and source_budget:
        raise ValueError("Target-LAP control must not receive source history")
    event_fields = checkpoint.config.feature_to_encoder(event=True)
    context_fields = checkpoint.config.feature_to_encoder(context=True)
    events = checkpoint.encoder_to_preprocessor["event"].values
    target_ids = events.filter(pl.col("value").str.starts_with(args.target_prefix))[
        "index"
    ].to_list()
    cardinalities = {
        name: preprocessor.cardinality
        for name, preprocessor in checkpoint.encoder_to_preprocessor.items()
    }
    names = list(dataset.task_preprocessor.classes)
    model = LCN(
        cardinalities,
        list(context_fields),
        args.identity_fields,
        target_ids,
        len(names),
        cross_domain=model_cross,
        fallback_to_target=args.fallback_to_target,
    )
    if audit is not None:
        if (
            args.history_limit != 512
            or not args.fallback_to_target
            or not model_cross
            or not args.no_validation
            or args.epochs != 1
        ):
            raise ValueError(
                "The recorded T-ECD study requires fixed source-LAP with fallback, 512 events and one epoch"
            )
        audit["history_checks"] = audit_history(
            reader, dataset.train_samples, target_ids, args.history
        )
        (args.output / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def loader(frame: pl.DataFrame, *, training: bool) -> DataLoader:
        return core.dataloaders.samples(
            frame.with_columns(pl.col("client_id").alias("_client_id")),
            reader,
            {
                name: checkpoint.encoder_to_preprocessor[enc]
                for name, enc in event_fields.items()
            },
            {
                name: checkpoint.encoder_to_preprocessor[enc]
                for name, enc in context_fields.items()
            },
            task_preprocessor=dataset.task_preprocessor,
            extras=["_client_id"] if training else ["_client_id", "sample_id"],
            fixed_padding=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=training,
            persistent_workers=args.workers > 0,
        )

    train = loader(dataset.train_samples, training=True)
    if args.no_validation:
        if "split" in dataset.test_samples.columns:
            raise ValueError(
                "Train/test preparation must remove the validation split column"
            )
        valid = None
        test = loader(dataset.test_samples, training=False)
        model, optimizer, train, test = accelerator.prepare(
            model, optimizer, train, test
        )
    else:
        valid = loader(
            dataset.test_samples.filter(pl.col("split") == "valid"), training=False
        )
        test = loader(
            dataset.test_samples.filter(pl.col("split") == "test"), training=False
        )
        model, optimizer, train, valid, test = accelerator.prepare(
            model, optimizer, train, valid, test
        )
    specification = {
        "paper": "2312.06424v2",
        "implementation": "independent_LAP_CRP_reimplementation",
        "adapted_dataset": True,
        "dim": 64,
        "heads": 4,
        "top_k1": 200,
        "top_k2": 50,
        "short_length": 50,
        "long_length": args.history_limit or 500,
        "csa_msa_attention": "unnormalized_equations_7_to_10",
        "crp_denominator": "negative_pairs_only_equations_3_to_5",
        "crp_temperature": 0.1,
        "crp_negative_pairs": 512,
        "crp_weight": args.crp_weight,
        "crp_pair_weights": [1, 1, 1],
        "optimizer": "Adam",
        "lr": 0.001,
        "micro_batch": args.batch_size,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "selection": "fixed epochs" if args.no_validation else "valid mean logloss",
        "identity_fields": args.identity_fields,
        "context_fields": list(context_fields),
        "target_names": names,
        "history": args.history,
        "model_mode": "source_lap" if model_cross else "target_lap",
        "long_stream": (
            "source with per-example target fallback"
            if model_cross and args.fallback_to_target
            else "source"
            if model_cross
            else "target"
        ),
        "crp_active": model_cross,
        "target_history_budget": target_budget,
        "source_history_budget": source_budget,
        "total_history_budget": args.history_limit,
        "fallback_to_target": args.fallback_to_target,
        "initial_parameter_hash": parameter_hash(model)
        if args.history_limit is not None
        else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_audit": audit,
        "smoke_only": bool(args.max_train_batches or args.eval_rows),
        "domain_window_layout": "valid tokens right-aligned across streams",
        "dataloader_start_method": "spawn",
    }
    (args.output / "model-spec.json").write_text(
        json.dumps(specification, indent=2) + "\n"
    )
    best = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        window_rows = 0
        rolling = []
        for step, batch in enumerate(tqdm(train, desc=f"train epoch={epoch}"), 1):
            rows = batch.target.shape[0]
            window_rows += rows
            users = torch.tensor(
                [int(user) for user in batch.extras["_client_id"]],
                device=accelerator.device,
            )
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits, aux = model(
                        batch.events_features,
                        batch.events_positions,
                        batch.context_features,
                        users,
                    )
                bce = (
                    functional.binary_cross_entropy_with_logits(
                        logits.float(), batch.target.float(), reduction="none"
                    )
                    .mean(0)
                    .sum()
                )
                objective = bce + args.crp_weight * aux
                if not torch.isfinite(objective):
                    raise ValueError(
                        f"Nonfinite objective at epoch {epoch}, step {step}"
                    )
                accelerator.backward(objective * rows / args.batch_size)
                if accelerator.sync_gradients:
                    nominal = args.batch_size * args.accumulation
                    if window_rows != nominal:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(nominal / window_rows)
                    dense = [
                        parameter
                        for name, parameter in model.named_parameters()
                        if "embeddings." not in name
                    ]
                    accelerator.clip_grad_norm_(dense, 10)
                    window_rows = 0
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            rolling.append((float(bce.detach()), float(aux.detach())))
            if step % 100 == 0:
                print(
                    json.dumps(
                        {"step": step, "bce": rolling[-1][0], "crp": rolling[-1][1]}
                    ),
                    flush=True,
                )
                rolling = []
        state = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "specification": specification,
        }
        torch.save(state, args.output / "last.pt")
        if args.no_validation:
            continue
        metrics = evaluate(
            model, valid, names, accelerator, args.output, "valid", epoch
        )
        if metrics["mean_logloss"] < best:
            best = metrics["mean_logloss"]
            best_epoch = epoch
            torch.save(state, args.output / "best.pt")
        else:
            break
    if args.no_validation:
        selected_epoch = args.epochs
        torch.save(state, args.output / "final.pt")
    else:
        state = torch.load(
            args.output / "best.pt", map_location="cpu", weights_only=False
        )
        accelerator.unwrap_model(model).load_state_dict(state["model"])
        selected_epoch = best_epoch
    test_metrics = evaluate(
        model, test, names, accelerator, args.output, "test", selected_epoch
    )
    shutil.copytree(
        args.workdir / "checkpoint/encoders/preprocessors",
        args.output / "preprocessors",
        dirs_exist_ok=True,
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "selection": specification["selection"],
                "selected_epoch": selected_epoch,
                "best_epoch": None if args.no_validation else best_epoch,
                "valid_mean_logloss": None if args.no_validation else best,
                "test": test_metrics,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
