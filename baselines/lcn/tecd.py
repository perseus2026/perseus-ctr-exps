import hashlib
import math

import numpy as np
import polars as pl
import torch


def parameter_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(str((value.shape, value.dtype)).encode())
        result.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return result.hexdigest()


def audit_dataset(checkpoint, dataset, args):
    names = list(dataset.task_preprocessor.classes)
    if names != ["click", "cart", "order"]:
        raise ValueError(f"Wrong task order: {names}")
    if "split" in dataset.test_samples.columns:
        raise ValueError("Unexpected validation split")
    if args.batch_size * args.accumulation != 32768:
        raise ValueError("Global batch must match the primary grid: 32768")
    counts = dataset.test_samples.select(pl.col("target").struct.unnest()).sum().row(0)
    if not args.max_train_batches:
        if (
            dataset.train_samples.height != 213635177
            or dataset.test_samples.height != 53026640
        ):
            raise ValueError("Unexpected full-data sample counts")
        if list(counts) != [410511, 708199, 1279677]:
            raise ValueError(f"Test label counts differ: {counts}")
    return {
        "train_rows": dataset.train_samples.height,
        "test_rows": dataset.test_samples.height,
        "test_positives": dict(zip(names, counts, strict=True)),
        "global_batch": 32768,
        "seed": args.seed,
        "history": args.history,
        "epochs": 1,
        "expected_optimizer_updates": math.ceil(dataset.train_samples.height / 32768),
        "event_vocabulary": checkpoint.encoder_to_preprocessor[
            "event"
        ].values.to_dicts(),
    }


def audit_history(reader, frame, target_ids, history):
    indices = np.linspace(
        0, frame.height - 1, min(32, frame.height), dtype=int
    ).tolist()
    samples = frame[indices]
    boundaries = samples["timestamp"].dt.epoch("ns").to_list()
    events, lengths = reader.read_batch(samples["client_id"].to_list(), boundaries)
    offset = 0
    for boundary, length in zip(boundaries, lengths, strict=True):
        if length > 512:
            raise ValueError("History exceeds the shared cap")
        selected = events.slice(offset, length)
        offset += length
        if length and selected["timestamp"].dt.epoch("ns").max() >= boundary:
            raise ValueError("Future events reached the model")
        if (
            history == "single"
            and length
            and not selected["event"].is_in(target_ids).all()
        ):
            raise ValueError("Single-domain history contains auxiliary events")
    return {
        "probes": len(indices),
        "max_history": max(lengths),
        "strict_temporal_cutoff": True,
    }
