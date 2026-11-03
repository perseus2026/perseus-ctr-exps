"""Select fixed target and auxiliary budgets, then merge chronologically."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from perseus.core.event_store.cached import EventsReader


class AdditiveReader(EventsReader):
    def __init__(self, path, target_ids, *, source_budget):
        if source_budget not in [0, 500]:
            raise ValueError("Auxiliary budget must be 0 or 500")
        super().__init__(
            Path(path), max_events_per_sequence=550, allow_empty_sequences=True
        )
        self.source_budget = source_budget
        target = np.isin(self._columns["event"].numpy(), target_ids)
        self._domain_indices = (np.flatnonzero(target), np.flatnonzero(~target))

    def read_batch(self, client_ids, boundaries, /, *, features=None):
        if len(client_ids) != len(boundaries):
            raise ValueError("Clients and boundaries must match")
        names = list(self._columns) if features is None else features
        timestamps = self._timestamps.numpy()
        indices, positions, lengths = [], [], []
        for client, boundary in zip(client_ids, boundaries, strict=True):
            start, stop = self._bounds(client)
            chunks = []
            for domain, budget in zip(
                self._domain_indices, [50, self.source_budget], strict=True
            ):
                lo, hi = np.searchsorted(domain, [start, stop])
                selected = domain[lo:hi]
                if boundary is not None:
                    selected = selected[
                        : np.searchsorted(timestamps[selected], boundary, side="left")
                    ]
                chunks.append(selected[-budget:] if budget else selected[:0])
            selected = np.sort(np.concatenate(chunks))
            indices.append(selected)
            lengths.append(len(selected))
            positions.append(np.arange(len(selected), 0, -1, dtype=np.int64))
        selected = torch.from_numpy(
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
        )
        frame = pl.DataFrame(
            {
                "position": pl.Series(
                    np.concatenate(positions) if positions else [], dtype=pl.Int64
                ),
                "timestamp": pl.Series(self._timestamps[selected].numpy())
                .cast(pl.Datetime("ns"))
                .cast(self._schema["timestamp"]),
                **{
                    name: pl.Series(self._columns[name][selected].numpy()).cast(
                        self._schema[name]
                    )
                    for name in names
                },
            }
        )
        return frame, lengths

    def read(self, client_id, /, *, before=None):
        frame, _ = self.read_batch([client_id], [before])
        return frame.with_columns(
            pl.lit(client_id).cast(self._schema["client_id"]).alias("client_id")
        )


class AdditiveStore:
    def __init__(self, path, target_ids, source_budget):
        self.path = path
        self.target_ids = target_ids
        self.source_budget = source_budget

    def open_reader(self, max_events_per_sequence, /):
        if max_events_per_sequence != 550:
            raise ValueError(
                "Both conditions require the same 550-token model capacity"
            )
        return AdditiveReader(
            self.path, self.target_ids, source_budget=self.source_budget
        )
