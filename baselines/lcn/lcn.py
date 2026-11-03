# ruff: noqa: INP001

import math

import torch
from torch import nn
from torch.nn import functional


def gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(
        1, indices.unsqueeze(-1).expand(*indices.shape, values.shape[-1])
    )


def recent(
    values: torch.Tensor, mask: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = (
        torch.arange(mask.shape[1], device=mask.device)
        .expand_as(mask)
        .masked_fill(~mask, -1)
    )
    indices = ranks.topk(min(length, mask.shape[1]), dim=1).indices
    return gather(values, indices), mask.gather(1, indices)


def weighted(
    values: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return (values * scores.masked_fill(~mask, 0).unsqueeze(-1)).sum(1)


class LCN(nn.Module):
    def __init__(
        self,
        cardinalities: dict[str, int],
        context_fields: list[str],
        identity_fields: list[str],
        target_events: list[int],
        num_targets: int,
        *,
        cross_domain: bool = True,
        fallback_to_target: bool = False,
        dim: int = 64,
        top_k1: int = 200,
        top_k2: int = 50,
        short_length: int = 50,
        temperature: float = 0.1,
        negative_pairs: int = 512,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality + 2, dim, padding_idx=0)
                for name, cardinality in cardinalities.items()
            },
        )
        self.cardinalities = cardinalities
        self.context_fields = context_fields
        self.identity_fields = identity_fields
        self.cross_domain = cross_domain
        self.fallback_to_target = fallback_to_target
        self.top_k1, self.top_k2, self.short_length = top_k1, top_k2, short_length
        self.temperature, self.negative_pairs = temperature, negative_pairs
        self.register_buffer("target_events", torch.tensor(target_events))
        self.identity = nn.Linear(len(identity_fields) * dim, dim, bias=False)
        self.msa_q = nn.Linear(dim, dim, bias=False)
        self.msa_k = nn.Linear(dim * 2, dim, bias=False)
        self.msa_v = nn.Linear(dim * 2, dim, bias=False)
        self.fsa_q = nn.Linear(dim, dim, bias=False)
        self.fsa_k = nn.Linear(dim * 2, dim, bias=False)
        self.fsa_v = nn.Linear(dim * 2, dim, bias=False)
        self.fsa_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim)
        )
        self.short_q = nn.Linear(dim, dim, bias=False)
        self.short_k = nn.Linear(dim * 2, dim, bias=False)
        self.short_v = nn.Linear(dim * 2, dim, bias=False)
        self.head = nn.ModuleList(
            nn.Sequential(
                nn.Linear((len(context_fields) + 5) * dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(num_targets)
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(layer.weight)
                if isinstance(layer, nn.Linear) and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                if isinstance(layer, nn.Embedding):
                    with torch.no_grad():
                        layer.weight[0].zero_()

    def identity_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        fields = []
        for name in self.identity_fields:
            ids = features[name].long()
            known = (ids > 0) & (ids <= self.cardinalities[name])
            fields.append(self.embeddings[name](ids) * known.unsqueeze(-1))
        return self.identity(torch.cat(fields, dim=-1))

    def short_attention(
        self, query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scores = (self.short_k(values) * self.short_q(query).unsqueeze(1)).sum(
            -1
        ) / math.sqrt(query.shape[-1])
        weights = scores.float().masked_fill(~mask, -1e9).softmax(-1) * mask
        return (self.short_v(values) * weights.unsqueeze(-1)).sum(1)

    def pyramid(
        self,
        query: torch.Tensor,
        identity: torch.Tensor,
        contextual: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = (identity * query.unsqueeze(1)).sum(-1)
        csa = weighted(identity, scores, mask)
        first = (
            scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            .topk(min(self.top_k1, scores.shape[1]), dim=1)
            .indices
        )
        identity, contextual, mask = (
            gather(identity, first),
            gather(contextual, first),
            mask.gather(1, first),
        )
        values = torch.cat([identity, contextual], dim=-1)
        scores = (self.msa_k(values) * self.msa_q(query).unsqueeze(1)).sum(
            -1
        ) / math.sqrt(query.shape[-1])
        msa = weighted(self.msa_v(values), scores, mask)
        second = (
            scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            .topk(min(self.top_k2, scores.shape[1]), dim=1)
            .indices
        )
        values, mask = gather(values, second), mask.gather(1, second)
        heads, width = 4, query.shape[-1] // 4
        q = self.fsa_q(query).reshape(-1, heads, width)
        k = self.fsa_k(values).reshape(values.shape[0], values.shape[1], heads, width)
        v = self.fsa_v(values).reshape_as(k)
        scores = torch.einsum("bhd,bkhd->bhk", q, k) / math.sqrt(width)
        weights = scores.float().masked_fill(~mask.unsqueeze(1), -1e9).softmax(
            -1
        ) * mask.unsqueeze(1)
        fsa = torch.einsum("bhk,bkhd->bhd", weights.to(v.dtype), v).flatten(1)
        fsa = self.fsa_ffn(fsa) * mask.any(1).unsqueeze(-1)
        return csa, msa, fsa

    def contrastive(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        eligible: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        indices = eligible.nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            return left.sum() * 0
        shuffled = indices[torch.randperm(indices.numel(), device=indices.device)]
        valid = users[indices] != users[shuffled]
        first, second = (
            indices[valid][: self.negative_pairs],
            shuffled[valid][: self.negative_pairs],
        )
        if first.numel() == 0:
            return left.sum() * 0
        left, right = (
            functional.normalize(left.float(), dim=-1),
            functional.normalize(right.float(), dim=-1),
        )
        positive = (left[indices] * right[indices]).sum(-1) / self.temperature
        negative = (left[first] * right[second]).sum(-1) / self.temperature
        return negative.logsumexp(0) - positive.mean()

    def crp(
        self,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        def pair(
            values: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            priorities = torch.rand(mask.shape, device=mask.device).masked_fill(
                ~mask, 2
            )
            indices = priorities.argsort(1)[:, :2]
            if indices.shape[1] == 1:
                indices = indices.expand(-1, 2)
            return gather(values, indices), mask.sum(1)

        t, nt = pair(target, target_mask)
        s, ns = pair(source, source_mask)
        return (
            self.contrastive(t[:, 0], t[:, 1], nt >= 2, users)
            + self.contrastive(s[:, 0], s[:, 1], ns >= 2, users)
            + self.contrastive(t[:, 0], s[:, 0], (nt >= 1) & (ns >= 1), users)
        )

    def forward(
        self,
        events: dict[str, torch.Tensor],
        positions: torch.Tensor,
        context: dict[str, torch.Tensor],
        users: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = positions > 0
        target_mask = mask & torch.isin(events["event"].long(), self.target_events)
        source_mask = (
            mask & ~target_mask if self.cross_domain else torch.zeros_like(mask)
        )
        identity = self.identity_embedding(events)
        contextual = self.embeddings["event"](events["event"].long())
        values = torch.cat([identity, contextual], dim=-1)
        query = self.identity_embedding(context)
        target, tm = recent(values, target_mask, self.short_length)
        source, sm = recent(values, source_mask, self.short_length)
        target_interest = self.short_attention(query, target, tm)
        source_interest = self.short_attention(query, source, sm)
        long_mask = source_mask if self.cross_domain else target_mask
        if self.cross_domain and self.fallback_to_target:
            long_mask = torch.where(
                source_mask.any(dim=1, keepdim=True), source_mask, target_mask
            )
        interests = self.pyramid(
            query,
            identity,
            contextual,
            long_mask,
        )
        profile = torch.cat(
            [
                self.embeddings[name](context[name].long())
                for name in self.context_fields
            ],
            dim=-1,
        )
        representation = torch.cat(
            [profile, target_interest, source_interest, *interests], dim=-1
        )
        logits = torch.cat([head(representation) for head in self.head], dim=-1)
        aux = (
            self.crp(
                target[..., : query.shape[-1]],
                tm,
                source[..., : query.shape[-1]],
                sm,
                users,
            )
            if self.training and self.cross_domain
            else logits.sum() * 0
        )
        return logits, aux
