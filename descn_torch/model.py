from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class DESCNConfig:
    dense_scalar_dims: Dict[str, int]
    dense_seq_dims: Dict[str, int]
    sparse_scalar_vocab_sizes: Dict[str, int]
    sparse_seq_vocab_sizes: Dict[str, int]
    embedding_dim: int = 16
    cross_layers: int = 3
    deep_hidden_dims: tuple[int, ...] = (128, 64)
    head_hidden_dim: int = 32
    dropout: float = 0.1


class CrossNetwork(nn.Module):
    """Classic DCN cross network."""

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.kernels, self.bias):
            xw = x @ w  # [B, 1]
            x = x0 * xw + b + x
        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DESCN(nn.Module):
    """Deep Entire Space Cross Networks with 4 heads (1 control + 3 treatment)."""

    def __init__(self, config: DESCNConfig) -> None:
        super().__init__()
        self.config = config

        self.sparse_scalar_emb = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
                for name, vocab_size in config.sparse_scalar_vocab_sizes.items()
            }
        )
        self.sparse_seq_emb = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
                for name, vocab_size in config.sparse_seq_vocab_sizes.items()
            }
        )

        total_dense_scalar_dim = sum(config.dense_scalar_dims.values())
        total_dense_seq_dim = sum(config.dense_seq_dims.values())
        total_sparse_scalar_dim = len(config.sparse_scalar_vocab_sizes) * config.embedding_dim
        total_sparse_seq_dim = len(config.sparse_seq_vocab_sizes) * config.embedding_dim

        self.input_dim = (
            total_dense_scalar_dim
            + total_dense_seq_dim
            + total_sparse_scalar_dim
            + total_sparse_seq_dim
        )

        self.cross = CrossNetwork(self.input_dim, config.cross_layers)
        self.deep = MLP(self.input_dim, config.deep_hidden_dims, config.dropout)

        fusion_dim = self.input_dim + self.deep.output_dim
        self.head_shared = nn.Sequential(
            nn.Linear(fusion_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.heads = nn.ModuleDict(
            {
                "control": nn.Linear(config.head_hidden_dim, 1),
                "treatment_1": nn.Linear(config.head_hidden_dim, 1),
                "treatment_2": nn.Linear(config.head_hidden_dim, 1),
                "treatment_3": nn.Linear(config.head_hidden_dim, 1),
            }
        )

    def _pool_dense_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L]
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _pool_sparse_sequence(self, emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # emb: [B, L, E], ids: [B, L], where 0 is padding
        mask = (ids != 0).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _build_input(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        for name in self.config.dense_scalar_dims:
            parts.append(batch["dense_scalar"][name].float())

        for name in self.config.dense_seq_dims:
            seq = batch["dense_seq"][name].float()
            mask = batch["dense_seq_mask"][name].float()
            parts.append(self._pool_dense_sequence(seq, mask))

        for name, emb in self.sparse_scalar_emb.items():
            ids = batch["sparse_scalar"][name].long()
            parts.append(emb(ids))

        for name, emb in self.sparse_seq_emb.items():
            ids = batch["sparse_seq"][name].long()
            seq_emb = emb(ids)
            parts.append(self._pool_sparse_sequence(seq_emb, ids))

        return torch.cat(parts, dim=-1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        x = self._build_input(batch)
        cross_out = self.cross(x)
        deep_out = self.deep(x)
        fused = torch.cat([cross_out, deep_out], dim=-1)
        shared = self.head_shared(fused)

        return {name: torch.sigmoid(head(shared)) for name, head in self.heads.items()}
