from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DemoFeatureSpec:
    dense_scalar_dims: Dict[str, int]
    dense_seq_dims: Dict[str, int]
    sparse_scalar_vocab_sizes: Dict[str, int]
    sparse_seq_vocab_sizes: Dict[str, int]
    seq_len: int = 8


class DESCNDemoDataset(Dataset):
    def __init__(self, n_samples: int = 2048, seed: int = 42) -> None:
        super().__init__()
        self.spec = DemoFeatureSpec(
            dense_scalar_dims={"age": 1, "income": 1},
            dense_seq_dims={"behavior_dense": 2},
            sparse_scalar_vocab_sizes={"gender": 4, "city": 64},
            sparse_seq_vocab_sizes={"item_hist": 500},
            seq_len=8,
        )
        g = torch.Generator().manual_seed(seed)

        self.dense_scalar = {
            "age": torch.rand(n_samples, 1, generator=g),
            "income": torch.rand(n_samples, 1, generator=g),
        }

        self.dense_seq = {
            "behavior_dense": torch.rand(n_samples, self.spec.seq_len, 2, generator=g),
        }

        lengths = torch.randint(1, self.spec.seq_len + 1, (n_samples,), generator=g)
        self.dense_seq_mask = {
            "behavior_dense": (
                torch.arange(self.spec.seq_len).unsqueeze(0) < lengths.unsqueeze(1)
            ).float(),
        }

        self.sparse_scalar = {
            "gender": torch.randint(1, 4, (n_samples,), generator=g),
            "city": torch.randint(1, 64, (n_samples,), generator=g),
        }

        hist = torch.randint(1, 500, (n_samples, self.spec.seq_len), generator=g)
        hist_mask = torch.arange(self.spec.seq_len).unsqueeze(0) < lengths.unsqueeze(1)
        hist = hist * hist_mask
        self.sparse_seq = {"item_hist": hist}

        # synthetic targets with shared pattern and treatment variations
        age = self.dense_scalar["age"].squeeze(-1)
        income = self.dense_scalar["income"].squeeze(-1)
        behavior_mean = (
            self.dense_seq["behavior_dense"] * self.dense_seq_mask["behavior_dense"].unsqueeze(-1)
        ).sum(1) / self.dense_seq_mask["behavior_dense"].sum(1, keepdim=True).clamp_min(1.0)
        behavior_score = behavior_mean.sum(-1)
        gender = self.sparse_scalar["gender"].float() / 4.0
        city = self.sparse_scalar["city"].float() / 64.0
        item_signal = (self.sparse_seq["item_hist"].float() > 250).float().mean(-1)

        base = 0.8 * age + 1.2 * income + 0.6 * behavior_score + 0.3 * city + 0.4 * gender
        noise = 0.05 * torch.randn(n_samples, generator=g)

        ctrl_logit = base + 0.2 * item_signal + noise
        t1_logit = base + 0.5 * item_signal + 0.15 * age + noise
        t2_logit = base + 0.8 * item_signal - 0.1 * income + noise
        t3_logit = base + 0.3 * item_signal + 0.2 * behavior_score + noise

        self.targets = {
            "control": torch.bernoulli(torch.sigmoid(ctrl_logit)).unsqueeze(-1),
            "treatment_1": torch.bernoulli(torch.sigmoid(t1_logit)).unsqueeze(-1),
            "treatment_2": torch.bernoulli(torch.sigmoid(t2_logit)).unsqueeze(-1),
            "treatment_3": torch.bernoulli(torch.sigmoid(t3_logit)).unsqueeze(-1),
        }

        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "dense_scalar": {k: v[idx] for k, v in self.dense_scalar.items()},
            "dense_seq": {k: v[idx] for k, v in self.dense_seq.items()},
            "dense_seq_mask": {k: v[idx] for k, v in self.dense_seq_mask.items()},
            "sparse_scalar": {k: v[idx] for k, v in self.sparse_scalar.items()},
            "sparse_seq": {k: v[idx] for k, v in self.sparse_seq.items()},
            "targets": {k: v[idx] for k, v in self.targets.items()},
        }


def _collate_dict_tensor(batch_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch_list[0].keys()
    return {k: torch.stack([item[k] for item in batch_list], dim=0) for k in keys}


def descn_collate_fn(batch: list[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "dense_scalar": _collate_dict_tensor([b["dense_scalar"] for b in batch]),
        "dense_seq": _collate_dict_tensor([b["dense_seq"] for b in batch]),
        "dense_seq_mask": _collate_dict_tensor([b["dense_seq_mask"] for b in batch]),
        "sparse_scalar": _collate_dict_tensor([b["sparse_scalar"] for b in batch]),
        "sparse_seq": _collate_dict_tensor([b["sparse_seq"] for b in batch]),
        "targets": _collate_dict_tensor([b["targets"] for b in batch]),
    }


def build_demo_dataloader(batch_size: int = 64, n_samples: int = 2048) -> DataLoader:
    dataset = DESCNDemoDataset(n_samples=n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=descn_collate_fn)
