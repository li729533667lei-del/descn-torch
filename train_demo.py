from __future__ import annotations

import torch
import torch.nn as nn

from descn_torch import DESCN, DESCNConfig, build_demo_dataloader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_demo_dataloader(batch_size=128, n_samples=4096)
    dataset = dataloader.dataset

    cfg = DESCNConfig(
        dense_scalar_dims=dataset.spec.dense_scalar_dims,
        dense_seq_dims=dataset.spec.dense_seq_dims,
        sparse_scalar_vocab_sizes=dataset.spec.sparse_scalar_vocab_sizes,
        sparse_seq_vocab_sizes=dataset.spec.sparse_seq_vocab_sizes,
        embedding_dim=16,
        cross_layers=3,
        deep_hidden_dims=(128, 64),
        head_hidden_dim=32,
        dropout=0.1,
    )

    model = DESCN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = {
                g: {k: v.to(device) for k, v in val.items()}
                for g, val in batch.items()
            }
            preds = model(batch)
            loss = sum(bce(preds[name], batch["targets"][name]) for name in preds.keys())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:02d} | loss={total_loss / len(dataloader):.4f}")

    model.eval()
    with torch.no_grad():
        sample = next(iter(dataloader))
        sample = {g: {k: v.to(device) for k, v in val.items()} for g, val in sample.items()}
        outputs = model(sample)
        print("\nPrediction sample (first 5 rows):")
        for head, value in outputs.items():
            print(f"{head}: {value[:5, 0].cpu().numpy()}")


if __name__ == "__main__":
    main()
