"""Minimal training loop for the multi-task CGCNN."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from .data_loader import get_data_loaders
from .loss import MultiTaskMSELoss
from .predict import build_model


def train(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_loader, val_loader = get_data_loaders(config)
    model = build_model(config)
    criterion = MultiTaskMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    epochs = int(config["training"].get("epochs", 5))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for atom_fea, nbr_idx, nbr_fea, targets, *_ in train_loader:
            optimizer.zero_grad()
            preds = model(atom_fea, nbr_idx, nbr_fea, None)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for atom_fea, nbr_idx, nbr_fea, targets, *_ in val_loader:
                preds = model(atom_fea, nbr_idx, nbr_fea, None)
                val_running += float(criterion(preds, targets).item())
                val_batches += 1
        val_loss = val_running / max(val_batches, 1)
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    ckpt = Path(config["paths"].get("checkpoint", "checkpoints/model.pth"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"saved checkpoint to {ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-task CGCNN")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
