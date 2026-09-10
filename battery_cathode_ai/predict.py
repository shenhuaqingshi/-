"""Prediction script with Monte Carlo Dropout uncertainty estimates."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import torch
import yaml

from .models.cgcnn_mtl import CGCNN_MTL


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(config: dict) -> CGCNN_MTL:
    return CGCNN_MTL(
        orig_atom_fea_len=config.get("graph", {}).get("atom_fea_len", 92),
        nbr_fea_len=config["graph"]["nbr_fea_len"],
        atom_fea_len=config["model"]["atom_fea_len"],
        n_graph_conv=config["model"]["n_graph_conv"],
        h_fea_len=config["model"]["h_fea_len"],
        n_tasks=config["model"]["n_tasks"],
        dropout=config["model"].get("dropout", 0.1),
    )


def load_model_and_predict(
    config_path: str | Path = "config.yaml",
    checkpoint_path: str | Path | None = None,
) -> Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, torch.Tensor]]:
    config = _load_config(config_path)
    model = build_model(config)

    ckpt = Path(checkpoint_path or config["paths"].get("checkpoint", "checkpoints/model.pth"))
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)

    n_iter = int(config.get("uq", {}).get("mc_dropout_iterations", 20))

    def predict_with_uq(sample_data):
        model.train()  # keep dropout active for MC sampling
        draws = []
        with torch.no_grad():
            for _ in range(n_iter):
                atom_fea, nbr_idx, nbr_fea = sample_data[:3]
                if atom_fea.dim() == 2:
                    atom_fea = atom_fea.unsqueeze(0)
                    nbr_idx = nbr_idx.unsqueeze(0)
                    nbr_fea = nbr_fea.unsqueeze(0)
                preds = model(atom_fea, nbr_idx, nbr_fea, None)
                draws.append(torch.cat(preds, dim=1))
        stacked = torch.stack(draws, dim=0)
        return stacked.mean(dim=0).squeeze(0), stacked.std(dim=0).squeeze(0)

    return predict_with_uq


def main() -> None:
    predictor = load_model_and_predict("config.yaml")
    dummy = (
        torch.rand(10, 92),
        torch.randint(0, 10, (10, 6)),
        torch.rand(10, 6, 41),
    )
    mean, std = predictor(dummy)
    print("Model loaded. Dummy prediction mean:", mean.tolist())
    print("Dummy prediction std:", std.tolist())


if __name__ == "__main__":
    main()
