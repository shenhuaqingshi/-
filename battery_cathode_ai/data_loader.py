"""Data loaders for graph and descriptor modes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .graph_builder import CIFData

TASK_KEYS = (
    "practical_capacity_mAh_g",
    "voltage_vs_Li_Li_mV",
    "cycle_life_cycles",
)


def _to_scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)) and value:
        return float(sum(value)) / float(len(value))
    if value is None:
        return 0.0
    return float(value)


class BenchmarkSetLoader:
    """Load the benchmark JSON file."""

    @staticmethod
    def load(path: str | Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "benchmark_materials" in data:
            return data["benchmark_materials"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unrecognized benchmark JSON schema in {path}")


class GraphDataset(Dataset):
    """PyTorch dataset that pairs crystal graphs with multi-task labels."""

    def __init__(
        self,
        cif_paths: Sequence[str],
        benchmark_data_map: Dict[str, Dict[str, Any]],
        task_keys: Sequence[str] = TASK_KEYS,
        graph_cfg: Dict[str, Any] | None = None,
    ) -> None:
        self.cif_paths = list(cif_paths)
        self.benchmark_data_map = benchmark_data_map
        self.task_keys = list(task_keys)
        graph_cfg = graph_cfg or {}
        self.cif_data = CIFData(
            self.cif_paths,
            atom_fea_len=graph_cfg.get("atom_fea_len", 92),
            nbr_fea_len=graph_cfg.get("nbr_fea_len", 41),
            max_num_nbr=graph_cfg.get("max_num_nbr", 6),
        )

    def __len__(self) -> int:
        return len(self.cif_paths)

    def __getitem__(self, idx: int):
        atom_fea, nbr_idx, nbr_fea, crystal_atom_idx = self.cif_data[idx]
        stem = Path(self.cif_paths[idx]).stem.split("__")[0]
        record = self.benchmark_data_map.get(stem, {})
        properties = record.get("properties", {}) if isinstance(record, dict) else {}
        target_values = [_to_scalar(properties.get(key, 0.0)) for key in self.task_keys]
        return (
            torch.tensor(atom_fea, dtype=torch.float32),
            torch.tensor(nbr_idx, dtype=torch.long),
            torch.tensor(nbr_fea, dtype=torch.float32),
            torch.tensor(target_values, dtype=torch.float32),
            crystal_atom_idx,
        )


def get_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from config."""
    benchmark_data = BenchmarkSetLoader.load(config["paths"]["benchmark_json"])
    benchmark_map = {item["material_id"]: item for item in benchmark_data}
    cif_dir = Path(config["paths"].get("cif_dir", "cif_files"))
    cif_paths = []
    for material in benchmark_data:
        mat_id = material["material_id"]
        real = list(cif_dir.glob(f"{mat_id}*.cif")) if cif_dir.exists() else []
        cif_paths.append(str(real[0]) if real else f"{cif_dir}/{mat_id}__dummy.cif")

    dataset = GraphDataset(
        cif_paths,
        benchmark_map,
        task_keys=config.get("tasks", TASK_KEYS),
        graph_cfg=config.get("graph"),
    )

    val_split = float(config["training"].get("validation_split", 0.2))
    val_size = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size, val_size = len(dataset), 0

    generator = torch.Generator().manual_seed(int(config["training"].get("seed", 42)))
    if val_size == 0:
        train_dataset, val_dataset = dataset, dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    batch_size = int(config["training"].get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
