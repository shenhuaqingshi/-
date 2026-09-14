"""Multi-task regression loss with optional per-task weights."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


class MultiTaskMSELoss(nn.Module):
    """Average MSE across task heads.

    Accepts either a list of ``[B, 1]`` tensors or a stacked ``[B, T]`` tensor.
    """

    def __init__(self, task_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        if task_weights is None:
            self.task_weights = None
        else:
            self.register_buffer("task_weights", torch.tensor(task_weights, dtype=torch.float32))

    def forward(self, preds: Iterable[torch.Tensor] | torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(preds, (list, tuple)):
            pred = torch.cat(list(preds), dim=1)
        else:
            pred = preds
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        n_tasks = pred.size(1)
        losses = []
        for i in range(n_tasks):
            losses.append(self.mse(pred[:, i], targets[:, i]))
        stacked = torch.stack(losses)
        if self.task_weights is None:
            return stacked.mean()
        weights = self.task_weights.to(stacked.device)
        weights = weights / weights.sum()
        return (stacked * weights).sum()
