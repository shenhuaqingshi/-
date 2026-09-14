"""Crystal Graph Convolutional Neural Network for multi-task learning."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLReadOut(nn.Module):
    """Read-out function for multi-task prediction."""

    def __init__(self, atom_fea_len: int, h_fea_len: int, n_tasks: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(atom_fea_len, h_fea_len)
        self.task_heads = nn.ModuleList([nn.Linear(h_fea_len, 1) for _ in range(n_tasks)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, crystal_atom_fea: torch.Tensor) -> List[torch.Tensor]:
        hidden = F.relu(self.fc1(crystal_atom_fea))
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.task_heads]


class CGCNNConv(nn.Module):
    """Simplified CGCNN convolution that preserves feature dimensions."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int) -> None:
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.linear = nn.Linear(atom_fea_len + nbr_fea_len, atom_fea_len)

    def forward(self, atom_in_fea: torch.Tensor, nbr_idx: torch.Tensor, nbr_fea: torch.Tensor) -> torch.Tensor:
        if atom_in_fea.dim() == 2:
            atom_in_fea = atom_in_fea.unsqueeze(0)
            nbr_fea = nbr_fea.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if nbr_fea.dim() == 4:
            neighbor_summary = nbr_fea.mean(dim=2)
        else:
            neighbor_summary = nbr_fea

        combined = torch.cat([atom_in_fea, neighbor_summary], dim=-1)
        atom_out = F.relu(self.linear(combined))
        return atom_out.squeeze(0) if squeeze else atom_out


class CGCNN_MTL(nn.Module):
    """CGCNN backbone with one linear head per regression task."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_graph_conv: int = 3,
        h_fea_len: int = 128,
        n_tasks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [CGCNNConv(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_graph_conv)]
        )
        self.readout = MTLReadOut(
            atom_fea_len=atom_fea_len,
            h_fea_len=h_fea_len,
            n_tasks=n_tasks,
            dropout=dropout,
        )

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_fea: torch.Tensor,
        crystal_atom_idx: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        if atom_fea.dim() == 2:
            atom_fea = atom_fea.unsqueeze(0)
            nbr_idx = nbr_idx.unsqueeze(0)
            nbr_fea = nbr_fea.unsqueeze(0)

        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_idx, nbr_fea)

        if crystal_atom_idx:
            pooled = []
            flat = atom_fea.reshape(-1, atom_fea.size(-1))
            for start_idx, end_idx in crystal_atom_idx:
                pooled.append(flat[start_idx:end_idx].mean(dim=0, keepdim=True))
            crystal_fea = torch.cat(pooled, dim=0)
        else:
            crystal_fea = atom_fea.mean(dim=1)

        return self.readout(crystal_fea)
