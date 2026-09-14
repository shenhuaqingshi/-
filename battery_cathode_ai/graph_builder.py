"""CIF to crystal-graph builder.

This module currently ships a deterministic mock implementation so the rest of
the pipeline can run without pymatgen / real CIF files. Replace CIFData with a
real parser when structure files are available.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np


class CIFData:
    """Build (or mock) per-crystal graph tensors from CIF paths."""

    def __init__(
        self,
        cif_paths: Sequence[str],
        atom_fea_len: int = 92,
        nbr_fea_len: int = 41,
        max_num_nbr: int = 6,
        num_atoms: int = 10,
    ) -> None:
        self.cif_paths = list(cif_paths)
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_num_nbr = max_num_nbr
        self.num_atoms = num_atoms

    def __len__(self) -> int:
        return len(self.cif_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        seed = int(hashlib.md5(self.cif_paths[idx].encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        atom_fea = rng.random((self.num_atoms, self.atom_fea_len), dtype=np.float32)
        nbr_idx = rng.integers(0, self.num_atoms, size=(self.num_atoms, self.max_num_nbr))
        nbr_fea = rng.random((self.num_atoms, self.max_num_nbr, self.nbr_fea_len), dtype=np.float32)
        crystal_atom_idx = [(0, self.num_atoms)]
        return atom_fea, nbr_idx, nbr_fea, crystal_atom_idx
