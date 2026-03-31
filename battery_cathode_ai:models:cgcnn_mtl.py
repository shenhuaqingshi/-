"""
Crystal Graph Convolutional Neural Network for Multi-Task Learning (MTL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MTLReadOut(nn.Module):
    """Read-out function for multi-task prediction."""
    def __init__(self, atom_fea_len, h_fea_len, n_tasks):
        super(MTLReadOut, self).__init__()
        self.fc1 = nn.Linear(atom_fea_len, h_fea_len)
        # One output head per task
        self.task_heads = nn.ModuleList([
            nn.Linear(h_fea_len, 1) for _ in range(n_tasks)
        ])
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, crystal_atom_fea):
        # crystal_atom_fea: [batch_size, atom_fea_len]
        h = F.relu(self.fc1(crystal_atom_fea))
        h = self.dropout(h)
        outputs = []
        for head in self.task_heads:
            out = head(h)
            outputs.append(out)
        return outputs # List of [batch_size, 1] tensors


class CGCNN_MTL(nn.Module):
    """
    Crystal Graph Convolutional Neural Network with Multi-Task Output.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_graph_conv=3, h_fea_len=128, n_tasks=3):
        super(CGCNN_MTL, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        self.convs = nn.ModuleList([
            CGCNNConv(is_v2=False, atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_graph_conv)
        ])
        
        self.readout = MTLReadOut(atom_fea_len=atom_fea_len, h_fea_len=h_fea_len, n_tasks=n_tasks)

    def forward(self, atom_fea, nbr_idx, nbr_fea, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_idx, nbr_fea)
        
        # Pooling: average pooling over all atoms in the crystal
        crys_fea = []
        for i, (start_idx, end_idx) in enumerate(crystal_atom_idx):
            crys_fea.append(torch.mean(atom_fea[start_idx:end_idx], dim=0, keepdim=True))
        crys_fea = torch.cat(crys_fea, dim=0) # [batch_size, atom_fea_len]
        
        task_outputs = self.readout(crys_fea)
        return task_outputs # List of predictions for each task

# Placeholder for CGCNNConv layer
class CGCNNConv(nn.Module):
    """
    Placeholder for the actual CGCNN convolution layer.
    In a real implementation, this would contain the complex logic for message passing.
    For this simulation, it's a simple placeholder that maintains dimensions.
    """
    def __init__(self, is_v2, atom_fea_len, nbr_fea_len):
        super(CGCNNConv, self).__init__()
        self.is_v2 = is_v2
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # Simplified linear transformation to simulate conv operation
        self.linear = nn.Linear(atom_fea_len + nbr_fea_len, atom_fea_len)

    def forward(self, atom_in_fea, nbr_idx, nbr_fea):
        # This is a placeholder logic.
        # Real CGCNN would aggregate neighbor features according to the graph structure.
        batch_size, num_atoms, _ = atom_in_fea.size()
        expanded_nbr_fea = nbr_fea.view(batch_size, num_atoms, -1, self.nbr_fea_len).mean(dim=2) # Average neighbors
        combined_fea = torch.cat([atom_in_fea, expanded_nbr_fea], dim=-1)
        atom_out_fea = F.relu(self.linear(combined_fea))
        return atom_out_fea