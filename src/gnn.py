import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import PyTorch Geometric conv layers; if unavailable, we'll fall back
try:
    from torch_geometric.nn import GCNConv
    _PYG_AVAILABLE = True
except Exception:
    GCNConv = None
    _PYG_AVAILABLE = False

class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, p):
        super().__init__()
        self.p = p
        self.out_feats = out_feats
        self.hidden = hidden
        # If PyG is available, use GCNConv layers; otherwise use simple Linear layers with adjacency multiplication
        if _PYG_AVAILABLE:
            self.conv1 = GCNConv(in_feats, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        else:
            self.lin1 = nn.Linear(in_feats, hidden)
            self.lin2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, out_feats * p)

    def forward(self, x, adj):
        # x: [n, in_feats], adj: [n, n] or sparse adjacency; returns [out_feats, p]
        if _PYG_AVAILABLE and isinstance(adj, torch.Tensor):
            # build edge_index from adjacency matrix
            if adj.dim() == 2:
                edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
            else:
                # assume adj already edge_index-like
                edge_index = adj
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
        else:
            # adjacency-based message passing (works without PyG)
            if isinstance(adj, torch.Tensor):
                h = torch.matmul(adj, x)
            else:
                # if adj is numpy, convert
                adj = torch.tensor(adj, dtype=x.dtype, device=x.device)
                h = torch.matmul(adj, x)
            h = F.relu(self.lin1(h))
            if isinstance(adj, torch.Tensor):
                h = torch.matmul(adj, h)
            else:
                adj = torch.tensor(adj, dtype=x.dtype, device=x.device)
                h = torch.matmul(adj, h)
            h = F.relu(self.lin2(h))
        g = h.mean(dim=0, keepdim=True)  # global mean pool -> [1, hidden]
        out = self.readout(g).view(self.out_feats, self.p)  # [out_feats, p]
        return out

# Expose flag for other modules
PYG_AVAILABLE = _PYG_AVAILABLE
