import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import  GATv2Conv
from torch_geometric.data import Data


class GatConvEmbedding(nn.Module):
    def __init__(self, node_features, embedding_dim):
        super().__init__()
        self.e1 = GATv2Conv(node_features, embedding_dim, flow='target_to_source')
        self.e2 = GATv2Conv(embedding_dim, embedding_dim, flow='target_to_source')
        self.e3 = GATv2Conv(embedding_dim, embedding_dim, flow='target_to_source')
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.e1(x,  edge_index))
        h2 = torch.relu(self.e2(h1, edge_index))
        h3 = torch.relu(self.e3(h2, edge_index))
        return h3