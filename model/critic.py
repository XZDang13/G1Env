import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, CriticHead

class ValueNet(nn.Module):
    def __init__(self, feature_dim:int, hidden_dims:list[int]):
        super().__init__()
        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)

        self.value = CriticHead(dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        step = self.value(x)

        return step