import torch
import torch.nn as nn
import torch.nn.functional as F

from RLAlg.nn.layers import make_mlp_layers, GaussianHead

class ActorLearnNet(nn.Module):
    def __init__(self, feature_dim:int, action_dim:int, hidden_dims:list[int], max_action:float=1):
        super().__init__()

        self.layers, dim = make_mlp_layers(feature_dim, hidden_dims, F.silu, True)
        self.max_action = max_action
        self.policy = GaussianHead(dim, action_dim, max_action=max_action)

    def forward(self, x:torch.Tensor, action:torch.Tensor|None=None) -> tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        x = self.layers(x)

        step = self.policy(x, action)

        return step