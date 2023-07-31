from dataclasses import dataclass
from math import pi as PI
from pickle import FALSE

import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from src.utils.pandas import edep_to_cluster_size
from src.utils.pct_restrict_detector_boundaries import position_to_pixel


class SphericalInverse(BaseTransform):
    """Uses pytorch geometric implementation (https://pytorch-geometric.readthedocs.io/
    en/latest/_modules/torch_geometric/transforms/spherical.html) for reference.
    """
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data):
        (from_index, to_index), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart =  pos[from_index] - pos[to_index]

        rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        phi = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)

        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * PI)
            phi = phi / PI

        spherical = torch.cat([rho, theta, phi], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spherical.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spherical

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
        
        
class InvariantNorm(BaseTransform):
    def __init__(self, pixel: bool = True, cluster: bool = True):
        super().__init__()
        self.edep_index = 0
        self.pos_x_index = 1
        self.pos_y_index = 2
        assert pixel and cluster, \
            "Currently only supported for cluster and pixel definitions"
        
    def __normalize_edep(self, x: torch.Tensor):
        x[:,self.edep_index] /= edep_to_cluster_size(0.06)
        return x
    
    def __normalize_pos(self, x: torch.Tensor, spot_x: int, spot_y: int):
        spot_x_px = 0 if spot_x is None else position_to_pixel(spot_x, x_dim=True)
        spot_y_px = 0 if spot_x is None else position_to_pixel(spot_y, x_dim=False)
        
        x[:,self.pos_x_index] = (x[:,self.pos_x_index] - spot_x_px)/1000
        x[:,self.pos_y_index] = (x[:,self.pos_y_index] - spot_y_px)/1000
        return x
    
    def __call__(self, data: Data):
        assert hasattr(data, "spot_x"), \
            "Invariant norm is only available for TGraph definitions"
        assert hasattr(data, "spot_y"), \
            "Invariant norm is only available for TGraph definitions"
            
        data.x = self.__normalize_edep(data.x)
        data.x = self.__normalize_pos(data.x, data.spot_x, data.spot_y)
        return data