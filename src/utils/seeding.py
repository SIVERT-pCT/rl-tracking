
from abc import ABC

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from src.utils.graph import TGraph, get_act_features, norm, query_neighborhood


@torch.jit.script
def k_nearest(neigh_pos, node_pos, k: int):
    return torch.topk(norm(node_pos[:,None,:] - neigh_pos[:,:,:], dim=-1), 
                      k=k, dim=-1, largest=False)[1]
    
@torch.jit.script
def dist(t1: torch.Tensor, t2: torch.Tensor):
    return torch.sqrt(torch.sum((t1[:,None,:] - t2[None,:,:] ) ** 2, dim=-1))


@torch.jit.script
def dist_stacked(t1: torch.Tensor, t2: torch.Tensor):
    return torch.sqrt(torch.sum((t1[:,None,:,:] - t2[:,:,None,:] ) ** 2, dim=-1))


class TrackSeedingBase(ABC):
    def __init__(self, graph: TGraph, k: int) -> None:
        super().__init__()
        self._graph = graph
        self._k = k
        
    def get_actions(self, obs: torch.Tensor, seed: torch.BoolTensor, 
                    occupied: torch.Tensor = None, done: torch.Tensor = None) -> torch.Tensor:
        pass
    
    @property
    def k(self) -> int:
        return self._k
    
    @property
    def graph(self) -> TGraph:
        return self._graph
    

class GroundTruthSeeding(TrackSeedingBase):
    def __init__(self, graph: TGraph) -> None:
        """Initial track seeding algorithm using the ground 
        truth in order to determine the next action for an 
        observation (1 observation -> 1 action).

        @param env: Environment describing the dynamics of the system
        """
        super().__init__(graph, k=1)
        
    def _select_action(self, obs: torch.Tensor, neighborhood: torch.Tensor) -> torch.Tensor:
        """Selects the corresponding action for a single observation
        based on the eventID of the neighbor nodes. Returns -1 if no
        action found.

        @param obs: Tensor containing a single observation
        @param neighborhood: Tensor determining the neighborhood of 
                             the observation. 

        @returns: Corresponding action for observation [long/-1]
        """
        act = torch.where(self.graph.y[obs] == self.graph.y[neighborhood])[0]
        
        # Return -1 action if no neighbors exist for seeding.
        return act[0] if len(act) > 0 \
                   else torch.empty_like(obs, dtype=torch.long).fill_(-1)
           
    def get_actions(self, obs: torch.Tensor, seed: torch.BoolTensor, 
                    occupied: torch.Tensor = None, done: torch.Tensor = None) -> torch.Tensor:
        """Returns the corresponding actions for a set of observations 
        according to the eventID of the track.

        @param obs: Tensor containing multiple observations.
        @param seed: Masking tensor which is true for all observations 
                     that should be used for track seeding.

        @returns: Corresponding actions for obs[seed].
        """
        obs = obs[seed]
        query_neighborhood(self.graph, obs)

        neighborhoods = query_neighborhood(self.graph, obs)
        act = [self._select_action(obs[i], neighborhoods[i]) \
                for i in range(len(neighborhoods))]
        
        if len(torch.stack(act)) > len(obs):
             raise Exception()
        
        return torch.stack(act) if len(act) > 0 \
                              else torch.empty((0, ))