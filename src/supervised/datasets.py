
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ..reinforcement.environment import (ParallelEnvironment,
                                         ParallelEnvironmentDataset)
from ..utils.eval import true_primary_tracks_from_graph
from ..utils.graph import query_neighborhood


class TrackDataset:
    def __init__(self, env_dataset: ParallelEnvironmentDataset) -> None:
        """Dataset containing tracking transitions to be learned by the
        neural network for a supervised learning approach.
        
        @param env_dataset: ParallelEnvironment information containing the
                            track information (requires ground truth (y) to 
                            be defined).
        """
        super().__init__()
        self.env_dataset = env_dataset
        
    def sample_batch(self, batch_size: int = 128) -> Tuple:
        """Samples random batch of track transitions from a randomly
        sampled environment (readout frame). 

        @param batch_size: Batch size, defaults to 128
        
        @return: Batch of track data (containing obs, old_obs and act).
        """
        env = self.env_dataset.sample()
        
        tracks = true_primary_tracks_from_graph(env)
        tracks = [torch.tensor(t, device=env.graph.x.device) for t in tracks]
        track_segments = torch.cat([torch.stack([t[:-2], t[1:-1], t[2:]]).T for t in tracks])

        target_obs, obs, old_obs = track_segments.T 
        n = query_neighborhood(env.graph, obs, to_list=False)

        mask = (n == target_obs.unsqueeze(1)).any(dim=1)
        _, act = torch.where(n == target_obs.unsqueeze(1))
        
        w = self._calculate_weights(env, obs[mask])
        p = self._apply_weights(env, obs[mask], w)
        sampler = WeightedRandomSampler(weights=p, num_samples=1024, replacement=True)
        indices = torch.ones_like(obs[mask]).cumsum(dim=0) - 1
        loader = DataLoader(dataset=indices, batch_size=batch_size, sampler=sampler)
        minibatch_indices = next(iter(loader))
    
        minibatch = {"obs": obs[mask][minibatch_indices],
                     "old_obs": old_obs[mask][minibatch_indices],
                     "act": act[minibatch_indices]}
        
        return env, minibatch
    
    
    def _apply_weights(self, env: ParallelEnvironment, obs: torch.Tensor, 
                       weights: Tuple[float, float, float]) -> torch.Tensor:
        
        w0, w1, w2 = weights
        nit = env.graph.next_is_tracker[obs]
        it = env.graph.is_tracker[obs]
        
        w_tensor = torch.ones_like(obs).float()
        w_tensor[nit &  it] *= w0
        w_tensor[nit & ~it] *= w1
        w_tensor[~nit] *= w2
        return w_tensor
    
    def _calculate_weights(self, env: ParallelEnvironment, obs: torch.Tensor) -> Tuple[float, float, float]:
        nit = env.graph.next_is_tracker[obs]
        it = env.graph.is_tracker[obs]

        num_1st = (nit & it).sum()
        num_2nd = (nit & ~it).sum()
        num_det = (~nit).sum()
        total = len(obs)

        return ((total - num_1st)/total, 
                (total - num_2nd)/total, 
                (total - num_det)/total)