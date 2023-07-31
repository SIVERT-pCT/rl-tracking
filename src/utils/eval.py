from itertools import compress
from math import pi as PI

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.reinforcement.environment import (ParallelEnvironment,
                                           ParallelEnvironmentDataset)
from src.reinforcement.episodes import ParallelEpisodes
from src.reinforcement.sampling import sample_parallel_trajectories
from src.utils.pandas import preprocess


def is_spot_scanning(env: ParallelEnvironment) -> bool:
    return env.graph.spot_x is not None and env.graph.spot_y is not None

def get_metrics_for_dataset(dataset: ParallelEnvironmentDataset, actor_critic: nn.Module, embedding_net: nn.Module, events: int):
    results = []
    
    for env in dataset:
        episodes = sample_parallel_trajectories(env, actor_critic, events, embedding_net=embedding_net, 
                                                embedding_type="act", stochastic=False, evaluate=True)
    
        _, mask = reject_tracks(env, episodes)
        filtered_tracks = list(compress(episodes.trajectories, mask.tolist()))
        true_tracks = true_primary_tracks_from_graph(env)
        
        rej = (len(episodes.trajectories) - len(filtered_tracks))/ len(episodes.trajectories)
        pur = purity(env, filtered_tracks)
        eff = efficiency(env, true_tracks, filtered_tracks)
        results.append((env.graph.spot_x, env.graph.spot_y, pur, eff, rej))
        
    results = pd.DataFrame(results)
    results.columns = ["spotX", "spotY", "pur", "eff", "rej"]
    
    return results


def reject_tracks(env: ParallelEnvironment, episodes: ParallelEpisodes, edep_threshold: float = 0.0625,
                  theta_threshold_d: float = 0.271, theta_threshold_t: float = 0.271):
    indices = episodes.data_dict["indices"]
    indices = torch.where(indices == -1, 0, indices)
    
    mask = (episodes.data_dict["indices"] != -1).sum(dim=1) > 4
    
    last_indices_loc = torch.count_nonzero(episodes.data_dict["indices"][mask] + 1, dim=-1) - 1
    last_indices = torch.gather(indices, dim=-1, index=last_indices_loc.unsqueeze(-1)).squeeze(-1)
    
    x0_x1 = env.graph.pos[indices[mask,0:-2]] - env.graph.pos[indices[mask,1:-1]]
    x1_x2 = env.graph.pos[indices[mask,1:-1]] - env.graph.pos[indices[mask,2:]]
    thetas = torch.acos(F.cosine_similarity(x1_x2, x0_x1, dim=-1))
    thetas[episodes.data_dict["indices"][mask, 2:] == -1] = 0.0
    
    thetas_mask = (thetas[:,:2] < theta_threshold_t).all(dim=-1) & \
                  (thetas[:,2:] < theta_threshold_d).all(dim=-1) 

    edep_mask = env.graph.edep[last_indices] > edep_threshold
    mask[mask.clone()] =  thetas_mask & edep_mask
    
    return thetas, mask

def true_tracks_from_graph(env: ParallelEpisodes):
    node_idx = torch.arange(env.graph.y.shape[0], device=env.graph.y.device)
    s, indices = torch.sort(env.graph.y)
    _, counts = torch.unique_consecutive(s, return_counts=True)

    true_tracks = [sorted(t.tolist(), reverse=True) for t in torch.split(node_idx[indices], counts.tolist())] 
    return true_tracks

def true_primary_tracks_from_graph(env):
    node_idx = torch.arange(env.graph.y.shape[0], device=env.graph.y.device)
    s, indices = torch.sort(env.graph.y)
    _, counts = torch.unique_consecutive(s, return_counts=True)

    true_tracks = [sorted(t.tolist(), reverse=True) \
        for t in torch.split(node_idx[indices], counts.tolist()) \
        if torch.all(env.graph.is_primary[t])] 
    return true_tracks

def to_spherical(env, track):
    x0_x1 = torch.tensor([0.0, 0.0, 1.0], device=env.graph.pos.device)
    x2_x3 = env.graph.pos[track[1]] - env.graph.pos[track[0]] 
    cart = x2_x3 - x0_x1
    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
    theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
    phi = torch.acos(cart[..., 2] / rho.view(-1)).view(-1, 1)

    spherical = torch.cat([rho, theta, phi], dim=-1)
    return torch.nan_to_num(spherical, 0.0)

def purity(env, tracks):
    corr = [(env.graph.y[t] == env.graph.y[t[0]]).all() for t in tracks if env.graph.is_primary[t].all()] 
    if len(corr) == 0: return 0
    return (torch.stack(corr).sum()/len(corr)).cpu().numpy()

def efficiency(env, true_sets, tracks):
    corr = [(env.graph.y[t] == env.graph.y[t[0]]).all() for t in tracks if env.graph.is_primary[t].all()]
    if len(corr) == 0: return 0
    return (torch.stack(corr).sum()/len(true_sets)).cpu().numpy()


def get_particle_densities(df, max_layer: int = 10, relative: bool = False):
    df = preprocess(df)
    rhos = np.empty((max_layer,))
    for i in range(max_layer):
        df_layer = df[df.z == i]
        q = max(df_layer.posY.quantile(0.9545), df_layer.posX.quantile(0.93))
        df_dens = (df_layer.posX**2 + df_layer.posY**2 <= q**2)
        rhos[i] = len(df_dens)/(PI * q**2)
        
    if relative: rhos = rhos/rhos[0]
        
    return rhos