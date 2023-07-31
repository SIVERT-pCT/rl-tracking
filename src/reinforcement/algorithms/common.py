import torch

from src.reinforcement.episodes import ParallelEpisodes

def explained_variance(episodes: ParallelEpisodes) -> torch.Tensor:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert episodes.returns.ndim == 1 and episodes.values.ndim == 1
    var_y = torch.var(episodes.returns)
    return torch.nan if var_y == 0 else 1 - torch.var(episodes.returns - episodes.values) / var_y