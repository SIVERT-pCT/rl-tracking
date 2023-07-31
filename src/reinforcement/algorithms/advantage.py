from typing import Tuple, Union

import torch

from src.reinforcement.environment import ParallelEnvironment
from src.reinforcement.episodes import ParallelEpisodes
from src.utils.phys import estim_track_log_prob


class ExpMovingAverage:
    def __init__(self, alpha: float) -> None:
        """Maintains moving averages of variables.
        
        @param alpha: Weighting factor alpha
        """
        self._num_values = 0
        self._value: torch.Tensor = 0
        self._alpha = alpha
        
    def update(self, x: torch.Tensor) -> None:
        """Update the exponential moving average estimate using the 
        the current input x.

        @param x: Current incoming data point.
        """
        self._value = self._alpha * x + (1-self._alpha) * self._value \
            if self._num_values != 0 else x
            
        self._num_values += 1
    
    @property
    def value(self) -> torch.Tensor:
        return self._value


class GenAdvEstimator:
    def __init__(self, gae_gamma: float = 0.99, gae_lambda: float = 1, alpha: float = 1.0,
                 reward_norm: bool = True, indep_norm: bool = True,  clip_range: Tuple[int, int] = [-5000, 10]) -> None:
        """Implementation of the generalized advantage estimation algorithm by
        Schulman et al. (2018): 'HIGH-DIMENSIONAL CONTINUOUS CONTROL USING 
        GENERALIZED ADVANTAGE ESTIMATION' + additional exponential reward normalization
        and clipping with rolling exponential average filter.

        @param gae_gamma: Gamma factor, defaults to 0.99.
        @param gae_lambda: Lambda factor, defaults to 0.95.
        @param individual_normalize: Determines whether rewards of different transitions should be 
                                     normalized independently, defaults to True
        @param clip_range: Max and min range of reward clipping, defaults to [-10, 10]
        """
        self.ema_std = None
        self.ema_mu = None
        self.alpha = alpha
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.indep_norm = indep_norm
        self.reward_norm = reward_norm
        
        if self.indep_norm:
            self.mu_d1, self.std_d1 = ExpMovingAverage(alpha), ExpMovingAverage(alpha)
            self.mu_t1, self.std_t1 = ExpMovingAverage(alpha), ExpMovingAverage(alpha)
            self.mu_t2, self.std_t2 = ExpMovingAverage(alpha), ExpMovingAverage(alpha)
        else:
            self.mu, self.std = ExpMovingAverage(alpha), ExpMovingAverage(alpha)
        
        
    def _calculate_rewards(self, env: ParallelEnvironment, episodes: ParallelEpisodes) -> torch.Tensor:
        """Calculate the individual rewards obtained for each action. Determined
        using the probabilities of underlying 
         interactions (multiple coulomb 
        scattering).

        @param env: Reinforcement learning environment containing trajectory graph that describes the current 
                    reconstruction environment and lookup object for faster edge queries.
        @param episodes: Sampled episodes (sampled in environment env under current policy pi).
                         
        @return: Rewards for each timestep and track candidate.
        """
        rewards = torch.zeros_like(episodes.values_raw)
        masks = episodes._create_property_mask(skip_first=True, skip_seed=True)

        for i, (track, mask) in enumerate(zip(episodes.trajectories, masks)): 
            _, rew = estim_track_log_prob(env.graph, track)
            rewards[i, ~mask] = rew.float()
            
        return rewards
        
    
    def _calculate_advantages(self, episodes: ParallelEpisodes, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate advantage estimates using generalized advantage estimation.

        @param env: Reinforcement learning environment containing trajectory graph that describes the current 
                    reconstruction environment and lookup object for faster edge queries.
        @param episodes: Sampled episodes (sampled in environment env under current policy pi).
        @param rewards: Estimated rewards with same shape as episodes.values_raw (contain padding).

        @returns: Calculated advantages estimates.
        """
        adv = torch.zeros_like(episodes.values_raw)
        timesteps = rewards.shape[1]
        
        mask = episodes._create_property_mask(skip_seed=True, skip_first=False)
        
        # Buffer (single trajectory) [tPad,...,tPad, tT, tT-1,...,t1, t0]
        for i in range(timesteps):
            old_v = episodes.values_raw[:,i]   * ~mask[:,i]
            new_v = episodes.values_raw[:,i-1] * ~mask[:,i-1] \
                 if i > 0 else torch.zeros_like(episodes.values_raw[:,i])

            delta_t = rewards[:,i] + self.gae_gamma * new_v - old_v
            adv[:,i] = adv[:,i-1] * self.gae_gamma * self.gae_lambda + delta_t
            
        return adv
    
    
    def _expand_to_mask_shape(self, value: torch.Tensor, mask: torch.Tensor):
        """Expand boolean value tensor to mask shape

        @param value: Boolean value tensor with N = #False in mask elements 
        @param mask: Mask tensor describing masked values in episodes
        
        @return: Expanded value tensor of shape mask.
        """
        value_expand = torch.zeros_like(mask).type(torch.bool)
        value_expand[~mask] = value
        return value_expand
    
    
    def _normalize(self, value: torch.Tensor, mu_aggr: ExpMovingAverage, std_aggr: ExpMovingAverage):
        """Normalize rewards based on mean and std estimate.

        @param value: Reward values.
        @param mu_aggr: Mean estimate
        @param std_aggr: Std estimate

        @return: Normalized rewards
        """
        return (value - mu_aggr.value) / (std_aggr.value + 1e-10)
        
    
    def calculate_returns_and_advantages(self, env: ParallelEnvironment, episodes: ParallelEpisodes):
        """Calculates the returns and advantage estimates of multiple track candidates
        (trajectories) using generalized advantage estimation.

        @param env: Reinforcement learning environment containing trajectory graph that 
                    describes the current reconstruction environment and lookup object 
                    for faster edge queries.
        @param episodes: Sampled episodes (sampled in environment env under current
                         policy pi).
        
        @returns: Estimated returns and advantages.
        """
        rew_raw = self._calculate_rewards(env, episodes)
        mask = episodes._create_property_mask(skip_first=True, skip_seed=True)
        
        # Mask empty returns (zero values) in order to obtain accurate 
        # mean and std estimations
        rew_mask = rew_raw[~mask]
        rew_norm = torch.zeros_like(rew_raw)
        
        if self.reward_norm:
            if self.indep_norm:
                # Normalizes reward for detector -> detector & detector -> tracker & 
                # tracker -> tracker transitions independently.
                nit = env.graph.next_is_tracker[episodes.obs_indices]
                it = env.graph.is_tracker[episodes.obs_indices]
                
                nit_expand = self._expand_to_mask_shape(nit, mask)
                it_expand = self._expand_to_mask_shape(it, mask)
                
                rcm_d1, rcm_t1, rcm_t2 = (rew_mask[~nit], rew_mask[nit & ~it], rew_mask[nit & it]) 
                
                # Update mean and std estimates based on new sampled track candidates
                self.mu_d1.update(torch.mean(rcm_d1)); self.std_d1.update(torch.std(rcm_d1))
                self.mu_t1.update(torch.mean(rcm_t1)); self.std_t1.update(torch.std(rcm_t1))
                self.mu_t2.update(torch.mean(rcm_t2)); self.std_t2.update(torch.std(rcm_t2))
                
                rew_norm[~mask & nit_expand & ~it_expand] = self._normalize(rcm_t1, self.mu_t1, self.std_t1)                          
                rew_norm[~mask & nit_expand & it_expand] = self._normalize(rcm_t2, self.mu_t2, self.std_t2)
                rew_norm[~mask & ~nit_expand] = self._normalize(rcm_d1, self.mu_d1, self.std_d1)
            else:
                self.mu = self.mu.update(torch.mean(rew_mask))
                self.std = self.std.update(torch.std(rew_mask))
                
                rew_norm[~mask] = self._normalize(rew_mask, self.mu, self.std)
        else:
            rew_norm = rew_raw
        
        adv = self._calculate_advantages(episodes, rew_norm)
        ret = adv + episodes.values_raw
        
        return ret, adv, rew_raw