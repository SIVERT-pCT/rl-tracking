from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.reinforcement.environment import ParallelEnvironment


class ParallelEpisodes:
    def __init__(self, evaluate: bool = True):
        """Data structure for storing and handling sampled particle trajectories.

        @param evaluate: True if evaluation mode is used for sampling, defaults to True
        """
        self.data_dict = {
            "indices": [],
            "actions": [],
            "log_probs": [],
            "entropy": [],
            "values": [],
            "hidden": [],
            "done": [],
            "seed": [],
            "return": None,
            "advantages": None
        }
        
        self._pad_keys = ["indices", "actions", "log_probs", 
                          "entropy", "values", "hidden", "done", "seed"]
        self._pad_val = [-1, -1, 0, 0, 0, 0, False, False]
        self._pad_val_finish = [-1, -1, 0, 0, 0, 0, True, False]
        self._train_only_keys = ["log_probs", "values"]
        self._finished = None
        self._evaluate = evaluate
        self._shuffled_indices = None
        self._finished = False
            
    def update(self, indices: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor, 
               entropy: torch.Tensor, values: torch.Tensor, hidden: torch.Tensor,
               done: torch.BoolTensor, seed: torch.BoolTensor) -> None:
        """Update episodes representation by adding data of a new sampling step.

        @param indices: Observation indices of current time steps.
        @param actions: Selected actions, determined by policy.
        @param log_probs: log(p) determined using softmax policy (under action a)
        @param entropy: Entropy of policy.
        @param values: Value estimates V(s) determined by value network.
        @param hidden: Hidden representation of LSTM module (only for POMDP)
        @param done: Done vector containing status of episodes for all parallel instances.
        @param seed: Seeding vector containing seeding information for all parallel instances.
        """
        
        arg_dict = locals()  # Fetch the method locals (arguments) and remove self
        del arg_dict["self"] # since it doesn't contain any information about the episode
        
        for key, value in arg_dict.items():
            self._sanity_check(key, value)
            self.data_dict[key].insert(0, value)
            
    def finish(self, indices: torch.Tensor) -> None:
        """Last update of final indices after last traversal step.

        @param indices: Observation indices obtained after last traversal step (all done).
        """
        self.data_dict["indices"].insert(0, indices)
        for key, value in zip(self._pad_keys[1:], self._pad_val_finish[1:]):
            if key == "hidden" and self.data_dict[key][0] == None:
                continue
            
            self.data_dict[key].insert(0, torch.empty_like(self.data_dict[key][-1]).fill_(value))
        
        for key, val in zip(self._pad_keys, self._pad_val):
            if key == "hidden" and self.data_dict[key][0] == None:
                continue
            
            self.data_dict[key] = pad_sequence(self.data_dict[key], batch_first=False, 
                                               padding_value=val)
            
        self._finished = True
            
    def assign_returns(self, returns: torch.Tensor) -> None:
        self._assign_values("returns", returns)
    
    def assign_advantages(self, advantages: torch.Tensor)  -> None:
        self._assign_values("advantages", advantages)
        
    def iterate_minibatches(self, env: ParallelEnvironment, batch_size: int = 64, 
                            sampler = None) -> dict:
        """Python iterator used in order to loop over minibatches of sampled experience
        used for policy updates. 

        @param env: Parallel environment corresponding to the sampled episodes.
        @param batch_size: Number of elements in each minibatch, defaults to 64
        @param sampler: Torch sampler used in order to iterate minibatches, defaults to None
        
        @yields: Minibatch of sampled experience of policy updates.
        """
        
        indices = torch.ones_like(self.obs_indices).cumsum(dim=0) - 1
        loader = DataLoader(dataset=indices, batch_size=batch_size, sampler=sampler)
        for indices_batch in loader:
            minibatch = {"obs": self.obs_indices[indices_batch],
                         "old_obs": self.old_obs_indices[indices_batch],
                         "act": self.act_indices[indices_batch], 
                         "log_probs": self.log_probs[indices_batch], 
                         "hidden": None if self.data_dict["hidden"][0] is None else self.hidden[indices_batch],
                         "advantages": None if self.data_dict["advantages"] is None else self.advantages[indices_batch],
                         "values":  None if self.data_dict["values"] is None else self.values[indices_batch],
                         "returns": self.returns[indices_batch]}
            
            yield minibatch
            
        
    def iterate_minibatches_undersampled(self, env: ParallelEnvironment, batch_size = 64):
        """Python iterator (with additional undersampling of majority class: detector experience) 
        used in order to loop over minibatches of sampled experience used for policy updates.

        @param env: Parallel environment corresponding to the sampled episodes.
        @param batch_size: Number of elements in each sampled minibatch, defaults to 64.
        
        @returns: Python generator (iterate_minibatches) yielding undersampled minibatches.
        """
        N = len(self.obs_indices)
        w = self._calculate_weights(env)
        p = self._apply_weights(env, w)

        sampler = WeightedRandomSampler(weights=p, num_samples=min(N, 1024), replacement=True)
        return self.iterate_minibatches(env, batch_size, sampler)        
    
    def _sanity_check(self, key: str, value: torch.Tensor) -> None:
        if key in ["done", "seed"]:
            assert value.dtype == torch.bool, f"Type of {key} should be bool"
        elif key in ["indices", "actions"]:
            assert value.dtype == torch.long, f"Type of {key} should be long"
        elif key in ["hidden"]:
            assert (value == None or torch.is_floating_point(value)),f"Type of {key} should be none or float"
        else:
            assert torch.is_floating_point(value), f"Type of {key} should be float"
    
    def _assign_values(self, key, values) -> None:
        assert self._finished, "The episode must be finished before assigning advantages"
        assert values.shape == self.data_dict["values"].shape, "Invalid shape"
        self.data_dict[key] = values
        
        
    def _get_from_dict(self, key) -> None:
        assert self._finished, "The episode must be finished" \
                               "before accessing elements"
            
        mask = self.data_dict["done"] & self.data_dict["seed"]
        mask = mask & self._get_padding_mask()
        return self.data_dict[key], mask
    
    def _create_property_mask(self, skip_first: bool = True, skip_seed: bool = True):
        keep_first_mask = torch.ones_like(self.data_dict["done"]).bool()
        if not skip_first: 
            keep_first_mask[:,:-1] = self.data_dict["done"][:,:-1] & self.data_dict["done"][:,1:]
    
        mask = (self.data_dict["done"] & keep_first_mask)  | self._get_padding_mask()
        if skip_seed: mask = mask | self.data_dict["seed"]
        
        if skip_first:
            mask[:,0] = True
            
        return mask
    
    def _get_padding_mask(self) -> torch.BoolTensor:
        return self.data_dict["indices"] == -1
    
    def _split_trajectories(self, indices: torch.Tensor, mask: torch.BoolTensor) -> List[torch.Tensor]:
        count = torch.sum(~mask, dim=1)
        count = count[count != 0] # Mask empty trajectories
        return torch.split(indices[~mask], count.tolist())
    
    def _apply_weights(self, env: ParallelEnvironment, weights: Tuple[float, float, float]) -> torch.Tensor:
        w0, w1, w2 = weights
        nit = env.graph.next_is_tracker[self.obs_indices]
        it = env.graph.is_tracker[self.obs_indices]
        
        w_tensor = torch.ones_like(self.obs_indices).float()
        w_tensor[nit &  it] *= w0
        w_tensor[nit & ~it] *= w1
        w_tensor[~nit] *= w2
        return w_tensor
    
    def _calculate_weights(self, env: ParallelEnvironment) -> Tuple[float, float, float]:
        nit = env.graph.next_is_tracker[self.obs_indices]
        it = env.graph.is_tracker[self.obs_indices]

        num_1st = (nit & it).sum()
        num_2nd = (nit & ~it).sum()
        num_det = (~nit).sum()
        total = len(self.obs_indices)

        return ((total - num_1st)/total, 
                (total - num_2nd)/total, 
                (total - num_det)/total)
    
    @property
    def values_raw(self) -> torch.Tensor:
        return self.data_dict["values"]
    
    @property
    def indices_raw(self) -> torch.Tensor:
        return self.data_dict["indices"]
    
    @property
    def trajectories(self) -> List[torch.Tensor]:
        mask = self._create_property_mask(skip_first=False, skip_seed=False)
        return self._split_trajectories(self.data_dict["indices"], mask)
    
    @property
    def entropy_trajectories(self) -> List[torch.Tensor]:
        mask = self._create_property_mask(skip_first=False, skip_seed=False)
        return self._split_trajectories(self.data_dict["entropy"], mask)
    
    @property
    def log_probs_trajectories(self) -> List[torch.Tensor]:
        mask = self._create_property_mask(skip_first=False, skip_seed=False)
        return self._split_trajectories(self.data_dict["log_probs"], mask)
    
    @property
    def obs_indices(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["indices"][~mask]
    
    @property
    def old_obs_indices(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        old_indices = self.data_dict["indices"].clone()
        old_indices[:,:-1] = old_indices[:,1:].clone()
        return old_indices[~mask]
    
    @property
    def values(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["values"][~mask]
    
    @property
    def hidden(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["hidden"][~mask]
    
    @property
    def log_probs(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["log_probs"][~mask]
    
    @property
    def act_indices(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["actions"][~mask]
    
    @property
    def returns(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["returns"][~mask]
    
    @property
    def advantages(self) -> torch.Tensor:
        mask = self._create_property_mask(skip_first=True, skip_seed=True)
        return self.data_dict["advantages"][~mask]