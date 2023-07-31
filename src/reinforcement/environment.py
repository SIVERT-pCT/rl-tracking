import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..utils.graph import TGraph, query_neighborhood
from ..utils.pandas import (is_spot_scanning, open_dataframe_by_extension, preprocess)


class EnvironmentBase(ABC):
    def __init__(self, file: Union[str, pd.DataFrame], skip_tracker: bool = False, to_wet: bool = True, 
                 filter_secondaries: bool = False, cluster_threshold: float = None,
                 device: str = 'cpu') -> None:
        """Abstract representation of the reinforcement learning 
        environment containing the defined transition model for 
        traversing the graph based on actions.

        @param file: Filepath or data frame containing the readout data of the detector.
        @param skip_tracker: Determines whether the tracking layers
                             should be skipped, defaults to False.
        @param to_wet: Determines whether the relative z positions should
                        be converted to water equivalent thicknesses, defaults to False.
        @param filter_secondaries: Determines whether secondary particles should be 
                                   filtered and removed from the TGraph (USING GROUND TRUTH !!!),  
                                   defaults to False.
        @param cluster_threshold: Cluster size threshold which is required in order to consider 
                                  hit as primary particle , defaults to None.
        @param device: Torch torch.Device where the content of the environment 
                       should be located, defaults to 'cpu'.
        """
        self._device = device
        self._evaluate = False
        
        if file is not None and skip_tracker is not None:
            self._graph = self._load_trajectory_graph(file, skip_tracker, to_wet,
                                                    filter_secondaries, cluster_threshold,
                                                    device)
        else:
            self._graph = None
        
    @abstractmethod
    def reset(self, evaluate: Any):
        pass
    
    @abstractmethod
    def step(self, action: Any, k: int):
        pass
    
    def _load_trajectory_graph(self, file: Union[str, pd.DataFrame], skip_tracker: bool, to_wet: bool, 
                               filter_secondaries: bool, cluster_threshold: float,
                               device: str = 'cpu') -> TGraph:#
        """Converts given input file/data frame to a TGraph representation.

        @param file: Filepath or data frame containing the readout data of the detector.
        @param skip_tracker: Determines whether the tracking layers
                             should be skipped, defaults to False.
        @param to_wet: Determines whether the relative z positions should
                        be converted to water equivalent thicknesses, defaults to False.
        @param filter_secondaries: Determines whether secondary particles should be 
                                   filtered and removed from the TGraph (USING GROUND TRUTH !!!),  
                                   defaults to False.
        @param cluster_threshold: Cluster size threshold which is required in order to consider 
                                  hit as primary particle , defaults to None.
        @param device: Torch torch.Device where the content of the environment 
                       should be located, defaults to 'cpu'.
        
        @returns: TGraph object representation of detector readouts.
        """
        if not isinstance(file, pd.DataFrame):
            file = open_dataframe_by_extension(file)
            
        return TGraph.from_df(file, skip_tracker=skip_tracker, to_wet=to_wet, 
                              filter_secondaries=filter_secondaries,
                              cluster_threshold=cluster_threshold, device=device)
    
    @property
    def graph(self) -> TGraph:
        return self._graph


class ParallelEnvironment(EnvironmentBase):
    def __init__(self, file: Union[str, pd.DataFrame], num_instances: int, skip_tracker: bool = True, to_wet: bool = True,
                 filter_secondaries: bool = False,  cluster_threshold: float = None, last_n : int= 5, 
                 device: str = "cpu") -> None:
        """Abstract representation of a parallel reinforcement 
        learning environment for multiple parallel instances  
        (not multitask!) containing the defined transition model 
        for traversing the graph based on actions.

        @param file: Filepath or data frame containing the readout data of the detector.
        @param skip_tracker: Determines whether the tracking layers
                             should be skipped, defaults to False.
        @param to_wet: Determines whether the relative z positions should
                        be converted to water equivalent thicknesses, defaults to False.
        @param filter_secondaries: Determines whether secondary particles should be 
                                   filtered and removed from the TGraph (USING GROUND TRUTH !!!),  
                                   defaults to False.
        @param cluster_threshold: Cluster size threshold which is required in order to consider 
                                  hit as primary particle , defaults to None.
        @param device: Torch torch.Device where the content of the environment 
                       should be located, defaults to 'cpu'.
        """
        super().__init__(file, skip_tracker, to_wet, filter_secondaries, 
                         cluster_threshold, device)
        
        self._num_instances = num_instances
        self.last_n = last_n
        
        self._observations: torch.Tensor = None
        self._action_space: torch.Tensor = None
        self._seed: torch.Tensor = None
        self._step: int = None

        
    def reset(self, evaluate: bool = False):
        """ Resets the current states of the environment and samples n new 
        initial observations from the hits of the first calorimeter layer 
        
        @evaluate: Determines whether evaluation mode should be activated (all tracks finishing in
                   last layers are selected & missing hits are added as new tracks during 
                   reconstruction), defaults to False
        
        @returns: Sampled observations, seeding (bool if required), done
        """
        self._evaluate = evaluate
        self._observations = self._init_observations(evaluate)
        self._action_space = query_neighborhood(self._graph, self._observations, to_list=False) 
        self._seed = torch.ones_like(self._observations).type(torch.bool)
        self._step = 0

        return self._observations, self._seed, torch.zeros_like(self._observations).type(torch.bool)
    
    
    def step(self, actions: torch.Tensor, k: int = 1, add_missing: bool = True) -> Tuple[torch.Tensor, float, bool]:
        """ Updates the internal state of the environment by performing
        the corresponding state transition defined by the selected action.
        
        @param action: Action to be performed for current state.
        @param k: Number of candidates selected by seeding approach.
        @param add_missing: Determines whether remaining (missing) particles
                            should be added to the reconstruction queue.
        
        @returns: Tuple of state idx, masks [n_observations] (only required for eval) 
                  and done tensor [n_observations].
        """
        self._observations, self._seed = self._perform_actions(actions, k, add_missing)     
        self._action_space = query_neighborhood(self._graph, self._observations, to_list=False)

        if isinstance(self._action_space, tuple):
            done = [len(act_space) == 0 for act_space in self._action_space]
            done = torch.tensor(done, device=self._device, dtype=torch.bool)
        else:
            done = torch.all((self._action_space == -1), dim=-1)
            
        # Apply additional condition for done where action is
        # -1 (required for seeding where less then k possible 
        # seeds exist).
        done[:len(actions)] = torch.logical_or(done[:len(actions)], actions == -1)
        self._step += 1
        return self._observations, self._seed, done

    
    def _append_missing(self, obs: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Appends all node indices that aren't a subset of both the current observations 
        and the current detector layers to the set of observations. 

        @param obs: Current observations obtained from tracking.
        @param step: Current reconstruction step (starting from 0 with reset).
        
        @returns: Tuple containing concatenated observations and corresponding mask tensor.
        """
        lwn_pad = F.pad(self.graph.num_nodes_layerwise, (1, 1), 'constant', 0.0)[:-1]
        lwn_pad_cumsum = lwn_pad.cumsum(dim=-1)
        
        node_idx = torch.arange(lwn_pad_cumsum[step+1], lwn_pad_cumsum[step+1] + lwn_pad[step+2])

        obs_set, layer_set = set(obs.tolist()), set(node_idx.tolist())
        missing_observations = torch.tensor(list(layer_set - obs_set), device=self._device, dtype=torch.long)
        obs = torch.cat((obs, missing_observations))
        return obs
    
    
    def _perform_actions(self, actions: torch.Tensor, k: int = 1, add_missing: bool = True) -> torch.Tensor:
        """ Determines the next observation based on a defined set
        of chosen actions. The actions are represented as the indices 
        of actions in the action_space tensor respectively.
        
        @param action: Chosen set of actions. Actions for finished tracks 
                       should either be -1. 
        @param k: Number of seeds used for track seeding.
        
        @returns: A new set of observations based on the chosen actions.
        """
        indices = torch.empty_like(actions).fill_(-1)
        repeats = self._seed.type(torch.long) * k + (~self._seed).type(torch.long)
        
        #Initialize all indices to old indices by default (leave for masked actions)
        indices = self._observations.repeat_interleave(repeats)
        n_indices = indices.shape[0] #Total number of indices before appending missing
        act_mask = actions == -1 # Mask if track is finished (action == -1)
        
        action_space_repeated = self._action_space.repeat_interleave(repeats, dim=0)

        indices[~act_mask] = torch.gather(action_space_repeated[~act_mask], dim=1,
                                            index=actions[~act_mask].unsqueeze(1)).squeeze(1)
                    
        assert torch.all(indices != -1), f"Invalid action for parallel worker(s): " + \
                                            f"{torch.where(indices == -1)[0].cpu().tolist()}"
        
        if add_missing: indices = self._append_missing(indices, self._step)
        
        requires_seed = torch.ones_like(indices).type(torch.bool)
        requires_seed[:n_indices] = False
        return indices, requires_seed
   
        
    def _init_observations(self, evaluate: bool):
        if evaluate: return self._select_init_observations()
        return self._sample_init_observations()
    
    
    def _select_init_observations(self) -> List[int]:
        return torch.arange(self.graph._layerwise_node_count[0], device=self._device)
    
    
    def _sample_init_observations(self) -> torch.Tensor:
        n_samples = self.parallel_instances
        n_choices = torch.sum(self.graph.num_nodes_layerwise[:self.last_n])
        probs = torch.ones(n_choices, dtype=torch.float32, device=self._device)
        obs = torch.multinomial(probs.cpu(), num_samples=n_samples, replacement=True).type(torch.long).to(self._device)
        return obs
    
    @property
    def current_step(self) -> int:
        return self._step
        
    @property
    def parallel_instances(self) -> int:
        if self._evaluate: return self.graph.num_nodes_layerwise[0]
        else: return self._num_instances
        
    @parallel_instances.setter
    def parallel_instances(self, num_instances: int) -> None:
        self._num_instances = num_instances      
        
        
class ParallelEnvironmentDataset:
    def __init__(self, directory: str, num_instances: int = None, device: torch.device = None) -> None:
        """Loads an created dataset containing multiple individual environments.

        @param directory: Directory containing the dataset.
        @param num_instances: Number of parallel instances to override 
                              existing value, defaults to None
        @param device: Mapping location of the environment
        """
        self.n = 0
        self.num_instances = num_instances
        self.directory = directory
        self.device = device
        self.files = [f for f in os.listdir(directory) if f.endswith('.penv')]
        
    def sample(self) -> ParallelEnvironment:
        """Randomly samples an environment from the list of environment.

        @returns: Randomly sampled environment.
        """
        index = int(np.random.choice(len(self.files), 1))
        return self.__load_environment(index)
    
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < len(self.files):
            env = self.__load_environment(self.n)
            self.n += 1
            return env
        else:
            raise StopIteration
        
    def __load_environment(self, index: int) -> ParallelEnvironment:
        env: ParallelEnvironment = torch.load(os.path.join(self.directory, self.files[index]), map_location=self.device)
        env._device = self.device
        
        if self.num_instances is not None:
            env.parallel_instances = self.num_instances
        return env
    
    
    @classmethod
    def __try_create(cls, path: str):
        if not os.path.isdir(path):
            os.makedirs(path)
                
    @classmethod
    def __export_to_file(cls, df: pd.DataFrame, splits: List[np.ndarray], 
                         db_dir: str, num_instances: int, skip_tracker: bool,
                         cluster_threshold: float, device: str, offset: int = None):
        
        for i, split in enumerate(splits):
            d = df.loc[df.eventID.isin(split)]
            env = ParallelEnvironment(d, num_instances, skip_tracker=skip_tracker,
                                      to_wet=False, filter_secondaries=False,
                                      cluster_threshold=cluster_threshold, device=device)
            torch.save(env, os.path.join(db_dir, f"{i if offset is None else i + offset}.penv"))
            
            
            
    @classmethod
    def from_file(cls, file_path: str, db_dir_join: str, events_per_env: int = 100, 
                  primaries_per_spot = None, num_instances: int = 64, skip_tracker: bool = False,
                  cluster_threshold: float = 2, device: str = "cpu", offset: int = 0) -> int:
        """Creates a partial dataset containing multiple environments from a single input files
        containing raw simulated data (detector readouts).

        @param file_path: File path pointing to the dataframe containing particle information.
        @param db_dir_join: Path where the database should be stored (directory).
        @param events_per_env: Number of events per environment, defaults to 100
        @param primaries_per_spot: Number of primaries contained in each spot, defaults to None
        @param num_instances: Number of parallel instances for default initialization, defaults to 64
        @param skip_tracker: If true all tracking layers are excluded from graph creation, defaults to False
        @param cluster_threshold: Cluster size used for filtering secondaries, defaults to 2
        @param device: Computational device where dataset should be crated, defaults to "cpu"
        @param offset: Numerical offset for numbering the dataset files, defaults to 0
        
        @returns: New offset based on the created files.
        """
        df = open_dataframe_by_extension(file_path)
        df = preprocess(df)
        
        cls.__try_create(db_dir_join)
        
        if not is_spot_scanning(df):
            df["spotX"] = 0
            df["spotY"] = 0
            
        for _, spot in df[["eventID", "spotX", "spotY"]].groupby(["spotX", "spotY"]):
            events = spot.groupby("eventID").size().keys().astype(int)
            event_groups = [events[i*events_per_env:(i+1)*events_per_env]  \
                            for i in range(len(events)//events_per_env)\
                            if (i+1)*events_per_env <= len(spot)]
            
            if primaries_per_spot is not None:
                assert events_per_env <= primaries_per_spot
                event_groups_selection = primaries_per_spot//events_per_env
                event_groups = event_groups[:event_groups_selection]
                
            cls.__export_to_file(df, event_groups, db_dir_join, num_instances, 
                                skip_tracker, cluster_threshold, device, offset=offset)
            offset += len(event_groups)
        
        return offset
    
    @classmethod
    def from_files(cls, file_paths: List[str], db_names: List[str], prefix: str, db_dir: str, events_per_env: int = 100, 
                        primaries_per_spot = None, num_instances: int = 64, skip_tracker: bool = False, cluster_threshold: float = 2, 
                        device: str = "cpu"):
        """Creates a dataset containing multiple environments from multiple input files 
        containing raw simulated data (detector readouts). Data requires manual separation 
        of train/test data --> should be independent of each other.

        @param train_files: List of file path containing all training data.
        @param test_files: List of file path containing all test data.
        @param db_dir: Path where the database should be stored (directory).
        @param events_per_env: Number of events per environment, defaults to 100
        @param num_instances: Number of parallel instances for default initialization, defaults to 64.
        @param skip_tracker: If true all tracking layers are excluded from graph creation, defaults to False.
        @param cluster_threshold: Cluster size used for filtering secondaries, defaults to 2.
        @param device: Computational device where dataset should be crated, defaults to "cpu"
        """
        offset = 0
        for file_path, name in zip(file_paths, db_names):
            db_dir_join = f"{db_dir}/{prefix}_{name}/{events_per_env}"
            offset = cls.from_file(file_path, db_dir_join, events_per_env, primaries_per_spot,
                                    num_instances, skip_tracker, cluster_threshold,
                                    device, offset)
                    