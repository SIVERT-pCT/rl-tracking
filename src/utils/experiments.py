import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import os
import fcntl
import random
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import jsons
import numpy as np
import pandas as pd
import torch
from deepdiff import DeepDiff
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.utils.eval import get_metrics_for_dataset

from ..embedding.gnn import GatConvEmbedding
from ..reinforcement.algorithms.agent import PPOAgent
from ..reinforcement.environment import ParallelEnvironmentDataset
from ..reinforcement.models import PointerPolicyNet
from ..reinforcement.sampling import evaluate_actions
from ..supervised.datasets import TrackDataset


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True, warn_only=True)


def delete_test_info(json: dict):
    json = jsons.dumps(json) #FIXME:
    json = jsons.loads(json)
    
    keys = ["directory", "test_events", "test_files"]
    for key in keys:
        del json["dataset"][key]

    return json


class LockDirectory():
    def __init__(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory

    def __enter__(self):
        self.dir_fd = os.open(self.directory, os.O_RDONLY)
        try:
            fcntl.flock(self.dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError as ex:             
            raise Exception(f"Another training instance is already running in dir {self.directory} - quitting.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # fcntl.flock(self.dir_fd,fcntl.LOCK_UN)
        os.close(self.dir_fd)


class ConfigBase(ABC):
    def __init__(self) -> None:
        pass
    
    def to_json(self, del_test_info: bool = False, no_jdkwargs: bool = False):
        d = self.__dict__
        if del_test_info: d = delete_test_info(d)
        jdkwargs =  None if no_jdkwargs else {"indent": 4}
        return jsons.dumps(d, jdkwargs=jdkwargs)
    
    def to_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, data: str, to_object: bool = True):
        if to_object:
            return jsons.loads(data, cls)
        else:
            return jsons.loads(data)
    
    @classmethod
    def from_file(cls, path: str, to_object: bool = True):
        with open(path, "r") as f:
            return cls.from_json(f.read(), to_object)

    
class DatasetConfig(ConfigBase):
    def __init__(self, directory: str, train_file: str, test_files: Union[Dict[str, str], None],
                 train_events: int, test_events: Union[List[int], None], filter_secondaries: bool = False,
                 skip_tracker: bool = False, cluster_threshold: int = 2) -> None:
        
        super().__init__()
        self.directory = directory
        self.train_file = train_file
        self.test_files = test_files
        self.train_events = train_events
        self.test_events = test_events
        self.filter_secondaries = filter_secondaries
        self.skip_tracker = skip_tracker
        self.cluster_threshold = cluster_threshold
        
        
    def _get_config_dir(self):
        return os.path.join(self.directory, "dataset.json")

    def exists(self):
        f = self._get_config_dir()
        
        if not os.path.exists(self.directory) or \
           not os.path.exists(f) :
            return False, False
    
        try: comp = DatasetConfig.from_file(f)
        except: return False, False
        
        return True, comp.__dict__ == self.__dict__
    
         
    def generate_from_config(self, device: str = "cpu"):
        exists, matches = self.exists()
        
        if exists and matches:
            print("Skipping dataset creation..."); return
        elif exists and not matches:
            raise ValueError("Existing dataset does not match the dataset specified in config.")
        
        print("Generating datasets from config...")
        train_dir_join = f"{self.directory}/train_{self.train_events}"
        ParallelEnvironmentDataset.from_file(self.train_file, train_dir_join, self.train_events, skip_tracker=self.skip_tracker, 
                                             cluster_threshold=self.cluster_threshold, device=device)
        
        if self.test_files != None:
            for events in tqdm(self.test_events):
                ParallelEnvironmentDataset.from_files(self.test_files.values(), self.test_files.keys(), "test", self.directory, 
                                                    events, skip_tracker=self.skip_tracker, cluster_threshold=self.cluster_threshold, 
                                                    device=device)
            
        with open(self._get_config_dir(), "w") as f:
            f.write(self.to_json())
            
    
    def load_train_from_config(self, device: str = "cpu"):
        train_dir_join = f"{self.directory}/train_{self.train_events}"
        return ParallelEnvironmentDataset(train_dir_join, device=device)
    
    def iterate_events(self):
        for events in self.test_events:
            yield events
            
    def iterate_phantoms(self):
        for phantom in self.test_files.keys():
            yield phantom
    
    def load_test_from_config(self, phantom, events, device: str = "cpu"):
        dir_join = f"{self.directory}/test_{phantom}/{events}"
        dataset = ParallelEnvironmentDataset(dir_join, device=device)
        return dataset
                
                
class EmbeddingNetConfig(ConfigBase):
    def __init__(self, node_features: int, embedding_dim: int) -> None:
        super().__init__()
        self.node_features = node_features
        self.embedding_dim = embedding_dim
        
    def generate_from_config(self, device: str) -> GatConvEmbedding:
        return GatConvEmbedding(**self.__dict__).to(device)
    
    def save_model(self, model: GatConvEmbedding, model_dir: str, run: int) -> None:
        save_dir = os.path.join(model_dir, f"embedding_net_{run}.pt")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), save_dir)
    
    def load_from_config(self, model_dir: str, run: int, device: str) -> GatConvEmbedding:
        model = self.generate_from_config(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"embedding_net_{run}.pt")))
        return model
    
        
class ActorCriticConfig(ConfigBase):
    def __init__(self, act_size: int, obs_size: int, embedding_size: int, sim_scaling: int = 100,
                 use_act_enc: bool = True, enc_type: str = "PE-ARF", pomdp: bool = False) -> None:
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size
        self.embedding_size = embedding_size
        self.sim_scaling = sim_scaling
        self.use_act_enc = use_act_enc
        self.enc_type = enc_type
        self.pomdp = pomdp
        
    def generate_from_config(self, device: str) -> PointerPolicyNet:
        return PointerPolicyNet(**self.__dict__).to(device)
    
    def save_model(self, model: PointerPolicyNet, model_dir: str, run: int) -> None:
        save_dir = os.path.join(model_dir, f"actor_critic_{run}.pt")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), save_dir)
    
    def load_from_config(self, model_dir: str, run: int, device: str) -> PointerPolicyNet:
        model = self.generate_from_config(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"actor_critic_{run}.pt")))
        return model
    
    
class OptimizerConfig(ConfigBase):
    def __init__(self, lr: float, batch_size: int) -> Any:
        self.lr = lr
        self.batch_size = batch_size
    
    def generate_from_config(self, params):
        return Adam(lr=self.lr, params=params, amsgrad=True)
    
                
class ModelConfig(ConfigBase):
    def __init__(self, model_dir: str, actor_critic: ActorCriticConfig, 
                 embedding_net: EmbeddingNetConfig) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.actor_critic = actor_critic
        self.embedding_net = embedding_net 
        
    def generate_from_config(self, device: str) -> Tuple[PointerPolicyNet, GatConvEmbedding]:
        return self.actor_critic.generate_from_config(device), \
               self.embedding_net.generate_from_config(device)
               
    def load_from_config(self, run: int, device: str) -> Tuple[PointerPolicyNet, GatConvEmbedding]:
        return self.actor_critic.load_from_config(self.model_dir, run, device), \
               self.embedding_net.load_from_config(self.model_dir, run, device)
               
    def save_models(self, actor_critic: PointerPolicyNet, embedding_net: GatConvEmbedding,run: int):
        self.actor_critic.save_model(actor_critic, self.model_dir, run)
        self.embedding_net.save_model(embedding_net, self.model_dir, run)
    
    
class PPOAgentConfig(ConfigBase):
    def __init__(self, lr: float, gamma: float, step: int, gae_gamma: float, gae_lambda: float,
                 gae_alpha: float, embedding_type: str, clip_p: float, clip_v: float,
                 epochs: int, batch_size: int, e_loss_scale: float, v_loss_scale: float, 
                 tensorboard: bool, reward_norm: bool = True) -> None:
        super().__init__()
        self.lr = lr
        self.gamma = gamma 
        self.step =  step
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.gae_alpha = gae_alpha
        self.embedding_type = embedding_type
        self.clip_p = clip_p 
        self.clip_v = clip_v
        self.epochs = epochs
        self.batch_size = batch_size
        self.e_loss_scale = e_loss_scale
        self.v_loss_scale = v_loss_scale
        self.tensorboard = tensorboard
        self.reward_norm = reward_norm
        
    def generate_from_config(self, actor_critic, embedding_net) -> PPOAgent:
        return PPOAgent(actor_critic=actor_critic, embedding_net=embedding_net, 
                        **self.__dict__)
        
        
class RunConfig(ConfigBase):
    def __init__(self, num_runs: int, num_steps: int) -> None:
        super().__init__()
        self.num_runs = num_runs 
        self.num_steps = num_steps
            
            
class ExperimentBase(ConfigBase):
    def __init__(self, dataset: DatasetConfig, model: ModelConfig, run: RunConfig) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.run = run
        
    def _get_config_dir(self):
        return os.path.join(self.model.model_dir, "experiment.json")

    def exists(self):
        f = self._get_config_dir()
        exists, matches = False, False
        
        if os.path.exists(self.model.model_dir) and \
           os.path.exists(f) :
            try: 
                comp = type(self).from_file(f, to_object=False)
                src  = delete_test_info(self.__dict__) #Do not compare test info to allow a split evaluation
                                                       #using multiple json configs with a shared underlying model   
                diff = DeepDiff(src, comp, ignore_order=True)
                matches = diff == {}
                
                # Only print warning if num_runs < experiment
                if not matches and "values_changed" in diff.keys() and len(diff["values_changed"]) == 1:
                    key, values = list(diff["values_changed"].items())[0]
                    if key == "root['run']['num_runs']" and values["old_value"] < values["new_value"]:
                        print(f"Model directory contains more models than specified in experiment definition. Using first {values['old_value']}.")
                        matches = True
                
                exists = True
            except Exception as e: 
                print(e)
            
        if not exists:
            print("Starting model training")
        elif exists and matches:
            print("Skipping model training...")
        elif exists and not matches:
            raise ValueError("Trained model does not match the config.") 
    
        return exists and matches
    
    def print_message(self, exists):
        if exists:
            print("Skipping model training...")
        else:
            print("Starting model training...")
        
    @abstractmethod
    def train_model(self, device: str):
        raise NotImplementedError()   
    
    def generate_datasets(self, device: str = "cpu"):
        self.dataset.generate_from_config(device)
    
    def evaluate_model(self, device: str):
        print("Starting model evaluation...")
        for phantom in tqdm(self.dataset.iterate_phantoms()):
            results = dict()
            
            for events in tqdm(self.dataset.iterate_events()):
                pur_mus, pur_stds, eff_mus, eff_stds, rej_mus, rej_stds = [], [], [], [], [], []
                results[str(events)] = {"pur_mu": [], "pur_std": [], 
                                        "eff_mu": [], "eff_std": [],
                                        "rej_mu": [], "rej_std": []}
                for run in tqdm(range(self.run.num_runs)):
                    ac, en = self.model.load_from_config(run, device)
                    dataset = self.dataset.load_test_from_config(phantom, events, device)
                    results_df = get_metrics_for_dataset(dataset, ac, en, events)
                    
                    pur_mus += [results_df["pur"].mean()]; pur_stds += [results_df["pur"].std()]
                    eff_mus += [results_df["eff"].mean()]; eff_stds += [results_df["eff"].std()]
                    rej_mus += [results_df["rej"].mean()]; rej_stds += [results_df["rej"].std()]
                    
                    results_df.to_csv(os.path.join(self.model.model_dir, f"results_{phantom}_{events}_{run}.txt"))
                    
                results[str(events)]["pur_mu"] = np.array(pur_mus)
                results[str(events)]["pur_std"] = np.array(pur_stds)
                results[str(events)]["eff_mu"] = np.array(eff_mus)
                results[str(events)]["eff_std"] = np.array(eff_stds)
                results[str(events)]["rej_mu"] = np.array(rej_mus)
                results[str(events)]["rej_std"] = np.array(rej_stds)
        
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(self.model.model_dir, f"results_{phantom}.txt"))
            df.to_pickle(os.path.join(self.model.model_dir, f"results_{phantom}.pkl"))


class SVExperiment(ExperimentBase):
    def __init__(self, dataset: DatasetConfig, model: ModelConfig, run: RunConfig, 
                 optimizer: OptimizerConfig) -> None:
        
        super().__init__(dataset, model, run)
        self.optimizer = optimizer
        
    def train_model(self, device: str):
        exists = self.exists()
        self.print_message(exists)
        if exists: return
         
        # Lock directory to avoid starting multiple 
        # training runs for a shared model config.
        with LockDirectory(self.model.model_dir):        
            max_actions = self.dataset.train_events
            for run in tqdm(range(self.run.num_runs)):
                set_random_seeds(run)
                dataset = self.dataset.load_train_from_config(device)
                track_dataset = TrackDataset(dataset)
                ac, en = self.model.generate_from_config(device)
                params = chain(*[ac.parameters(), en.parameters()])
                optimizer = self.optimizer.generate_from_config(params)
                
                for i in range(self.run.num_steps):
                    optimizer.zero_grad()
                    env, minibatch = track_dataset.sample_batch(self.optimizer.batch_size)
                    _, log_prob, _ = evaluate_actions(env, ac, minibatch, max_actions, 
                                                    en, embedding_type="act")

                    loss = -(log_prob).mean()
                    loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(chain(*params), 0.5)
                    optimizer.step()

                self.model.save_models(ac, en, run)
            
            #Remove test info to allow a split evaluation
            #using multiple json configs with a shared underlying model
            with open(os.path.join(self.model.model_dir, "experiment.json"), "w") as f:
                f.write(self.to_json(del_test_info=True))

             
class RLExperiment(ExperimentBase):
    def __init__(self, dataset: DatasetConfig, model: ModelConfig,
                 agent: PPOAgentConfig, run: RunConfig) -> None:
        super().__init__(dataset, model, run)
        self.agent = agent
            
    def train_model(self, device: str):
        should_exit = self.exists()
        if should_exit: return
        
        # Lock directory to avoid starting multiple 
        # training runs for a shared model config.
        with LockDirectory(self.model.model_dir):
            
            run_logs = []
            max_actions = self.dataset.train_events
            for run in tqdm(range(self.run.num_runs)):
                set_random_seeds(run)
                dataset = self.dataset.load_train_from_config(device)
                ac, en = self.model.generate_from_config(device)
                agent = self.agent.generate_from_config(ac, en)
                _, log_dir = agent.train(dataset, self.run.num_steps, max_actions)
                run_logs += [log_dir]
                self.model.save_models(ac, en, run)
                
            runs_df = pd.DataFrame(run_logs, columns=["runs"])
            runs_df.to_csv(os.path.join(self.model.model_dir, f"run_logs.csv"))
                    
            #Remove test info to allow a split evaluation
            #using multiple json configs with a shared underlying model
            with open(os.path.join(self.model.model_dir, "experiment.json"), "w") as f:
                f.write(self.to_json(del_test_info=True))