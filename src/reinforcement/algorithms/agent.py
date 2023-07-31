from math import pi
from itertools import chain
from typing import Tuple, Union
from abc import abstractmethod, ABC

from src.reinforcement.episodes import ParallelEpisodes
from src.reinforcement.algorithms.advantage import GenAdvEstimator
from src.reinforcement.algorithms.common import explained_variance
from src.reinforcement.sampling import sample_parallel_trajectories, evaluate_actions
from src.reinforcement.environment import ParallelEnvironment, ParallelEnvironmentDataset

import torch
from torch import nn
from torch.optim.adam import Adam
from torch.nn import functional as F 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR


class BaseAgent(ABC):
    def __init__(self, actor_critic: Union[nn.Module, Tuple[nn.Module, nn.Module]], embedding_net: nn.Module, 
                 embedding_type: str, lr: float, gamma: float, step: int, tensorboard: Union[bool, str]):
        """Common shared implementation for all policy gradient based agents for 
        track reconstruction. 

        @param actor_critic: Actor/ actor_critic architecture containing either one of\
                             (shared -> actor  & critic, actor & critic, actor) networks.
        @param embedding_net: Embedding network for GNN based node embedding.
        @param tensorboard: Parameters for creating a tensorboard log (True -> default\
                            tensorboard log in runs, otherwise string containing path),\
                            defaults to None (no tensorboard log).
        """
        self.params = []
        self.actor_critic = actor_critic
        self.embedding_net = embedding_net
        self.embedding_type = embedding_type
      
        if isinstance(actor_critic, nn.Module):
            self.params = [self.actor_critic.parameters()]
        else:
            self.params = [self.actor_critic[0].parameters(), 
                           self.actor_critic[1].parameters()]
            
        if embedding_net is not None:
            self.params.append(embedding_net.parameters())
        
        self.optimizer = Adam(chain(*self.params), lr=lr)
        self.lr_scheduler = StepLR(self.optimizer, step, gamma)
            
        self.summary_writer = self._init_tensorboard(tensorboard)
        
            
    def _init_tensorboard(self, arg: Union[bool, str]) -> SummaryWriter:
        """Initializes a tensorboard SummaryWriter object for logging all 
        RL related metrics during training. Default path: runs/CURRENT_DATETIME_HOSTNAME

        @param arg: Initialization arguments, one of None (no tensorboard), True (tensor-
                    board log in default path), str (custom directory).
                    
        @return: Initialized Summary Writer object.
        """
        if arg is not None:
            return SummaryWriter(arg) \
                if isinstance(arg, str) \
                else SummaryWriter()
        else:
            return None
    
    @abstractmethod
    def train(self):
        pass


class PPOAgent(BaseAgent):
    def __init__(self, actor_critic: Union[nn.Module, Tuple[nn.Module, nn.Module]],
                 embedding_net: nn.Module, embedding_type: str, lr: float, 
                 gamma: float, step: int, clip_p: float,  clip_v: float,  epochs: int, 
                 batch_size: int, v_loss_scale: float, e_loss_scale: float, 
                 gae_gamma: float, gae_lambda: float, gae_alpha: float, reward_norm: bool,
                 tensorboard: Union[bool, str] = None):
        """Proximal Policy Optimization algorithm (PPO) (clip version) according to  https://arxiv.org/abs/1707.06347

        @param actor_critic: Actor critic architecture -> Output policy and value estimate.
        @param embedding_net: Embedding network architecture providing additional node embeddings 
                              given the underlying TGraph of the sampled environment.
        @param embedding_type: Type of embedding that should be used for actor_critic, 
                               either action embedding ("act"), observation embedding ("obs"), 
                               both ("both") or none
        @param lr: Learning rate used for optimizer.
        @param gamma: Learning rate decay.
        @param step: Number of training steps to be performed (each training step contains 
                     n sampled trajectories for an sampled environment).
        @param clip_p: Policy clipping parameter used in PPO.
        @param clip_v: Value function clipping parameter used in PPO.
        @param epochs: Number of epoch for each step when optimizing the surrogate loss.
        @param batch_size: Minibatch size used for calculating the loss for each optimization step.
        @param v_loss_scale: Value function scaling coefficient for the loss calculation.
        @param e_loss_scale: Entropy scaling coefficient for the loss calculation.
        @param gae_gamma: Discount factor used for GAE
        @param gae_lambda: Factor for trade-off of bias vs variance for GAE
        @param gae_alpha: Smoothing factor for exponential moving average used for reward norm in GAE.
        @param reward_norm: Whether to normalize sampled rewards.
        @param tensorboard: Optional path that should be used for tensorboard log, defaults to None
        """
        super().__init__(actor_critic, embedding_net, embedding_type, 
                         lr, gamma, step, tensorboard)
        
        self.clip_p = clip_p
        self.clip_v = clip_v
        self.epochs = epochs
        self.batch_size = batch_size
        self.v_loss_scale = v_loss_scale
        self.e_loss_scale = e_loss_scale
        self.reward_norm = reward_norm
        self.gae = GenAdvEstimator(gae_gamma, gae_lambda, gae_alpha, reward_norm=reward_norm)
        
        
    def train(self, data: Union[ParallelEnvironment, ParallelEnvironmentDataset], 
              iterations: int, max_actions: int):
        
        writer = SummaryWriter()
        step = 0
        
        if isinstance(data, ParallelEnvironment):
            env = data

        for _ in range(iterations):
            if isinstance(data, ParallelEnvironmentDataset):
                env: ParallelEnvironment = data.sample()
                env.last_n = 5

            with torch.no_grad():
                episodes = sample_parallel_trajectories(env, self.actor_critic, 
                                                        max_actions=max_actions,
                                                        embedding_net=self.embedding_net, 
                                                        embedding_type=self.embedding_type, 
                                                        no_grad=True)
            
            adv, ret, ret_raw = self.gae.calculate_returns_and_advantages(env, episodes)

            episodes.assign_returns(ret)
            episodes.assign_advantages(adv)
            
            writer.add_scalar('Total/return', ret_raw.mean(), step)
            
            for _ in range(self.epochs):
                for minibatch in episodes.iterate_minibatches_undersampled(env, self.batch_size):
                    self.optimizer.zero_grad()
                
                    values, log_probs, entropy = evaluate_actions(env, self.actor_critic, minibatch, 
                                                                  max_actions=max_actions,
                                                                  embedding_net=self.embedding_net, 
                                                                  embedding_type=self.embedding_type)
                                
                    ratio = torch.exp(log_probs - minibatch["log_probs"])
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_p, 1 + self.clip_p)
                    
                    nit = env.graph.next_is_tracker[minibatch["obs"]]
                    n_layer_types = len(nit.unique())
                    
                    # Policy loss calculation
                    policy_loss_1 = minibatch["advantages"] * ratio
                    policy_loss_2 = minibatch["advantages"] * clipped_ratio
                    policy_loss_min =  - torch.min(policy_loss_1, policy_loss_2)
                    
                    policy_loss_t = torch.nan_to_num(policy_loss_min[nit].mean(), 0.0)
                    policy_loss_n = policy_loss_min[~nit].mean()
                    
                    policy_loss = (policy_loss_t + policy_loss_n) / n_layer_types
                    
                    # Value clipping and value loss calculation
                    diff = values - minibatch["values"]
                    values_pred = minibatch["values"] + torch.clamp(diff, -self.clip_v, self.clip_v)
                   
                    value_loss_t = torch.nan_to_num(F.mse_loss(minibatch["returns"][nit], values_pred[nit]), 0.0)
                    value_loss_n = F.mse_loss(minibatch["returns"][~nit], values_pred[~nit])
                    value_loss = (value_loss_t + value_loss_n) / n_layer_types
                                  
                    entropy_loss_t = torch.nan_to_num(torch.mean(entropy[nit]), 0.0)
                    entropy_loss_n = torch.mean(entropy[~nit])       
                    entropy_loss = (entropy_loss_t + entropy_loss_n) / n_layer_types
                    
                    loss = policy_loss + self.v_loss_scale * value_loss - self.e_loss_scale * entropy_loss 

                    loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(chain(*self.params), 0.5)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    
                    # Clipping fraction and explained variance calculation for debugging purposes
                    clip_frac_policy = torch.mean((torch.abs(ratio - 1) > self.clip_p).float()).item()
                    clip_frac_value = torch.mean((torch.abs(values - minibatch["values"]) > self.clip_v).float()).item()
                    exp_variance = explained_variance(episodes)

            if self.summary_writer is not None:
                writer.add_scalar('Policy/loss', policy_loss, step)
                writer.add_scalar('Policy/loss_t', policy_loss_t, step)
                writer.add_scalar('Policy/loss_n', policy_loss_n, step)
                writer.add_scalar('Policy/clip', clip_frac_policy, step)
                writer.add_scalar('Policy/entropy', entropy_loss, step)
                writer.add_scalar('Policy/entropy_t', entropy_loss_t, step)
                writer.add_scalar('Policy/entropy_n', entropy_loss_n, step)
                writer.add_scalar('Critic/loss', value_loss, step)
                writer.add_scalar('Critic/loss_t', value_loss_t, step)
                writer.add_scalar('Critic/loss_n', value_loss_n, step)
                writer.add_scalar('Critic/estimate', values.mean(), step)
                writer.add_scalar('Critic/clip', clip_frac_value, step)
                writer.add_scalar('Total/loss', loss, step)
                writer.add_scalar('Total/exp_var', exp_variance, step)
                #GAE
                writer.add_scalar('GAE/mu_det', self.gae.mu_d1.value, step)
                writer.add_scalar('GAE/std_det', self.gae.std_d1.value, step)
                writer.add_scalar('GAE/mu_track', (self.gae.mu_t1.value + self.gae.mu_t2.value)/2, step)
                writer.add_scalar('GAE/std_track', (self.gae.std_t1.value + self.gae.std_t2.value)/2, step)
            
            step += 1
            
            writer.add_scalar('Total/learning_rate', self.lr_scheduler.get_last_lr()[0], step)
                
        return None, self.summary_writer.log_dir