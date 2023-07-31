from typing import Tuple, Union

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from src.reinforcement.environment import ParallelEnvironment
from src.reinforcement.episodes import ParallelEpisodes
from src.reinforcement.models import PointerPolicyNet
from src.utils.graph import get_act_features, get_obs_features
from src.utils.seeding import GroundTruthSeeding


def evaluate_policy(actor_critic: Union[nn.Module, Tuple[nn.Module, nn.Module]], 
                   obs_feat: torch.Tensor, act_feat: torch.Tensor, distances: torch.Tensor, 
                   act_mask: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the given actor_critic network for an given input and calculate policy
    probabilities and values.

    @param actor_critic: Actor critic network. (Either joint model architecture 
                         or two models of type nn.Module)
    @param obs_feat: Current observation feature batch.
    @param act_feat: Current action feature batch.
    @param distances: Distances between current track hit and next track candidates.
    @param act_mask:Masking vector of shape [n_batch, max_actions] where each value of 
                     batch dim is set to true if it doesn't contain an actual action and 
                     is only padded to match the tensor dim.
    
    @returns: Evaluated policy probabilities and values.
    """
    policy, values, hidden = actor_critic(obs_feat, act_feat, distances, act_mask, hidden)
    return policy, values, hidden


def get_actions_from_policy(dist: Categorical, stochastic: bool):
    """Evaluates an action given a defined categorical distribution
    defined by the underlying policy (pi) based on the specified 
    strategy (random, best action)

    @param dist Categorical distribution over all actions defined 
                 by the underlying policy
    @param stochastic: Determines whether the policy should be 
                       evaluated as a stochastic policy (random)
    
    @returns: Selected action
    """
    return dist.sample() if stochastic \
                         else torch.argmax(dist.probs, dim=1)
                         

def evaluate_actions(env: ParallelEnvironment, actor_critic: PointerPolicyNet,  minibatch: dict, 
                     max_actions: int, embedding_net: nn.Module = None, embedding_type: str = 'both') \
                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a batch of n sequences of sampled trajectories determining the values, log_probs
    and entropies of the individual actions.

    @param actor_critic: Actor critic architecture either shared or as tuple (actor, critic)
    @param minibatch: Dictionary containing a minibatch of collected experience
    @param embedding_net: Graph neural network used for node embedding
    @param embedding_type: Type of the embedding (either 'obs' or 'act' or 'none')

    @returns: Tuple of values, log_probs and entropies
    """
    max_actions = env.graph.edges_per_node.max()
    obs_emb, act_emb = get_embeddings(env, embedding_net, embedding_type)
    old_obs, obs, act = minibatch["old_obs"], minibatch["obs"], minibatch["act"]
    old_obs_pos = env.graph.pos[old_obs]
    
    obs_feat, obs_pos = get_obs_features(env.graph, old_obs, obs, obs_emb)
    act_feat, act_pos, act_mask = get_act_features(env.graph, obs, max_actions, act_emb)
    
    hidden = None if not actor_critic.pomdp else minibatch["hidden"]
    
    similarities = get_cosine_similarity(old_obs_pos, obs_pos, act_pos)
    
    policy, values, _ = evaluate_policy(actor_critic, obs_feat, act_feat, 
                                     similarities, act_mask, hidden)
                           
    dist = Categorical(policy)
    
    return values, dist.log_prob(act), dist.entropy()


def get_embeddings(env: ParallelEnvironment, embedding_net: nn.Module, embedding_type: str)\
    -> Tuple[torch.Tensor, torch.Tensor]:
    """[summary]

    @param env: Parallel environment determining the transition dynamics of the system
    @param embedding_net: Graph neural network used for node embedding
    @param embedding_type: Type of the embedding (either 'obs' or 'act' or 'none')
    
    @returns: Tuple containing specific embeddings (either embeddings or None for 
              observation and action embedding)
    """
    if embedding_net is None:
        return None, None
    
    embeddings = embedding_net(env.graph.x, env.graph.edge_index, env.graph.edge_attr)
    obs_emb = embeddings if embedding_type in ['both', 'obs'] else None
    act_emb = embeddings if embedding_type in ['both', 'act'] else None
    
    return obs_emb, act_emb


def expand_seeding_mask(seed: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand the seeding mask (one element per observation) to match the 
    total number of elements after seeding with k seeds per observation.
    repeats = (1 if no seeding required, k otherwise)

    @param seed: Seeding mask for observations as defined by environment.
    @param k: Number of initial seeds per observation.
    
    @returns: Tensor containing repeats per observation and expanded seeding mask.
    """
    repeats = seed.type(torch.long) * k + (~seed).type(torch.long)
    return repeats, seed.repeat_interleave(repeats)


def get_cosine_similarity(last: torch.Tensor, current: torch.Tensor, next: torch.Tensor):
    """Calculates cosine similarity between possible track segments (last -> current, 
    current -> next) used for similarity encoding.

    @param last: Position of previously reconstructed [batch, 3].
    @param current: Position of current reconstructed [batch, 3].
    @param next: Positions of next hit candidates [batch, n_hits, 3].

    @returns: Cosine similarities [batch, 1].
    """
    x0_x1 = current[:,None,:] - last[:,None,:]
    x1_x2 = next - current[:,None,:]

    return F.cosine_similarity(x0_x1, x1_x2, dim=-1)

def expand_mask_for_k_seeds(seed_expand: torch.Tensor, mask: torch.Tensor):
    """Returns new expanded mask, where elements for new tracks (k-seeds) are 
    expanded in new masking tensor.

    @param seed_expand: _description_
    @param mask: _description_
    
    @return: _description_
    """
    exp_mask = torch.zeros_like(seed_expand).type(torch.bool)
    exp_mask[~seed_expand] = mask
    return exp_mask


def configure_sampling(evaluate: bool, actor_critic: nn.Module, embedding_net: nn.Module) -> None:
    """Enables/disables eval mode of network architecture based on evaluate variable

    @param evaluate: Determines which mode (eval/train) should be enabled. 
    @param actor_critic: Actor-critic network architecture
    @param embedding_net: Embedding network architecture
    """
    if evaluate:  
        actor_critic.eval()
        embedding_net.eval()
    else:
        actor_critic.train()
        embedding_net.train()
        

def sample_parallel_trajectories(env: ParallelEnvironment, actor_critic: PointerPolicyNet, 
    max_actions: int, embedding_net: nn.Module = None,  embedding_type: str = 'act', 
    stochastic: bool=True, evaluate: bool = False, no_grad=True) -> ParallelEpisodes:
    """ Sample multiple parallel trajectory rollouts (track candidates) using the current
    stochastic policy for determining the transition probabilities for each time step.

    @param env: Parallel environment determining the transition dynamics of the system.
    @param actor_critic: Torch actor critic model architecture.
    @param embeddings: Tensor containing gnn embeddings for every node in the graph, defaults to None.
    @param embedding_type: Type of the embedding (either 'obs' or 'act' or 'both), defaults to 'both'.
    @param stochastic: Determines whether the selection of an action should
                       be stochastic based on the probabilities determined 
                       by the policy pi, defaults to True.
    @param evaluate: Enable evaluation mode of the environment. All initial nodes are selected 
                     for sampling parallel trajectories, defaults to False.
    @no_grad: Disable gradient calculation for trajectories, defaults to True.

    @returns: N sampled trajectories if evaluate = False (determined by the number of instances 
              of the parallel environment) or all reconstructed trajectories.
    """
    configure_sampling(evaluate, actor_critic, embedding_net)
    
    old_obs, old_obs_pos = None, torch.zeros_like(env.graph.pos[0]).unsqueeze(0)
    max_actions = env.graph.edges_per_node.max()
    seeding = GroundTruthSeeding(env.graph)
    
    with torch.set_grad_enabled(not no_grad):
        obs, seed, done = env.reset(evaluate) # sample random if not eval mode
        obs_emb, act_emb = get_embeddings(env, embedding_net, embedding_type)
        
        episodes = ParallelEpisodes() 
        
        hidden_shape = (obs.shape[0], 2, actor_critic.embedding_size)
        hidden = None if not actor_critic.pomdp else torch.zeros(hidden_shape, device=obs.device)
        hid = None
        
        while not all(done):
            # Repeat all seeded components k times for k seeds    
            obs_feat, obs_pos = get_obs_features(env.graph, old_obs, obs[~seed], obs_emb)
            act_feat, act_pos, act_mask = get_act_features(env.graph, obs[~seed], max_actions, act_emb)
            
            # Mask input (mask == False) if track is finished or requires seeding!
            network_mask = torch.logical_and(torch.sum(act_mask, dim=1) < max_actions, ~done[~seed])
            similarities = get_cosine_similarity(old_obs_pos, obs_pos, act_pos)
            repeats, seed_expand = expand_seeding_mask(seed, seeding.k)
            
            # Combined masking (only update values covered by policy) -> removes
            # seeding and finished tracks
            exp_network_mask = expand_mask_for_k_seeds(seed_expand, network_mask) 
            done_expand = expand_mask_for_k_seeds(seed_expand, done[~seed])
            comb_mask = torch.logical_and(~seed_expand, exp_network_mask)

            hidden_repeat = (1, 2, actor_critic.embedding_size)
            actions = torch.empty_like(seed_expand, dtype=torch.long).fill_(-1)
            log_probs = torch.zeros_like(seed_expand, dtype=torch.float)
            values = torch.zeros_like(seed_expand, dtype=torch.float)
            entropy = torch.zeros_like(seed_expand, dtype=torch.float)
            
            if actor_critic.pomdp:
                hid = hidden.clone()
                hidden = torch.zeros_like(seed_expand[:,None,None].repeat(hidden_repeat), dtype=torch.float)
            
            
            # Skip policy evaluation if all tracks require seeding.
            if (~seed).any() and network_mask.any():
                hid = hidden if hidden == None else hidden[exp_network_mask]
                policy, val, hid = evaluate_policy(actor_critic, obs_feat[network_mask], act_feat[network_mask], 
                                              similarities[network_mask], act_mask[network_mask], hid)
                    
                dist = Categorical(policy) 
                act_policy = get_actions_from_policy(dist, stochastic)
                
                log_probs[comb_mask] = dist.log_prob(act_policy)
                entropy[comb_mask] = dist.entropy()
                actions[comb_mask] = act_policy
                
                if actor_critic.pomdp:
                    hidden[comb_mask] = hid
                
                if not evaluate:
                    values[comb_mask] = val

            if seed.any():
                if env.current_step == 0:
                    act_policy = None

                act_seeding = seeding.get_actions(obs, seed, act_policy, done)
                actions[seed_expand] = act_seeding
                              
            # Repeat observations for seeded tracks (k-times)
            old_obs = obs.repeat_interleave(repeats)
            old_obs_pos = env.graph.pos[old_obs]
            
            episodes.update(old_obs, actions, log_probs, 
                            entropy, values, hid, done_expand, seed_expand)

            current_layer = env.graph.num_layers - env.current_step 
            add_missing = (evaluate and current_layer > 4)
            obs, seed, done = env.step(actions, seeding.k, add_missing)
        
        episodes.finish(obs)
        
    return episodes