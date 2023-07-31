import math

import torch
from torch import nn


def masked_softmax(x: torch.Tensor, mask: torch.Tensor = None, dim: int = -1) -> torch.Tensor:
    """Masked version of the torch.softmax functionality to ignore masked actions in 
    the final output of the softmax operation.

    @param x: Input tensor
    @param mask: Masking vector that defines padded actions, defaults to None

    @return: Masked softmax of the input vector.
    """
    if mask is None:
            return torch.softmax(x, dim=dim)
        
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x-x_max)
    x_exp = x_exp * (mask == False).float().to(x.device) # this step masks
    return x_exp / torch.sum(x_exp + 1e-12, dim=dim, keepdim=True) # +1e-12 avoid nan for zero rows


def masked_mean(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Masked version of the torch.mean functionality to ignore masked actions in 
    the final output of the mean operation (sum(non_masked)/#non_masked).

    @param x: Input tensor
    @param mask: Masking vector that defines padded actions, defaults to None

    @return: Masked mean of the input vector.
    """
    if mask is None:
        return torch.mean(x, dim=-1)
    
    x = torch.sum(x * (~mask).type_as(x), dim=-1)
    y = torch.sum((mask == False), dim=-1)
    return x / y

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        """Scaled dot-product attention with additional softmax 
        activation (for scaling) as described in [Vaswani et al., 2017]

        @param dim_in: Dimensionality of the input features
        @param dim_out: Dimensionality of the output features
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        self.mha = nn.MultiheadAttention(embed_dim, 1, batch_first=True, bias=True, add_bias_kv=False)
        
        self.layer_norm0 = nn.LayerNorm(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=True), nn.ReLU(),
                                nn.Linear(embed_dim, embed_dim, bias=True))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the scaled dot-product attention producing
        both attentive output [n_batch, n_actions, out_dim] and attention 
        weights [n_batch, out_dim, out_dim] by processing input x of shape
        [n_batch, n_actions, dim_in]

        @param x: Input tensor [n_batch, n_actions, dim_in].

        @returns: Scaled-dot product attention [n_batch, n_actions, out_dim] 
                 and corresponding attention weights [n_batch, n_actions, dim_in].
        """
        if mask is not None: 
            x[mask,:] = 0.0
            
        x = self.layer_norm0(x)
        x_attn, attn_weights = self.mha(x, x, x)
        x_attn = self.layer_norm1(x_attn + x)  
        x_attn = self.layer_norm2(self.ff(x_attn) + x_attn)

        return x_attn, attn_weights

    
class MaskedAdditiveAttention(nn.Module):
    def __init__(self, dim_in: int):
        """Additive attention mechanism [Bahdanau et al. 2015] used 
        as the final layer in the pointer network paper [Vinyals el al., 2015]
        extended by an masking mechanism for masked actions.

        @param input_size: Input feature size of the incoming data.
        """
        super().__init__()
        self.embedding_size = dim_in
        self.V = nn.Linear(dim_in, 1, bias=True)
        self.W1 = nn.Linear(dim_in, dim_in, bias=True)
        self.W2 = nn.Linear(dim_in, dim_in, bias=True)
        self.tanh = nn.Tanh()


    def forward(self, e: torch.Tensor, d: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the masked attention layer. Each attention 
        weight of the masked input is set to zero.

        @param d: First input to the attention mechanism. (in PointNet paper: decoder)
        @param e: Second input to the attention mechanism. (in PointNet paper: encoder)
        @param mask: Masked as defined by action tensor, defaults to None.
        
        @returns: Masked attention weights.
        """
        w1e: torch.Tensor = self.W1(e)
        w2d: torch.Tensor = self.W2(d)
        w2d = w2d.unsqueeze(1)
        
        tanh: torch.Tensor = self.tanh(w1e + w2d)
    
        attn_weights: torch.Tensor = self.V(tanh)
        attn_weights = attn_weights.squeeze(dim=-1)
        
        if mask is not None:
            attn_weights = torch.masked_fill(attn_weights, mask, 0.0)
            
        return attn_weights


class PE_ARF(nn.Module):
    def __init__(self, d_model: int, scaling: int = 100):
        super().__init__()
        self.s = scaling
        self.d_model = d_model
        self.attn = MaskedAdditiveAttention(d_model)
        self.W1 = nn.Linear(d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, 1), nn.Sigmoid())
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    
    def forward(self, sim: torch.Tensor, ho: torch.Tensor, 
                ha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns the positional encoding vector for an input feature x.

        @param x: Input feature x [Tensor]
        @return: Input feature x with positional encoding [Tensor]
        """
        w_i = masked_softmax(self.attn(ha, ho, mask), mask).unsqueeze(-1)
        c = (w_i * (self.W1(ha) + self.W2(ho.unsqueeze(1)))).sum(dim=1)
        q = torch.clip(self.fc(c), 1e-3, 1)
        sim = 1 - (sim + 1)/2
        sim = torch.clip(sim/q, 0, 1) * self.s
        
        pe = torch.empty_like(ho[:,None,:].repeat(1, sim.shape[-1], 1))
        pe[..., 0::2] = torch.sin(sim[:,:,None] * self.div_term[None, None, :]) 
        pe[..., 1::2] = torch.cos(sim[:,:,None] * self.div_term[None, None, :]) 
        
        return pe, q
    
    
class PE(nn.Module):
    def __init__(self, d_model: int, scaling: int = 100):
        super().__init__()
        self.s = scaling
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    
    def forward(self, sim: torch.Tensor, ho: torch.Tensor, 
                ha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns the positional encoding vector for an input feature x.

        @param x: Input feature x [Tensor]
        @return: Input feature x with positional encoding [Tensor]
        """
        sim = 1 - (sim + 1)/2 * self.s
        
        pe = torch.empty_like(ho[:,None,:].repeat(1, sim.shape[-1], 1))
        pe[..., 0::2] = torch.sin(sim[:,:,None] * self.div_term[None, None, :]) 
        pe[..., 1::2] = torch.cos(sim[:,:,None] * self.div_term[None, None, :]) 
        
        return pe, None


class PointerPolicyNet(nn.Module):
    def __init__(self, obs_size: int, act_size: int, embedding_size: int, sim_scaling: int = 100, 
                 pomdp: bool = False, use_act_enc: bool = True, enc_type: str = "PE-ARF"):
        """PointerPolicy actor critic network that processes inputs sampled 
        from the environment and produces a policy output pi (actor) and the 
        value function V (critic).

        @param obs_size: Size (dimensionality) of the incoming observation features
        @param act_size: Size (dimensionality) of the incoming action features
        @param embedding_size: Size (dimensionality) of the embedding
        @param sim_scaling: Scaling factor for similarity encoding N = floor(sim * s), defaults to 100
        @param use_act_enc: Defines whether encoding mechanism should be used for 
                            action features, defaults to True
        @param enc_type: Specification of encoding mechanism that should be used for 
                         positional encoding, one of ("PE-ARF", "PE")
        """
        super().__init__()
        # Size parameters
        self.obs_size = obs_size
        self.action_size = act_size
        self.embedding_size = embedding_size
        self.sim_scaling = sim_scaling
        self.use_act_enc = use_act_enc
        self.enc_type = enc_type
        self.pomdp = pomdp
        
        self.attn_weights = []
        self.q = None
        
        # Observation embedding
        if not pomdp:
            self.obs_embedding = nn.Sequential(nn.Linear(obs_size, embedding_size), nn.ReLU(),
                                               nn.Linear(embedding_size, embedding_size))
        else:
            self.lstm_obs_embedding = nn.LSTMCell(obs_size, embedding_size)
            self.obs_embedding = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(),
                                               nn.Linear(embedding_size, embedding_size))

        # Action embedding
        self.act_embedding = nn.Sequential(nn.Linear(act_size, embedding_size), nn.ReLU(),
                                           nn.Linear(embedding_size, embedding_size)) 

        # Positional encoding
        self.pos_encoding = PE_ARF(embedding_size, scaling=sim_scaling + 1) \
                            if enc_type == "PE-ARF" \
                            else PE(embedding_size, scaling=sim_scaling + 1)
        
        # Shared network branch
        self.sdpa1 = TransformerEncoder(embedding_size)
        self.sdpa2 = TransformerEncoder(embedding_size)
        self.sdpa3 = TransformerEncoder(embedding_size)
    
        self.attention_pi = MaskedAdditiveAttention(embedding_size)
        self.attention_v = MaskedAdditiveAttention(embedding_size)

        for m in self.modules():
            self._init_weights(m)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=torch.rand([1]))


    def forward(self, obs: torch.Tensor, act: torch.Tensor, sim: torch.Tensor,
                padding_mask: torch.Tensor = None, hidden = None) -> torch.Tensor:
        """Forward pass of the pointer policy net. Takes a single set ob observation features 
        of shape [n_batch, n_features] and action features of shape [n_batch, max_actions, n_features]
        together with a batch tensor of size [n_batch * n_actions] and padding mask of shape
        [n_batch, max_actions] and produces both an probability vector of shape [n_batch, max_actions] 
        as well as values for each element of the batch of shape [n_batch, 1].

        @param obs: Set of observation features of size [n_batch, n_features] sampled from 
                    the reinforcement learning environment
        @param act: Tensor containing all action features corresponding to the possible node 
                    transitions given the current observation.
        @param sim: Cosine similarities between current track segment and next track candidates 
                     [n_batch, max_actions].
        @param padding_mask: Masking vector of shape [n_batch, max_actions] where each value of 
                     batch dim is set to true if it doesn't contain an actual action and 
                     is only padded to match the tensor dim., defaults to None
        
        @return: Returns policy pi as a probability vector of shape [n_batch, max_actions]
                 and value function (V) of shape [n_batch, 1].
        """
        # Feature expansion for action and observation feature
        # + mapping to same dimensionality
        act_emb = self.act_embedding(act)
        
        if not self.pomdp:
            obs_emb = self.obs_embedding(obs)
        else:
            h1, c1 = self.lstm_obs_embedding(obs, (hidden[:,0,:], hidden[:,1,:]))
            hidden = torch.stack((h1, c1), dim=1)
            obs_emb = self.obs_embedding(h1)

        enc, self.q = self.pos_encoding(sim, obs_emb, act_emb, padding_mask)
        
        if self.use_act_enc:
            act_emb = act_emb + enc
        
        act_emb, w1 = self.sdpa1(act_emb, padding_mask)
        act_emb, w2 = self.sdpa2(act_emb, padding_mask)
        act_emb, w3 = self.sdpa3(act_emb, padding_mask)
        
        self.attn_weights = [w1, w2, w3]

        attn_weights_pi = self.attention_pi(act_emb, obs_emb, padding_mask)
        pi = masked_softmax(attn_weights_pi, padding_mask)

        attn_weights_v = self.attention_v(act_emb, obs_emb,padding_mask)
        v  = masked_mean(attn_weights_v, padding_mask)

        return pi, v, hidden