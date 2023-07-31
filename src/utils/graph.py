from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.nn import functional as F
from torch_geometric import transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from src.utils import transforms as CT
from src.utils.pandas import is_preprocessed, preprocess, remove_disconnected


class TGraph(Data):
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor, 
                 y: torch.Tensor, is_tracker: torch.Tensor, next_is_tracker: torch.Tensor,
                 edep: torch.Tensor, z: torch.Tensor, skip_tracker: bool, to_wet: bool,
                 is_primary: torch.Tensor, spot_x: float = None, spot_y: float = None) -> None:
        """Use from_csv or from_df for creating a new TGraph object!"""
        super().__init__(x=x, edge_index=edge_index, pos=pos, y=y, edep=edep)
        
        self._spot_x = spot_x
        self._spot_y = spot_y 
        self._edep = edep
        self._to_wet = to_wet
        self._is_primary = is_primary
        self._is_tracker = is_tracker
        self._next_is_tracker = next_is_tracker
        self._z = z
        self._contains_tracker = not skip_tracker
        _, c =self.z.unique(return_counts=True)
        self._layerwise_node_count = c.flip(dims=(0, ))
        edge_index_attr = torch.arange(self.edge_index.shape[1])
        self._edge_adjacency_csr = to_scipy_sparse_matrix(self.edge_index, edge_index_attr).tocsr()

        self._edges_per_node = torch.zeros_like(x[:,0], dtype=torch.long)
        _, x = torch.unique(edge_index[0], return_counts=True)
        self._edges_per_node[:len(x)] = x
        self._epn_zero_cumsum = F.pad(self._edges_per_node, (1, 1), 'constant', 0).cumsum(dim=-1)
    
    @classmethod
    def from_csv(cls, path: str, skip_tracker: bool = False, to_wet: bool = False, 
                 filter_secondaries: bool = False, cluster_threshold: float = None,
                 device: str = "cpu"): 
        """Generate a Trajectory Graph using a csv file containing all detector 
        information as an input.

        @param path: Filepath to *.csv file containing the detector readout data.
        @param skip_tracker: Determines whether the tracking layers
                             should be skipped, defaults to False
        @param to_wet: Determines whether the relative z positions should
                       be converted to water equivalent thicknesses, defaults to False
        @param filter_secondaries: Determines whether secondary particles should be 
                                   filtered and removed from the TGraph (USING GROUND TRUTH !!!),  
                                   defaults to False
        @param cluster_threshold: Cluster size threshold which is required in order to consider 
                                  hit as primary particle. 
        @param device: The desired computational device of returned graph object.
        """
        return TGraph.from_df(pd.read_csv(path), skip_tracker, to_wet, 
                              filter_secondaries, cluster_threshold, device)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame, skip_tracker: bool = False, 
                to_wet: bool = False, filter_secondaries: bool = False,
                cluster_threshold: float = None, device: str = "cpu"):
        """Generate a Trajectory Graph using a dataframe with all detector 
        information as an input.

        @param df: Dataframe containing the detector readout data.
        @param skip_tracker: Determines whether the tracking layers
                             should be skipped, defaults to False
        @param to_wet: Determines whether the relative z positions should
                       be converted to water equivalent thicknesses, defaults to False
        @param filter_secondaries: Determines whether secondary particles should be 
                                   filtered and removed from the TGraph (USING GROUND TRUTH !!!), 
                                   defaults to False
        @param cluster_threshold: Cluster size threshold which is required in order to consider 
                                  hit as primary particle. 
        @param device: The desired computational device of returned graph object.
        """

        def error_message_col(name: str):
            return f"Dataframe should contain {name} column"

        assert "edep" in df.columns, error_message_col("edep")
        assert "posX" in df.columns, error_message_col("posX")
        assert "posY" in df.columns, error_message_col("posY")
        assert "posZ" in df.columns, error_message_col("posZ")
        
        # Removes all particle that are not primary particles 
        if filter_secondaries and "parentID" in df.columns:
            df = df[df.parentID == 0]
        
        if not is_preprocessed(df):
            df = preprocess(df)

        if cluster_threshold is not None:
            df = df[df.cluster >= cluster_threshold]
        
        df = remove_disconnected(df)
        
        if skip_tracker:
            df = df[df.z >= 2] 
        
        df_sorted = df.sort_values('z', ascending=False)
        
        edep = cls.extract_edep_df(df_sorted, device)
        y, is_tracker, next_is_tracker, is_primary = cls.extract_ground_truth_df(df_sorted, device)
        x = cls.extract_node_features_df(df_sorted, pixel=True, device=device)
        z = cls.extract_z_index_df(df_sorted, device=device)
        pos = cls.extract_node_positions_df(df_sorted, pixel=False, device=device)
        edge_index = cls.generate_edge_index(df_sorted, device)
        
        spot_x, spot_y = None, None
        if "spotX" in df_sorted.columns and "spotY" in df_sorted.columns:
            assert len(df_sorted["spotX"].unique()) == 1, "Spot position must be unique."
            assert len(df_sorted["spotY"].unique()) == 1, "Spot position must be unique."
            spot_x = df_sorted["spotX"].iloc[0]
            spot_y = df_sorted["spotY"].iloc[0]
          
        transform = T.Compose([CT.SphericalInverse(norm=False), CT.InvariantNorm()])
        return transform(TGraph(x=x, edge_index=edge_index, pos=pos, y=y,
                                is_tracker=is_tracker, next_is_tracker=next_is_tracker,
                                edep=edep, z=z, skip_tracker=skip_tracker, to_wet=to_wet, 
                                is_primary=is_primary, spot_x=spot_x, spot_y=spot_y))
    
    @classmethod
    def generate_edge_index(cls, df: pd.DataFrame, device: str) -> torch.Tensor:
        """Generates the edge index for the track data connecting the one-hop neighborhood 
        between two layers (fully connected).

        @param df: Sorted dataframe (ascending, by z) containing all particle hits.
        @param device: The desired computational device of returned graph object.
        
        @return: Generated edge_index tensor [2, n_edge_connections]
        """    
        num_nodes_layerwise = torch.tensor(df.groupby("z").size(), device=device).flip(dims=(0,))

        repeats_edges = num_nodes_layerwise[1:].repeat_interleave(num_nodes_layerwise[:-1])

        nodes_from_unrepeated = torch.arange(0, num_nodes_layerwise[:-1].sum(), device=device)
        edges_from = nodes_from_unrepeated.repeat_interleave(repeats_edges)

        edges_to = torch.arange(num_nodes_layerwise[0], len(edges_from) + num_nodes_layerwise[0], device=device)

        edge_offset = repeats_edges.cumsum(dim=0).repeat_interleave(repeats_edges) - repeats_edges.repeat_interleave(repeats_edges)
        layerwise_node_offset = F.pad(num_nodes_layerwise[1:], pad=(1, 0))[:-1].cumsum(dim=0)\
                                .repeat_interleave(num_nodes_layerwise[:-1])\
                                .repeat_interleave(repeats_edges) 
        edges_to = edges_to - edge_offset + layerwise_node_offset

        edge_index = torch.stack([edges_from, edges_to])
                
        return edge_index
    
    @classmethod
    def extract_node_features_df(cls, df_sorted: pd.DataFrame, pixel: bool = False,
                                 device: str = "cpu") -> torch.Tensor:
        """Extracts the required node features from pandas dataframe and 
        calculates additional features (v), requires the following columns: `edep`.
        
        @param df_sorted: sorted dataframe (ascending, by z)
        @param device: The desired computational device of returned graph object.
        
        @returns node_feature tensor
        """
        edep = cls.extract_cluster_df(df_sorted, device)
        pos = cls.extract_node_positions_df(df_sorted, pixel=pixel, device=device)
        z_indices = cls.extract_z_indices(df_sorted, device)
        return torch.cat([edep.unsqueeze(-1), pos[:,:-1], F.one_hot(z_indices, 50)], dim=1)
    
    @classmethod 
    def extract_z_index_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:  
        return torch.tensor(df_sorted.z.values, dtype=torch.float32, device=device)
    
    @classmethod 
    def extract_edep_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:  
        return torch.tensor(df_sorted.edep.values, dtype=torch.float32, device=device)

    @classmethod
    def extract_cluster_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        return torch.tensor(df_sorted.cluster.values, dtype=torch.float32, device=device)

    @classmethod
    def extract_z_indices(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        return torch.tensor(df_sorted.z.values, dtype=torch.long, device=device)
    
    @classmethod
    def extract_ground_truth_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        """Extracts the track information (eventID) form pandas dataframe as 
        ground truth (y) data, requires `eventID` column:

        @param df_sorted: sorted dataframe (ascending, by z)
        @param device: The desired computational device of returned graph object.
        
        @returns ground truth (y) tensor
        """
        event_id = None if "eventID" not in df_sorted.columns \
                        else torch.tensor(df_sorted.eventID.values, 
                                          dtype=torch.long, device=device)
                        
        is_primary = None if "parentID" not in df_sorted.columns \
                          else torch.tensor(df_sorted.parentID.values == 0,
                                            device=device)
        
        is_tracker = torch.tensor(df_sorted.z.values < 2, device=device)
        next_is_tracker = torch.tensor(df_sorted.z.values < 3, device=device)
        
        return event_id, is_tracker, next_is_tracker, is_primary
    
    
    def to(self, device: Union[int, str], *args: List[str], non_blocking: bool = False):
        self._edep = self._edep.to(device)
        self._is_tracker = self._is_tracker.to(device)
        self._next_is_tracker = self._next_is_tracker.to(device)
        self._edges_per_node = self._edges_per_node.to(device)
        self._epn_zero_cumsum = self._epn_zero_cumsum.to(device)
        return super().to(device, *args, non_blocking=non_blocking)
            
    
    @classmethod
    def extract_node_positions_df(cls, df_sorted: pd.DataFrame, pixel: bool = False,
                                  device: str = "str") -> torch.Tensor:
        """Extracts the required node positions from pandas dataframe,
        requires the following columns: `posX, posY posZ`
        
        @param df_sorted: sorted dataframe (ascending, by z)
        @param device: The desired computational device of returned graph object.
        
        @returns node_position tensor
        """
        if pixel:
            return torch.stack([torch.tensor(df_sorted.x.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.y.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.z.values, dtype=torch.float32, device=device)]).T
        else:
            return torch.stack([torch.tensor(df_sorted.posX.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.posY.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.posZ.values, dtype=torch.float32, device=device)]).T
        
    @property
    def edge_adjacency_csr(self) -> csr_matrix:
        return self._edge_adjacency_csr
    
    @property 
    def edep(self) -> torch.Tensor:
        return self._edep
    
    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @property
    def is_primary(self) -> torch.Tensor:
        return self._is_primary
    
    @property
    def is_tracker(self) -> torch.Tensor:
        return self._is_tracker
    
    @property
    def next_is_tracker(self) -> torch.Tensor:
        return self._next_is_tracker
    
    @property
    def num_nodes_layerwise(self):
        return self._layerwise_node_count
    
    @property
    def num_layers(self):
        return len(self._layerwise_node_count)
    
    @property
    def edges_per_node(self):
        return self._edges_per_node
    
    @property
    def epn_zero_cumsum(self):
        return self._epn_zero_cumsum
    
    @property
    def contains_tracker(self):
        return self._contains_tracker    
    
    @property
    def to_wet(self):
        return self._to_wet    
    
    @property
    def spot_x(self):
        return self._spot_x
    
    @property 
    def spot_y(self):
        return self._spot_y
    

def get_obs_features(graph: Data,
                     last: Union[int, torch.Tensor], current: Union[int, torch.Tensor],
                     embeddings: torch.Tensor = None) -> torch.Tensor:
    """Returns the features for 1 to n observations determined by the node features
    of the current nodes and the edge features connecting the last and current nodes.

    @param env: Reinforcement learning environment containing trajectory graph that 
                describes the current  reconstruction environment.
    @param last: Index/ indices of the last selected node/s (None if no previous node exists).
    @param current: Index/ indices of the currently selected node/s.
    @param embedding: Tensor containing graph embeddings.

    @returns: Tensor containing the features for the determined observations (n_batch, n_feat).
    """
    if isinstance(current, int) or len(current.shape) == 0:
        current = torch.tensor([current], dtype=torch.long, device=graph.x.device)
        if last is not None:
            last = torch.tensor([last], dtype=torch.long, device=graph.x.device)
        
    def get_edge_features(f, t):
        rows = graph.edge_adjacency_csr[f.cpu()]
        indices_mat = rows[:, t.cpu()]
        indices = indices_mat.diagonal()
        return graph.edge_attr[indices]
        
    obs_node = graph.x[current]
    act_embedding = torch.empty((obs_node.shape[0], 0), device=obs_node.device) \
                    if embeddings is None \
                    else embeddings[current]
    obs_edge = get_edge_features(last, current) \
                    if last is not None \
                    else torch.zeros((obs_node.shape[0], graph.num_edge_features), 
                                     device=obs_node.device)
                    
    return torch.cat((obs_node, obs_edge, act_embedding), axis=1), graph.pos[current]


def get_act_features(graph: Data, obs: torch.Tensor, max_actions: int, 
                     embeddings: torch.Tensor = None) -> torch.Tensor:
    """Returns the features for a batch of 1 to n with m actions each determined
    by the node features of the current nodes and the node indices of possible actions.
    determined by the neighborhood of the current node/s.
    
    @param env: Reinforcement learning environment containing trajectory graph that describes the current 
                reconstruction environment and lookup object for faster edge queries.
    @param obs: Index/ indices of the last selected node/s (None if no previous node exists).
    @param max_actions: Maximum number of possible actions (graph) for defining batch dimensions.
    @param embedding: Tensor containing graph embeddings.
    
    @return: Tensor containing the features for the determined observations (n_batch, n_feat).
    """
    if isinstance(obs, int): obs = torch.tensor([obs], device=graph.x.device, dtype=torch.long)
    
    counts = graph.edges_per_node[obs.type(torch.long)]
    row = torch.arange(obs.shape[0], device=obs.device)

    row_indices, col_indices = to_row_col_indices(row, counts)
    #epn_zero_cumsum[obs] = offset (starting idx) in edge index
    edge_indices = col_indices + graph.epn_zero_cumsum[obs].repeat_interleave(counts)
    act_idx = graph.edge_index[1, edge_indices]

    act_node = graph.x[act_idx]
    act_edge = graph.edge_attr[edge_indices]
    act_embedding = torch.empty((act_node.shape[0], 0), device=obs.device) \
                    if embeddings is None else embeddings[act_idx]

    #Concatenate all components of the action features into a single sequence
    act_feat_seq = torch.cat((act_node, act_edge, act_embedding), axis=1)
    act_pos_seq = graph.pos[act_idx]

    #Define output shapes of act_feat and act_mask
    act_feat = torch.zeros(((obs.shape[0], max_actions, act_feat_seq.shape[1])), device=obs.device)
    act_pos = torch.zeros(((obs.shape[0], max_actions, 3)), device=obs.device)
    act_mask = torch.ones((obs.shape[0], max_actions), dtype=torch.bool, device=obs.device)
    
    #Return empty tensor if no neighbors were identified
    if counts.sum() == 0:
        return act_feat, act_pos, act_mask

    act_feat[row_indices, col_indices] = act_feat_seq
    act_pos[row_indices, col_indices] = act_pos_seq
    act_mask[row_indices, col_indices] = False
    
    return act_feat, act_pos, act_mask


def to_row_col_indices(row_values: torch.Tensor, counts: torch.Tensor) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a given set of row values (incremental list starting from 0) and
    corresponding counts to row and column indices respectively.

    :param row_values: Set of row values (incremental starting from 0).
    :param counts: Set of counts defining the number of elements in row.
    
    :return: Tuple containing both row and value indices.
    """
    arange = torch.arange(counts.sum(), device=counts.device)
    offset = F.pad(counts, (1, 1), 'constant', 0.0)[:-2]\
              .cumsum(dim=-1)\
              .repeat_interleave(counts)

    return row_values.repeat_interleave(counts), arange - offset


def query_neighborhood(graph: TGraph, obs: torch.Tensor, to_list: bool = True) \
    -> Union[List[torch.Tensor], torch.Tensor]:
    """Query the indices of the direct 1-hop neighborhood of given a tensor of 
    current node observations.

    @param graph: Pytorch geometric graph representation of the trajectory graph
    @param obs: Node indices of the current observation.
    @param to_list: Determines whether the neighborhood data should be 
                    splitted into a list of tensors. Otherwise a tensor
                    of shape [n_obs, max_neighbors] is created, defaults to True

    @returns: List/tensor of node indices adjacent to current observation.
    """
    if isinstance(obs, int): obs = torch.tensor([obs], device=graph.x.device, dtype=torch.long)
    
    counts = graph.edges_per_node[obs.type(torch.long)]
    row = torch.arange(obs.shape[0], device=obs.device)

    row_indices, col_indices = to_row_col_indices(row, counts)
    edge_indices = col_indices + graph.epn_zero_cumsum[obs].repeat_interleave(counts)
    neighbor_data = graph.edge_index[1, edge_indices]
    
    if to_list:
        return torch.split(neighbor_data, list(counts))
    
    neighbors = torch.full((counts.shape[0], max(counts.max(), 1)), -1, device=obs.device)
    neighbors[row_indices, col_indices] = neighbor_data
    return neighbors


def norm(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Returns the norm of a tensor for a given dim

    @param t: Tensor.
    @param dim: Dimension of norm.

    @returns norm of vector t given dimension dim
    """
    return torch.sqrt(torch.sum(t**2, dim=dim))


def get_detector_config(wet: bool = False, tracking_layers: bool = True):
    """Returns detector configuration containing distances between layers (x),
    total distance between start of detector and current layer (total_x) and
    radiation length for each layer.

    @param wet: Determines whether the distances should be converted to water 
                equivalent thicknesses, defaults to False
    @param tracking_layers: Determine wether tracker layer should be considered, 
                            defaults to False

    @returns: Tuple containing x, total_x and X0
    """
    x, total_x, X0 = None, None, None 
    if wet:
        x = np.array([0, 1.8, 2.34, *[8.2] * 48])
        total_x = np.cumsum(x[:-1])
        X0 = np.array([*[36.06] * 50])
    else:
        x = np.array([0, 57.8, 1, *[3.5] * 48])
        total_x = np.cumsum(x[:-1])
        X0 = np.array([36.62, *[24.01] * 49])
        
    if tracking_layers: 
        return x[1:], total_x, X0
    
    # Don't remove tracking layers from total x (otherwise: 
    # indexing in TGraph doesn't work anymore)
    return x[3:], total_x, X0[2:]