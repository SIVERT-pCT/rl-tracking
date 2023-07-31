import os
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd

from .pct_restrict_detector_boundaries import digitize

pd.options.mode.chained_assignment = None

format_lookup = {".csv": pd.read_csv,
                 ".pkl": pd.read_pickle}

class GateVersion(Enum):
    v91 = 0
    v92 = 1
    
VERSION: GateVersion = GateVersion.v92

def set_gate_version(version: GateVersion):
    VERSION = version


def open_dataframe_by_extension(path: str) ->  pd.DataFrame:
    """Loads a file path pointing to some kind of 
    tabular data storage format (currently .csv or .pkl) 
    and loads it to pandas dataframe.

    @param path: Path to storage file.

    @returns: Loaded pandas dataframe.
    """
    _, ext = os.path.splitext(path)
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".pkl":
        return pd.read_pickle(path)
    elif ext == ".npy":
        return pd.DataFrame(np.load(path))
    elif ext == ".npz":
        return pd.DataFrame(np.load(path)["arr_0"])
    else:
        raise KeyError(f"Invalid file extension {ext}")


def split_layers_z(df: pd.DataFrame, gate_version: GateVersion = VERSION):
    """Get individual z values for each layer starting from 0..42. 
    Decouples tracker layers and first two calorimeter layers.

    @param df: Pandas dataframe
    
    @returns: Updated dataframe with z column
    """
    if gate_version == GateVersion.v91:
        rear_scanner_layer_count = {i: 
                len(df[df["baseID"] < i].groupby(["level1ID", "baseID"])) \
                for i in range(0,df.baseID.max() + 1)}
        
        apply_layer_z = lambda x: rear_scanner_layer_count[x.baseID] + x.level1ID
        df["z"] = df.apply(apply_layer_z, axis=1)
    else:
        if "level2ID" in df.columns and "level1ID" in df.columns:
            df["z"] = (df["level2ID"] != -1) * (2 + df["level2ID"]) + \
                      (df["level2ID"] == -1) *  df["level1ID"]
        else:
            df["z"] = df["volumeID[2]"] * 2 + df["volumeID[3]"]
            
    return df


def merge_duplicate_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Merges duplicate pixel activations in a given pandas
    dataframe by summing up the energy deposition and calculating
    the mean of the exact X,Y and Z positions.

    @param df: Pandas dataframe
    
    @return: Dataframe with merged pixels
    """
    df = digitize(df, sub_pixel_resolution=1)
    df = split_layers_z(df)
    aggregation_dict = dict.fromkeys(df, 'first')
    aggregation_dict.update({'edep': 'sum',
                             'posX': 'mean',
                             'posY': 'mean',
                             'posZ': 'mean'})
    
    return df.groupby(["eventID", "trackID", "z"], as_index=False) \
            .agg(aggregation_dict) \
            .reset_index(drop=True)


def remove_disconnected(df: pd.DataFrame, allowed_skips=0):
    """Removes secondary particles from last layers, where the consecutive
    layers are not directly connected or more than a limited amount of layers
    are skipped.

    @param df: Dataframe containing the detector data.
    @param allowed_skips: Number of layers that can be skipped, defaults to 0
    
    @returns: Dataframe with filtered data.
    """
    unique_z = np.sort(df.z.unique())
    skip = (unique_z[1:] - unique_z[:-1] - 1) > allowed_skips
    occ = np.where(skip)[0]
    if len(occ) == 0: return df
    mask = ~(unique_z < occ[0])
    df = df[~df.z.isin(unique_z[mask])]

    return df


def edep_to_cluster_size(edep: Union[np.ndarray, float]):
    return (4.2267 * (edep * 1000./25.) ** 0.65 + 0.5)

def append_cluster_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines cluster size in pixels for a given energy deposition per thickness of the epitaxial layer
    and appends it to a given dataframe

    cf. H. Pettersen (https://github.com/HelgeEgil/DigitalTrackingCalorimeterToolkit)

    @param df: Dataframe containing the detector data with energy deposition (edep) by epitaxial layer MeV
    
    @returns: Dataframe with cluster size in number of pixels (column: cluster)
    """
    df["cluster"] = edep_to_cluster_size(df.edep).astype(int)
    return df


def generate_valid_tracks_by_graph_idx(df: pd.DataFrame) -> List[List[int]]:
    """Generate a list of trajectories containing the indices of 
    valid tracks generated from the ground truth of the monte carlo
    simulation

    @param df: Data frame containing the monte carlo simulation data
    """
    assert "z" in df.columns, "Data frame must contain z column"
    assert "eventID" in df.columns, "Data frame must contain eventID column"
    
    df_sorted = df.sort_values("z", ascending=True)
    df_sorted["idx"] = np.arange(len(df_sorted))
    df_tracks = df_sorted.groupby("eventID")
    return list(df_tracks.apply(lambda x: list(x.idx)))


def preprocess(df: pd.DataFrame):
    """Run all preprocessing steps required for generating the 
    TGraph object.
    
    @param df: Dataframe containing detector data.
    """
    df = digitize(df, sub_pixel_resolution=1)
    df = split_layers_z(df)
    df = remove_disconnected(df)
    df = merge_duplicate_pixels(df)
    df = append_cluster_size(df)
    df.attrs["preprocessed"] = True #FIXME: attr is experimental and may change in a future version
    return df


def is_preprocessed(df: pd.DataFrame):
    return df.attrs.get("preprocessed") == True

def is_spot_scanning(df: pd.DataFrame):
    return "spotX" in df.columns and "spotY" in df.columns