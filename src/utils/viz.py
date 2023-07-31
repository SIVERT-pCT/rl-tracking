import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from mpl_toolkits.mplot3d import Axes3D
from src.reinforcement.environment import ParallelEnvironment
from src.reinforcement.episodes import ParallelEpisodes


def create_figure() -> Axes3D:
    fig = plt.figure(figsize=(5, 10))
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 60)
    ax.set_box_aspect(aspect = (1,1,2))
    return fig, ax


def plot_centroids(ax: Axes3D, env: ParallelEnvironment, s:int = 1):
    c='black'    
    ax.scatter(env.graph.pos[:,0].cpu(), 
               env.graph.pos[:,1].cpu(), 
               env.graph.pos[:,2].cpu(), 
               marker='.', c=c, s=s)


def configure_plot(ax: Axes3D):
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])

    ax.set_xlabel("x-position [mm]")
    ax.set_ylabel("y-position [mm]")
    ax.set_zlabel("z-position [mm]")
    ax.margins(z=0.1)
    plt.tight_layout()


def save_figure(save_path):
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def iterate_tracks(tracks: Union[ParallelEpisodes, List]):
    if isinstance(tracks, ParallelEpisodes):
        tracks = tracks.trajectories

    for track in tracks:
        track_len = len(track)
        
        if track_len == 0:
            continue
        
        if isinstance(track, torch.Tensor):
            track_idx = track
        elif isinstance(track[0], torch.Tensor):
            track_idx = torch.stack(track)
        else:
            track_idx = np.array(track)
            
        yield track_idx, track_len


def plot_colored_tracks(env: ParallelEnvironment, tracks: Union[ParallelEpisodes, List],
                        save_path: str = None, filter="all", centroids=True) -> None:
    """Plots and colors sampled trajectories according to their correctness:
    Green: Correct reconstruction corresponding to base eventID.
    Orange: Correct reconstruction corresponding to a false track.
    Red: False reconstruction.

    @param env: Parallel environment determining the transition dynamics 
                of the system
    @param episodes: Sampled episodes obtained by using policy pi.
    @param save_path: Path where the graphic should be saved, defaults to None
    @param filter: Filter which tracks should be displayed. One of 
                   ["all", "true", "false"0], defaults to "all"
    """
    fig, ax = create_figure()
    if centroids:
        plot_centroids(ax, env)
          
    for track_idx, track_len in iterate_tracks(tracks):     
        track_pos = env.graph.pos[track_idx].cpu()
        track_event_ids = env.graph.y[track_idx].cpu()
        track_event_id = track_event_ids[-1]
        
        if filter == "true" and not torch.all(track_event_ids == track_event_id):
            continue
        if filter == "false" and torch.all(track_event_ids == track_event_id):
            continue
        
        for i in reversed(range(track_len - 1)):
            f, t = track_pos[i], track_pos[i+1] 
            
            color = 'green' if track_event_id == track_event_ids[i] else 'red'
            if color == 'red' and track_event_ids[i] == track_event_ids[i+1]:
                color = 'orange'
            
            ax.plot([f[0], t[0]], [f[1], t[1]], [f[2], t[2]], color=color)
            
    configure_plot(ax)
    save_figure(save_path)