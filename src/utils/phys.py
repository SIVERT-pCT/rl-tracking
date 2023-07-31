import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants as const
from typing import List, Union, Tuple
from torch.distributions import Normal
from torch.nn import functional as F

from .graph import TGraph, get_detector_config

## Physical constants (scattering)
pi = const.pi
m = const.proton_mass
c = const.speed_of_light

# Constants for range fitting (ref. Pettersen, 2018)
#rho = 17.7  # g cm^-3 
alpha = 0.0262  # cm/MeV
p = 1.736 # Fitted value


def diff_bragg_kleeman(z: np.ndarray, R: float) -> np.ndarray:
    """ Differential bragg kleeman equation with default values
    adapted from Pettersen, 2018 (PhD Thesis)
    
    @param z: Position for energy deposition estimates [mm]
    @param R: Calculated Range [mm]
    
    return: Energy depositions for positions in z [MeV]
    """
    rho_p_alpha =  p * np.power(alpha, 1. / p) # We are only "in Water" -> no rho
    exponent = 1. - 1. / p
    return 1. / (rho_p_alpha * np.power(R - z, exponent))


def fit_bragg_kleeman(edep: np.ndarray, z: np.ndarray)-> Tuple[float, float]:
    """ Estimates the proton range by fitting the bragg kleeman equation
    
    @param edep: Numpy array containing the measured energy depositions 
                 in the detector layers [MeV]
    @param z: Positions of the layer in the detector [mm]
    
    @returns Tuple containing range [mm] and r^2 value
    """
    pval, _ = curve_fit(diff_bragg_kleeman, z, edep/25*1000, bounds=(z.max(), 400))
    return pval


def estimate_energy(R: float, z: np.array) -> float:
    """ Energy obtained by integrating over bragg kleeman equation
    
    @param R: Range determined by fit_bragg_kleeman [mm]
    @param x: Ranges used for estimating energies [mm]
              
    @returns: Estimated energies of the proton [MeV]
    """
    assert R > 0, "Ranges <= 0 are no valid input"
    assert all(z <= R), "Z position must be smaller/equal R"
    assert all(z >= 0), "Z position must be grater/equal to 0"
    
    return alpha ** (-1/p) * (R - z) ** (1/p)


def highland_scattering(E: Union[float, np.ndarray], x: float = 3, X0: float = 36.08):
    """ Estimates the standard deviation of the MCS process
    in a given thickness x using highlands approximation.

    @param E: Proton energy at the current position [MeV]
    @param x: Thickness of material [mm]
    @param X0: Radiation length of the material [g/cm^3], defaults to 36.08 (water)
    
    @returns: Scattering angle theta0 for the given velocity [rad]
    """
    tau = E/(m*c**2)
    pv = (tau+2)/(tau+1) * E
    return (14.1/pv)*np.sqrt(x/X0)*(1+(1/9)*np.log10(x/X0))


def estim_track_log_prob(graph: TGraph, track_candidate: List[int]) -> Tuple[float, List[float]]:
    """ Determines the likelihood of a full sampled track candidate
    based on the likelihood (scattering) of the individual track 
    transitions.
    
    @param graph: Graph representation of the track reconstruction environment.
    @param track_candidate: List containing multiple indices (based on data)
                            of individual hits associated with the sampled 
                            track candidate
    
    @returns Tuple containing the total log likelihood of the sampled track candidate 
             and the individual log likelihoods of the connections
    """
    x = track_candidate
    prev, curr, next = x[0:-2], x[1:-1], x[2:]
    xs, total_x, X0 = get_detector_config(wet=True, tracking_layers=graph.contains_tracker)

    try: R = fit_bragg_kleeman(graph.edep[x].cpu().numpy(), total_x[:len(x)])
    except: 
        return -100 * len(curr), torch.ones_like(curr) * -100
    
    x0_x1 = graph.pos[prev] - graph.pos[curr]
    x1_x2 = graph.pos[curr] - graph.pos[next]

    energies = estimate_energy(*R, total_x[1:len(x)-1])
    
    thetas = torch.acos(F.cosine_similarity(x1_x2, x0_x1, dim=-1))
    thetas = torch.nan_to_num(thetas, 0.0) #nan if angle is 0

    xs = xs[:len(curr)]
    X0 = X0[:len(curr)]

    sigmas_scatter = torch.tensor(highland_scattering(energies, xs, X0), device=thetas.device)

    dist = Normal(0, sigmas_scatter)
    
    likelihoods = dist.log_prob(thetas) - dist.log_prob(torch.zeros_like(thetas))
    
    return likelihoods.sum(), likelihoods