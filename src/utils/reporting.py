import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from scipy.stats import ttest_ind, ttest_1samp

from .experiments import RLExperiment
from src.utils.experiments import RLExperiment, SVExperiment

def compare_algorithms_to_table(def_exp: RLExperiment, output: str):
    """Calculates performance gap between default experiment and 
    heuristic search  and writes results into latex table.

    @param def_exp: Experiment definition of default experiment.
    @param output: Output file used for latex table.
    """
    df_res = pd.DataFrame()
    df_grad = pd.DataFrame()

    for wpt in [100, 150, 200]:
        get_file = lambda exp, wpt: f"{exp.model.model_dir}/results_water{wpt}.pkl"
        df_def = pd.read_pickle(get_file(def_exp, wpt)).loc[["pur_mu", "eff_mu"]]
        df_com = pd.read_csv(f"data/comparison/results_water{wpt}.csv", index_col="Unnamed: 0")
        df = (df_def - df_com)
        
        standard_error = df.applymap(lambda x: (x * 100).std())
        mean_error = df.applymap(lambda x: (x * 100).mean())
        
        ttest_pur = [ttest_1samp(df_def.loc[f"pur_mu"][str(particles)], 
                                df_com.loc[f"pur_mu"][str(particles)])[1] for particles in [10, 20, 30, 40, 50, 100, 150, 200]]
        
        ttest_eff = [ttest_1samp(df_def.loc[f"eff_mu"][str(particles)], 
                                df_com.loc[f"eff_mu"][str(particles)])[1] for particles in [10, 20, 30, 40, 50, 100, 150, 200]]
        
        df_res[f"{wpt}_p"] = [fr"{mu:.1f}$\pm${se:.1f}" for mu, se in zip(mean_error.loc["pur_mu"], standard_error.loc["pur_mu"])]
        df_res[f"{wpt}_e"] = [fr"{mu:.1f}$\pm${se:.1f}" for mu, se in zip(mean_error.loc["eff_mu"], standard_error.loc["eff_mu"])]
        
        df_grad[f"{wpt}_p"] = ttest_pur
        df_grad[f"{wpt}_e"] = ttest_eff
        
    cm = get_cmap("Greys")
    norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1)    

    df_res["primaries"] = [10, 20, 30, 40, 50, 100, 150, 200]
    df_res = df_res.set_index("primaries")
    df_res.index.name = None

    styled = df_res.style.background_gradient(cmap=cm, gmap=norm(df_grad.values), low=0, high=1, axis=None)
    with open(output, "w") as f:
        f.write(styled.to_latex(convert_css=True))
    
    return df


def ablation_to_table(def_exp: RLExperiment, abl_exp: RLExperiment, output: str):
    """Calculates performance gap between default and ablation experiment 
    and writes results into latex table.

    @param def_exp: Experiment definition of default experiment.
    @param abl_exp: Experiment definition of ablation experiment.
    @param output: Output file used for latex table.
    """    
    df_res = pd.DataFrame()
    df_grad = pd.DataFrame()

    for wpt in [100, 150, 200]:
        get_file = lambda exp, wpt: f"{exp.model.model_dir}/results_water{wpt}.pkl"
        df_def = pd.read_pickle(get_file(def_exp, wpt)).iloc[::2]
        df_abl = pd.read_pickle(get_file(abl_exp, wpt)).iloc[::2]
        
        #Difference 
        df = (df_def - df_abl)
        standard_error = np.sqrt(df_def.applymap(lambda x: (x * 100).std()**2/len(x)) + df_abl.applymap(lambda x: (x * 100).std()**2/len(x)))
        mean_error = df.applymap(lambda x: (x * 100).mean())
        
        #pvalue
        ttest_pur = [ttest_ind(df_def.loc[f"pur_mu"][str(particles)], 
                               df_abl.loc[f"pur_mu"][str(particles)], equal_var=False)[1] \
                                for particles in [10, 20, 30, 40, 50, 100, 150, 200]]
        
        ttest_eff = [ttest_ind(df_def.loc[f"eff_mu"][str(particles)], 
                               df_abl.loc[f"eff_mu"][str(particles)], equal_var=False)[1] \
                                for particles in [10, 20, 30, 40, 50, 100, 150, 200]]
        
        df_res[f"{wpt}_p"] = [fr"{mu:.1f}$\pm${se:.1f}" for mu, se in zip(mean_error.loc["pur_mu"], standard_error.loc["pur_mu"])]
        df_res[f"{wpt}_e"] = [fr"{mu:.1f}$\pm${se:.1f}" for mu, se in zip(mean_error.loc["eff_mu"], standard_error.loc["eff_mu"])]
        
        df_grad[f"{wpt}_p"] = ttest_pur
        df_grad[f"{wpt}_e"] = ttest_eff
    
    cm = get_cmap("Greys")
    norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1)
    
    df_res["primaries"] = [10, 20, 30, 40, 50, 100, 150, 200]
    df_res = df_res.set_index("primaries")
    df_res.index.name = None
    
    styled = df_res.style.background_gradient(cmap=cm, gmap=norm(df_grad.values), low=0, high=1, axis=None)
    with open(output, "w") as f:
        f.write(styled.to_latex(convert_css=True))
        
        
def experiment_to_table(exp: RLExperiment, output: str):
    """Export experiment results to latex table.

    @param def_exp: Experiment definition.
    @param output: Output file used for latex table.
    """
    df_res = pd.DataFrame()

    for wpt in [100, 150, 200]:
        get_file = lambda exp, wpt: f"{exp.model.model_dir}/results_water{wpt}.pkl"
        df = pd.read_pickle(get_file(exp, wpt)).iloc[::2]
        df = df.applymap(lambda x: fr"{x.mean() * 100:.1f}$\pm${x.std() * 100:.1f}")
        
        df_res[f"{wpt}_p"] = df.loc["pur_mu"]
        df_res[f"{wpt}_e"] = df.loc["eff_mu"]
    
    with open(output, "w") as f:
        df_res.to_latex(index=False, escape=False, buf=f)