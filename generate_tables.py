import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.pyplot import get_cmap
from matplotlib import pyplot as plt

from src.utils.experiments import RLExperiment, SVExperiment
from src.utils.reporting import (ablation_to_table, experiment_to_table,
                                 compare_algorithms_to_table)


if not os.path.exists("tables/"):
    os.mkdir("tables/")

# Spot scanning dataset to table (Table 1)

events = [50, 100, 150]
df_res = pd.DataFrame()

for i in range(3):
    dfs = [pd.read_csv(f"models/experiment_default_rl/results_head_phantom_spot_{events[i]}_{j}.txt") for j in range(5)]
    x_p = np.array([d.pur.median() for d in dfs])
    x_e = np.array([d.eff.median() for d in dfs])
    y_p = np.array([d.pur.mean() for d in dfs])
    y_e = np.array([d.eff.mean() for d in dfs])
    res = [f"{x_p.mean() * 100:.1f}$\pm${x_p.std() * 100:.1f} & {x_e.mean() * 100:.1f}$\pm${x_e.std() * 100:.1f}",
           f"{y_p.mean() * 100:.1f}$\pm${y_p.std() * 100:.1f} & {y_e.mean() * 100:.1f}$\pm${y_e.std() * 100:.1f}"]

    df_res[str(events[i])] = res
    
with open("tables/tab_head.tex", "w") as f:
    df_res.to_latex(index=False, escape=False, buf=f)
    
# WPT experiment and ablation studies to tables (Table 2-8)

def_exp = RLExperiment.from_file("experiments/default/default_rl_water.json")

for exp_name in ["pomdp", "no_pe", "pe", "no_reward_norm", "supervised"]:
    abl_exp = RLExperiment.from_file(f"experiments/ablation/{exp_name}.json") \
        if "supervised" not in exp_name\
        else SVExperiment.from_file(f"experiments/ablation/{exp_name}.json")
        
    ablation_to_table(def_exp, abl_exp, f"tables/tab_{exp_name}.tex")
    
df = experiment_to_table(def_exp, "tables/tab_default.tex")
df = compare_algorithms_to_table(def_exp, "tables/tab_comparison.tex")


fig, ax = plt.subplots(figsize=(8, 0.25))
fig.subplots_adjust(bottom=0.5)

cm = get_cmap('Greys')

norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1)
cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cm,
                                norm=norm,
                                extend='both',
                                extendfrac=0.02,
                                spacing='uniform',
                                orientation='horizontal')
cb3.set_label("p-value (Welch's t-test)", size='large')
plt.savefig("tables/colorbar.pdf", bbox_inches='tight')