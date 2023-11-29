**Status:** Archive (code is provided as-is, no updates expected)

# Towards Neural Charged Particle Tracking in Digital Tracking Calorimeters with Reinforcement Learning
[Tobias Kortus, Ralf Keidel, Nicolas R. Gauger](https://sivert.info), on behalf of *the Bergen pCT* collaboration

The repository contains [PyTorch](https://pytorch.org/) implementation of "Towards Neural Charged Particle Tracking in Digital Tracking Calorimeters with Reinforcement Learning"

> We propose a novel reconstruction scheme for reconstructing charged particles in digital tracking calorimeters using model-free reinforcement learning aiming to benefit from the rapid progress and success of neural network architectures for tracking without the dependency on simulated or manually labeled data. Here we optimize by trial-and-error a behavior policy acting as a heuristic approximation to the full combinatorial optimization problem, maximizing the physical plausibility of sampled trajectories.

<a href="https://sivert.info"><img src="https://img.shields.io/website?style=flat-square&logo=appveyor?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=sivert.info&url=https://sivert.info" height=22.5></a>


## Installation

```powershell
pip install -r requirements.txt
```

## Data and Models

For reproducibility we provide the user with the exact Monte-Carlo simulations used throughout the paper. All simulations can be downloaded from [Zenodo](https://zenodo.org/record/7426388) and extracted using the following command (the downloaded archive should be copied into the `data/` directory):

```powershell
tar -xf data.tar.gz --strip-components 1
```

> Note: The files `data/comparison/results_water{100, 150, 200}.csv` contain the processed reconstruction performances for the heuristic search approach by Pettersen et al. The source code can be found under [1].

> Note: Please note that the following instructions are provided for Linux operating systems. Some commands may vary for different operating systems.

Similarly we provide the pretrained weights and results of all evaluated network variants used throughout the paper. The model data can be extracted in a similar fashion using:

```powershell
tar -xf models.tar.gz --strip-components 1
```

## Running Experiments

All experiments with the corresponding hyperparameters parameters, performed in the paper, are documented as `.json` files. An experiment, with the provided models, can be re-run using the following commands:

```powershell
python policy_evaluation.py  -e experiments/****.json -t ** -d ****
```

- `-e`: Experiment definition file. Either one of the predefined in `experiments/default`/ `experiments/ablation` or a custom definition following the json structure of the existing experiments.
- `-t`: Experiment definition file. One of `rl` (reinforcement learning) or `sv` (supervised).
- `-d`: Computation device that should be used by pytorch (cpu, cuda:1-N)

> Note: We faced issues with non-reproducible results when using a CUDA device during training. Thus all models provided were trained on a CPU. If you wish to retrain a model, the respective `*.pt` files should be deleted from the corresponding model directory.

## Running Reporting Scripts

```powershell
python generate_tables.py
```

> Note: All source code for creating spot scanning figures (figures 6-8) is provided in the jupyter notebooks `visualize_spot_metrics_head.ipynb` and `visualize_spot_metrics_solid.ipynb`.

## Referencing this Work

If you find this repository useful for your research, please cite the following work.

```
@ARTICLE{10219056,
  author={Kortus, Tobias and Keidel, Ralf and Gauger, Nicolas R. and Bergen pCT Collaboration},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Towards Neural Charged Particle Tracking in Digital Tracking Calorimeters With Reinforcement Learning}, 
  year={2023},
  volume={45},
  number={12},
  pages={15820-15833},
  doi={10.1109/TPAMI.2023.3305027}}
```

## References

[1] **Digital Tracking Calorimeter Toolkit**, Helge E.S. Pettersen, [Sorce code]: https://github.com/HelgeEgil/DigitalTrackingCalorimeterToolkit

Parts of this implementation are inspired by:

- https://github.com/DLR-RM/stable-baselines3 (Stable Baselines3)
- https://github.com/openai/gym (Gym)
