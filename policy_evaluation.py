import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, help='.json file containing the experiment definition', required=True)
parser.add_argument('-t', '--type', type=str, help='Experiment type [rl, sv]', required=True, choices=["rl", "sv"])
parser.add_argument('-d', '--device', type=str, help='CPU/GPU that should be used for evaluation', required=True)
args = parser.parse_args()

## Imports after argparse to make argparse more responsive
from src.utils.experiments import ExperimentBase, RLExperiment, SVExperiment

exp: ExperimentBase = None

if args.type == "rl":
    exp = RLExperiment.from_file(args.experiment)
elif args.type == "sv":
    exp =SVExperiment.from_file(args.experiment)
    
exp.generate_datasets(device=args.device)
exp.train_model(device=args.device)

if exp.dataset.test_files != None:
    exp.evaluate_model(device=args.device)