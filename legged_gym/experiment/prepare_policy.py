import re
from cv2 import split
import isaacgym
from yaml import scan

assert isaacgym
import torch

import os



def load_and_save_policy(args):

    run_path = args.run_path

    import wandb
    api = wandb.Api()
    run = api.run(run_path)

    run_name = run.name

    run_files  = run.files()

    all_cfg = run.config
    cfg = all_cfg["Cfg"]

    print(cfg.keys())

    phase = args.phase
    extra_name = args.name

    path = f'../logs/parkour_new/phase{phase}_{extra_name}'
    if not os.path.exists(path):
        os.makedirs(path)

    actor_files = [f for f in run_files if 'model_' in f.name]

    if args.max_iter:
        max_iter = args.max_iter
    else:
        max_iter = max([int(f.name.split('_')[-1].replace('.pt', '')) for f in actor_files])

    actor_path = os.path.join(path, f'model_latest.pt')
    print(f'{run_path}/{run_name}/model_{max_iter}.pt')
    actor_file = run.file(f'{run_name}/model_{max_iter}.pt').download(replace=True, root='./tmp')
    actor = torch.load(actor_file.name, map_location="cpu")
    torch.save(actor, actor_path)
    print('Saved actor')
   
    cfg_path = os.path.join(path, f'config_preferred.yaml')
    import yaml
    with open(cfg_path, 'w') as f:
        yaml.dump(all_cfg, f)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Load and save policy')
    parser.add_argument('--run_path', type=str, help='run path', required=True)
    parser.add_argument('--phase', type=int, help='phase', required=True)
    parser.add_argument('--name', type=str, help='extra descripition', required=True)
    parser.add_argument('--max_iter', type=int, help='max iter', required=False)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    #run_path = 'iai-eipo/walk-these-ways/3rvu8a6i'
    #phase = 1

    run_path = 'iai-eipo/walk-these-ways/t8sz5tj2'

    load_and_save_policy(args)