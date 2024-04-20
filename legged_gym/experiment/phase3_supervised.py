import isaacgym

import os

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def train(args):
    #args.headless = True
    #log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    log_pth = "/data/scratch-oc40/pulkitag/awj/extreme-parkour/legged_gym/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "offline"
    
    if args.no_wandb:
        mode = "disabled"

    env, env_cfg = task_registry.make_env(name=args.task, args=args)    
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)

    env_cfg.env.wandb_offline = False

    Cfg = class_to_dict(env_cfg)
    wandb.init(project="walk-these-ways", name=args.exptid, entity="iai-eipo", group=args.exptid[:3], mode='online', config={'Cfg': Cfg})
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")


    # load dataset



if __name__ == '__main__':
    args = get_args()
    train(args)