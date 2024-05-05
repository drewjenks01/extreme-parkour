# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from requests import get
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, terrain
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from tqdm import tqdm

def play(args):
    args.proj_name = 'final_models'
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    #env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.0,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 1.,
                        "parkour_step": 0.,
                        "parkour_gap": 0.,
                        "demo": 1.,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    terrain_names = [key for key in env_cfg.terrain.terrain_dict.keys() if env_cfg.terrain.terrain_dict[key]!=0 ]
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = True
    env_cfg.env.eval = True

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        if env.cfg.depth.use_rgb:
            vision_encoder = ppo_runner.get_rgb_encoder_inference_policy(device=env.device)
        else:
            vision_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
  
    total_steps = 1000

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}

    image_type = "rgb" if env.cfg.depth.use_rgb else "depth"
    
    if env.cfg.depth.use_rgb:
        infos["rgb"] = env.rgb_buffer.clone().to(env.device)[:, -1] if args.use_rgb else None
    else:
        infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1] if args.use_depth else None


    vision_latents = torch.zeros(env.num_envs, 10, 1500, device=env.device, requires_grad=False)

    one_ep = int(env.max_episode_length)
    env_goal_idxs = []
    env_goal = 0
    for i in tqdm(range(one_ep)):

        if env.cfg.depth.use_camera:
            if infos[image_type] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio]
                obs_student[:, 6:8] = 0
                with torch.no_grad():
                    vision_latent_and_yaw = vision_encoder(infos[image_type], obs_student)
                vision_latent = vision_latent_and_yaw[:, :-2]
                yaw = 1.5*vision_latent_and_yaw[:, -2:]
            obs[:, 6:8] = yaw     
        else:
            vision_latent = None

        if hasattr(ppo_runner.alg, "rgb_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.rgb_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        elif hasattr(ppo_runner.alg, "depth_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        else:
            actions = policy(obs.detach(), vision_latent)

        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, infos = env.step(actions.detach())

        if cur_goal_idx[0] != env_goal and cur_goal_idx[0]>env_goal:
            env_goal_idxs.append(i)
            env_goal += 1

    
    terrain_types  = env.terrain_types

    vision_latents = ppo_runner.alg.actor_critic.actor.classifications.cpu().detach().numpy()[0]

    # create a figure with 6 subplots
    plt.figure(figsize=(20, 10))
    plt.title('Classifcation results')

    var = np.var(vision_latents, axis=1)
    sig_classes = np.where(var > 0.01)
    if terrain_names[0] != 'parkour_flat':
        curr_latent = vision_latents[sig_classes]
    else:
        curr_latent = vision_latents
        sig_classes = np.where(var > 0.001)
    
    plt.plot(curr_latent.T)
    plt.vlines(x=env_goal_idxs, ymin=0, ymax=1, colors='black', linestyles='dashed', label='waypoints')
    plt.title(f'Terrain type: {terrain_names[0]}')
    plt.xlabel('Time step')
    plt.ylabel('Vision latent')
    plt.legend([f'Class {i}' for i in list(sig_classes[0])]+['waypoints'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f'classification_results_{terrain_names[0]}.png')
        


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)


# 038-10 no feet edge
# 038-91 ours
# 043-21 non-inner