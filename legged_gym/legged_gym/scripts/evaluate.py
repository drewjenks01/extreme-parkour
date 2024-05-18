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
import numpy as np

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
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
from torch.nn.functional import normalize

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
    env_cfg.env.num_envs = 256
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0.0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.25,
                                    "parkour_hurdle": 0.25,
                                    "parkour_flat": 0.0,
                                    "parkour_step": 0.25,
                                    "parkour_gap": 0.25, 
                                    "demo": 0}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.eval = True

    record_video = False
    video_saved = False

    if record_video:
        import wandb
        wandb.init(project="walk-these-ways", name=args.exptid, entity="iai-eipo", group=args.exptid[:3], mode='offline', dir=LEGGED_GYM_ROOT_DIR+'/experiment/eval_videos')

    # TODO: UPDATE
    # env_cfg.depth.rgb_horizontal_fov = env_cfg.depth.horizontal_fov
    # env_cfg.domain_rand.randomize_lighting = True
    # env_cfg.domain_rand.randomize_ground_texture = True

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    if not args.use_jit:
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    if not args.use_jit:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if env.cfg.depth.use_camera:
            if env.cfg.depth.use_rgb:
                vision_encoder = ppo_runner.get_rgb_encoder_inference_policy(device=env.device)
            else:
                vision_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    else:
        jit_path = LEGGED_GYM_ROOT_DIR + f"/logs/final_models/{args.exptid}/traced/"
        print(f'Jit path: {jit_path}')
        if env_cfg.env.num_envs == 1:
            policy = torch.jit.load(jit_path+'traced_actor_robot.jit', map_location=env.device).to(env.device)
            vision_encoder = torch.jit.load(jit_path+'traced_vision_encoder_robot.jit', map_location=env.device).to(env.device)
        else:
            policy = torch.jit.load(jit_path+'traced_actor_eval.jit', map_location=env.device).to(env.device)
            vision_encoder = torch.jit.load(jit_path+'traced_vision_encoder_eval.jit', map_location=env.device).to(env.device)
    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    time_to_fall_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)

    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_edge_violation = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}

    image_type = "rgb" if env.cfg.depth.use_rgb else "depth"
    
    if env.cfg.depth.use_rgb:
        infos["rgb"] = env.rgb_buffer.clone().to(env.device)[:, -1] if args.use_rgb else None
    else:
        infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1] if args.use_depth else None

    import math
    for i in tqdm(range(1500)):
        if args.use_jit:
            obs[:, env_cfg.env.n_proprio:env_cfg.env.n_proprio+env_cfg.env.n_scan+env_cfg.env.n_priv+env_cfg.env.n_priv_latent] = 0
        
        if env.cfg.depth.use_camera:
            if infos[image_type] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio]
                obs_student[:, 6:8] = 0
                with torch.no_grad():
                    vision_latent_and_yaw = vision_encoder(infos[image_type], obs_student)
                    #vision_latent_and_yaw = normalize(vision_latent_and_yaw, dim=1)
                vision_latent = vision_latent_and_yaw[:, :-2]
               # print(vision_latent)
                yaw = 1.5*vision_latent_and_yaw[:, -2:]
            #obs[:, 6:8] = yaw
                
        else:
            vision_latent = None

        if not args.use_jit and hasattr(ppo_runner.alg, "rgb_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.rgb_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        elif not args.use_jit and hasattr(ppo_runner.alg, "depth_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=vision_latent)
        else:
            actions = policy(obs.detach(), vision_latent)

        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        
        id = env.lookat_id
        # Log stuff
        edge_violation_buffer.extend(env.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())
        # cur_edge_violation += env.feet_at_edge.sum(dim=1).float()
        cur_reward_sum += rews
        cur_episode_length += 1
        cur_time_from_start += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        killed_ids = ((dones > 0) & (~infos["time_outs"])).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
        time_to_fall_buffer.extend(cur_time_from_start[killed_ids][:, 0].cpu().numpy().tolist())

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_edge_violation[new_ids] = 0
        cur_time_from_start[killed_ids] = 0

        if record_video:
            if i ==0:
                env.start_recording()
            else:
                frames = env.get_complete_frames()

                if len(frames) > 0:
                    env.pause_recording()
                    print("LOGGING VIDEO")
                    video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
                    print(video_array.shape)
                    # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
                    wandb.log({"video": wandb.Video(video_array, fps=50)}, step=i)
                    video_saved=True
                    break

    if record_video and not video_saved:
        frames = env.video_frames
        video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
        print(video_array.shape)
        # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
        wandb.log({"video": wandb.Video(video_array, fps=50)}, step=i)

    
    #compute buffer mean and std
    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)

    # time_to_fall_mean = statistics.mean(time_to_fall_buffer)
    # time_to_fall_std = statistics.stdev(time_to_fall_buffer)

    edge_violation_mean = np.mean(edge_violation_buffer)
    edge_violation_std = np.std(edge_violation_buffer)

    print("Mean reward: {:.2f}$\pm${:.2f}".format(rew_mean, rew_std))
    print("Mean episode length: {:.2f}$\pm${:.2f}".format(len_mean, len_std))
    print("Mean number of waypoints: {:.2f}$\pm${:.2f}".format(num_waypoints_mean, num_waypoints_std))
    # print("Mean time to fall: {:.2f}$\pm${:.2f}".format(time_to_fall_mean, time_to_fall_std))
    print("Mean edge violation: {:.2f}$\pm${:.2f}".format(edge_violation_mean, edge_violation_std))


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)


# 038-10 no feet edge
# 038-91 ours
# 043-21 non-inner