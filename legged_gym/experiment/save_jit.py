import os, sys
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RGBOnlyFCBackbone58x87, RecurrentDepthBackbone, RecurrentDepthBackboneClassifier
import argparse
import code
import shutil

from legged_gym.utils.helpers import get_args

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareActorNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareActorNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs, vision_latent, vision_yaw):
        obs[:, 6:8] = vision_yaw
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=vision_latent)



def load_vision_encoder(vision_type, num_prop, scan_encoder_dims, depth_encoder_hidden_dim):
    vision_encoder = None
    if vision_type == 'depth':
        print('Using depth backbone')
        vision_backbone = DepthOnlyFCBackbone58x87(num_prop, scan_encoder_dims[-1], depth_encoder_hidden_dim)
    
    elif vision_type == 'rgb':
        print('Using rgb backbone')
        vision_backbone = RGBOnlyFCBackbone58x87(num_prop, scan_encoder_dims[-1],depth_encoder_hidden_dim)
    
    elif vision_type =='classifier':
        print('Using depth classifier encoder')
        vision_backbone = DepthOnlyFCBackbone58x87(num_prop, 32, depth_encoder_hidden_dim)
        vision_encoder = RecurrentDepthBackboneClassifier(vision_backbone, num_prop)

    if vision_encoder is None:
        vision_encoder = RecurrentDepthBackbone(vision_backbone, num_prop)
    else:
        vision_encoder = vision_encoder

    return vision_encoder


class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        vision_type,
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.vision_type = vision_type
        
        vision_encoder = None
        if self.vision_type == 'depth':
            print('Using depth backbone')
            vision_backbone = DepthOnlyFCBackbone58x87(self.num_prop, scan_encoder_dims[-1], depth_encoder_hidden_dim)
        
        elif self.vision_type == 'rgb':
            print('Using rgb backbone')
            vision_backbone = RGBOnlyFCBackbone58x87(self.num_prop, scan_encoder_dims[-1],depth_encoder_hidden_dim)
      
        elif self.vision_type =='classifier':
            print('Using depth classifier encoder')
            vision_backbone = DepthOnlyFCBackbone58x87(self.num_prop, 32, depth_encoder_hidden_dim)
            vision_encoder = RecurrentDepthBackboneClassifier(vision_backbone, self.num_prop)

        if vision_encoder is None:
            self.vision_encoder = RecurrentDepthBackbone(vision_backbone, self.num_prop)
        else:
            self.vision_encoder = vision_encoder
        
    def forward(self, vision_img, obs):
        obs_prop_vision = obs[:, :self.num_prop].clone()
        obs_prop_vision[:, 6:8] = 0
        vision_latent_and_yaw = self.vision_encoder(vision_img.clone(), obs_prop_vision)

        vision_latent = vision_latent_and_yaw[:, :-2]
        yaw = 1.5*vision_latent_and_yaw[:, -2:]

        return vision_latent, yaw
    
    @torch.jit.export
    def reset_hidden_states(self):
        self.vision_encoder.reset_hidden_states()

def play(args):    
    load_run = "../logs/final_models/" + args.exptid
    checkpoint = args.checkpoint

    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 4 + 1 + 12 +12+4
    num_scan = 132
    num_actions = 12
    
    # depth_buffer_len = 2
    depth_resized = (87, 58)
    
    n_proprio = 3 + 2 + 3 + 4 + 36 + 4 +1
    history_len = 10

    if args.use_depth:
        vision_type = 'depth'
    elif args.use_rgb:
        vision_type = 'rgb'

    device = torch.device('cpu')


    for num_envs, save_name in zip((1, 192), ('robot','eval')):

        policy = HardwareActorNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions).to(device)
        #vision_encoder = HardwareVisionNN(n_proprio, vision_type).to(device)
        vision_encoder = load_vision_encoder(vision_type, n_proprio, [128, 64, 32], 512).to(device)

        load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
        load_run = os.path.dirname(load_path)
        print(f"Loading model from: {load_path}")
        ac_state_dict = torch.load(load_path, map_location=device)
        # policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
        if vision_type == 'rgb':
            policy.actor.load_state_dict(ac_state_dict['rgb_actor_state_dict'], strict=True)
            #vision_encoder.vision_encoder.load_state_dict(ac_state_dict['rgb_encoder_state_dict'], strict=True)
            vision_encoder.load_state_dict(ac_state_dict['rgb_encoder_state_dict'], strict=True)

        elif vision_type == 'depth':
            policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
            #vision_encoder.vision_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'], strict=True)
            vision_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'], strict=True)


        policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])

        vision_encoder = vision_encoder.to(device)
        
        policy = policy.to(device)#.cpu()
        if not os.path.exists(os.path.join(load_run, "traced")):
            os.mkdir(os.path.join(load_run, "traced"))
        
        # state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}
        # torch.save(state_dict, os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))

        # Save the traced actor
        policy.eval()
        vision_encoder.eval()
        with torch.no_grad(): 
            obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
            depth_latent = torch.ones(num_envs, 32, device=device)
            depth_yaw = torch.ones(num_envs, 2, device=device)
            test = policy(obs_input, depth_latent, depth_yaw)
            
            traced_policy = torch.jit.trace(policy, (obs_input, depth_latent, depth_yaw))
            
            # traced_policy = torch.jit.script(policy)
            save_path = os.path.join(load_run, "traced", f"traced_actor_{save_name}.jit")
            traced_policy.save(save_path)
            print("Saved traced_actor at ", os.path.abspath(save_path))

            obs_input = torch.ones(num_envs, n_proprio, device=device)
            if vision_type == 'depth':
                depth_img = torch.ones(num_envs, 58, 87, device=device)
            elif vision_type == 'rgb':
                depth_img = torch.ones(num_envs, 3, 58, 87, device=device)

            print(obs_input.shape, depth_img.shape)

            test = vision_encoder(depth_img, obs_input)
            
            traced_vision_encoder = torch.jit.script(vision_encoder, (depth_img, obs_input))
            
            # traced_policy = torch.jit.script(policy)
            save_path = os.path.join(load_run, "traced", f"traced_vision_encoder_{save_name}.jit")
            traced_vision_encoder.save(save_path)
            print("Saved traced_vision_encoder at ", os.path.abspath(save_path))

    
if __name__ == '__main__':
    args = get_args()
    play(args)
    