
from copy import deepcopy
from json import load
import torch
import torch.nn as nn

from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone

class HardwareScandotNN(nn.Module):
    def __init__(   self,  
                    num_prop,
                    num_scan,
                    num_priv_explicit,
                    num_hist,
                    load_path=None):
        super(HardwareScandotNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_priv_explicit = num_priv_explicit

        if load_path is not None:
            self.adaptation_module = torch.jit.load(load_path+"/adaptation_module_latest.jit", map_location="cpu").eval()
            self.actor_body = torch.jit.load(load_path+"/body_latest.jit", map_location="cpu").eval()
            self.scan_encoder = torch.jit.load(load_path+"/scan_encoder_latest.jit", map_location="cpu").eval()
            self.estimator = torch.jit.load(load_path+"/estimator_latest.jit", map_location="cpu").eval()
        else:
            self.adaptation_module = None
            self.actor_body = None
            self.scan_encoder = None
            self.estimator = None


    def forward(self, obs):
        priv_explicit = self.estimator(obs[:, :self.num_prop])

        observation_history = obs[:, -self.num_hist:]
        latent = self.adaptation_module(observation_history)
        
        obs_scan = obs[:, self.num_prop: self.num_prop+self.num_scan]
        scan_latent = self.scan_encoder(obs_scan)

        obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)

        backbone_input = torch.cat([obs_prop_scan, priv_explicit, latent], dim=-1)
        backbone_output = self.actor_body(backbone_input)
        return backbone_output 
    
    def set_models_from_actor(self, actor):
        self.adaptation_module = actor.adaptation_module
        self.actor_body = actor.actor_body
        self.scan_encoder = actor.scan_encoder
        #self.estimator = actor.estimator
    
    @torch.jit.export
    def get_scandot_latent(self,obs):
        scan_dots = obs[:, self.num_prop: self.num_prop+self.num_scan]
        scandot_latent = self.scan_encoder(scan_dots)
        return scandot_latent
    
class HardwareActorNN(nn.Module):
    def __init__(   self,  
                    num_prop,
                    num_scan,
                    num_priv_explicit,
                    num_hist,
                    vision_model_type,
                    load_path = None,
                    scan_encoder_dims=[128, 64, 32],
                    depth_encoder_hidden_dim=512):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_priv_explicit = num_priv_explicit

        if load_path is not None:
            self.adaptation_module = torch.jit.load(load_path+"/adaptation_module_latest.jit", map_location="cpu").eval()
            self.actor_body = torch.jit.load(load_path+"/body_latest.jit", map_location="cpu").eval()
            self.estimator = torch.jit.load(load_path+"/estimator_latest.jit", map_location="cpu").eval()
        else:
            self.adaptation_module = None
            self.actor_body = None
            # self.estimator = None

    def forward(self, obs, depth_latent, yaw):
        obs[:, 6:8] = yaw
        priv_explicit = self.estimator(obs[:, :self.num_prop])
        #priv_explicit = obs[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.num_priv_explicit]
        
        observation_history = obs[:, -self.num_hist:]
        latent = self.adaptation_module(observation_history)

        obs_prop_scan = torch.cat([obs[:, :self.num_prop], depth_latent], dim=1)

        backbone_input = torch.cat([obs_prop_scan, priv_explicit, latent], dim=-1)
        backbone_output = self.actor_body(backbone_input)
        return backbone_output 
    
    @torch.jit.export
    def get_depth_latent_and_yaw(self, depth, obs):
        depth_latent_and_yaw = self.depth_encoder(depth, obs)
        return depth_latent_and_yaw
    

    def set_models_from_teacher(self, scandot_teacher:HardwareScandotNN):
        self.adaptation_module = deepcopy(scandot_teacher.adaptation_module)
        self.actor_body = deepcopy(scandot_teacher.actor_body)
        #self.estimator = scandot_teacher.estimator




class HardwareVisionNN(nn.Module):
    def __init__(   self,  
                    num_prop,
                    num_scan,
                    num_priv_explicit,
                    num_hist,
                    vision_model_type,
                    load_path = None,
                    scan_encoder_dims=[128, 64, 32],
                    depth_encoder_hidden_dim=512):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_priv_explicit = num_priv_explicit

        if load_path is not None:
            self.adaptation_module = torch.jit.load(load_path+"/adaptation_module_latest.jit", map_location="cpu").eval()
            self.actor_body = torch.jit.load(load_path+"/body_latest.jit", map_location="cpu").eval()
            self.estimator = torch.jit.load(load_path+"/estimator_latest.jit", map_location="cpu").eval()
        else:
            self.adaptation_module = None
            self.actor_body = None
            # self.estimator = None
        
        self.vision_model_type = vision_model_type
        depth_encoder = None

        if self.vision_model_type == 'depth':
            print('Using depth backbone')
            depth_backbone = DepthOnlyFCBackbone58x87(self.num_prop, scan_encoder_dims[-1], depth_encoder_hidden_dim)
        
        elif self.vision_model_type == 'rgb':
            print('Using rgb backbone')
            depth_backbone = RGBOnlyFCBackbone58x87(self.num_prop, scan_encoder_dims[-1],depth_encoder_hidden_dim)
      
        elif self.vision_model_type =='classifier':
            print('Using depth classifier encoder')
            depth_backbone = DepthOnlyFCBackbone58x87(self.num_prop, 32, depth_encoder_hidden_dim)
            depth_encoder = RecurrentDepthBackboneClassifier(depth_backbone, self.num_prop)

        if depth_encoder is None:
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, self.num_prop)
        else:
            self.depth_encoder = depth_encoder

    def forward(self, obs, depth_latent):
        # priv_explicit = self.estimator(obs[:, :self.num_prop])
        priv_explicit = obs[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.num_priv_explicit]
        
        observation_history = obs[:, -self.num_hist:]
        latent = self.adaptation_module(observation_history)

        obs_prop_scan = torch.cat([obs[:, :self.num_prop], depth_latent], dim=1)

        backbone_input = torch.cat([obs_prop_scan, priv_explicit, latent], dim=-1)
        backbone_output = self.actor_body(backbone_input)
        return backbone_output 
    
    @torch.jit.export
    def get_depth_latent_and_yaw(self, depth, obs):
        depth_latent_and_yaw = self.depth_encoder(depth, obs)
        return depth_latent_and_yaw
    

    def set_models_from_teacher(self, scandot_teacher:HardwareScandotNN):
        self.adaptation_module = deepcopy(scandot_teacher.adaptation_module)
        self.actor_body = deepcopy(scandot_teacher.actor_body)
        #self.estimator = scandot_teacher.estimator
