import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import open_clip


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, num_prop, use_l2_norm=False) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.use_l2_norm = use_l2_norm
        if self.use_l2_norm:
            print('Using L2 normalization')

        if num_prop == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + num_prop, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))

        if self.use_l2_norm:
            depth_latent = F.normalize(depth_latent, p=2, dim=1)
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    @torch.jit.export
    def reset_hidden_states(self):
        self.hidden_states[:] = 0

class LiquidBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        #self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
    
        self.rnn = CfC(32, 512, batch_first=True,)
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
        

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent

class RGBOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=3):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]

            # 62, 400
            nn.Linear(64 * 25 * 39, 128),
            #nn.Linear(102400, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class RGBLargeFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=3):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [3, 224, 224]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=3),
            # [32, 222, 222]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 111, 111]
            activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            # [32, 107, 107]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 53, 53]
            activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            # [64, 49, 49]
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            # [64, 49, 49]

            # 62, 400
            nn.Linear(64 * 49 * 49, 128),
            #nn.Linear(102400, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent
    

class RecurrentDepthBackboneClassifier(nn.Module):
    def __init__(self, base_backbone, num_prop) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + num_prop, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_yaw = nn.Sequential(
                                nn.Linear(512, 2),
                                last_activation
                            )
        self.output_classifier = nn.Sequential(
                                nn.Linear(512, 10),
                                nn.Softmax(dim=1)
                            )
        self.hidden_states = None

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        yaw = self.output_yaw(depth_latent.squeeze(1))
        depth_classification = self.output_classifier(depth_latent.squeeze(1))

        output_cat = torch.cat((depth_classification, yaw), dim=-1)
        return output_cat

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class RGBMobileNetBackbone(nn.Module):
    def __init__(self, scandots_output_dim):
        super().__init__()
        #self.model,_ = clip.load("RN50")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout to match mnv2
            nn.Linear(in_features=1280, out_features=scandots_output_dim),
        )

        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor):
        latent = self.model(images)
        return latent
    
class RGBDinoBackbone(nn.Module):
    def __init__(self, scandots_output_dim):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        activation = nn.ELU()
        self.latent_compression = nn.Sequential(
            activation,
            # clip output -> latent dim
            nn.Linear(384, scandots_output_dim)
        )

    def forward(self, images: torch.Tensor):
        with torch.no_grad():
            image_features = self.model(images)
        latent = self.latent_compression(image_features)

        return latent

class RGBClipBackbone(nn.Module):
    
    def __init__(self, scandots_output_dim):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

        for param in self.model.parameters():
            param.requires_grad = False

        activation = nn.ELU()
        self.latent_compression = nn.Sequential(
            activation,
            # clip output -> latent dim
            nn.Linear(512, scandots_output_dim)
        )

    def forward(self, images: torch.Tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        latent = self.latent_compression(image_features)

        return latent