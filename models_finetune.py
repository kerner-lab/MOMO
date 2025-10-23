
import numpy as np
import random
import torch
from torch import nn
import os

from timm.models.layers import trunc_normal_

from models_pretrain import *
from models.models_vit import vit_customized
from utils.pos_embed import interpolate_pos_embed

from models.unetformer import Segmentation_ViT_UNetFormer

class Classification(nn.Module):

    def __init__(self, train_model, if_pretrained, config, device):
        super(Classification, self).__init__()
        self.encoder = create_model(
            train_model=train_model,
            model_unit="encoder",
            device=device,
            if_pretrained=if_pretrained
        )
        if if_pretrained:
            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get the output features of the encoder
        with torch.no_grad():
            if "vit" in train_model:
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
            else:
                dummy_input = torch.randn(1, 3, config["input_size"][0], config["input_size"][1]).to(device)
            encoder_output = self.encoder(dummy_input)
            num_features = encoder_output.shape[1]

        # Add a fully connected layer for classification
        self.fc = nn.Linear(num_features, config["num_classes"])

    def forward(self, x):
        features = self.encoder(x)
        if len(features.shape) != 2:
            # Global average pooling
            features = torch.mean(features, dim=[2, 3])
        logits = self.fc(features)
        return logits


class Segmentation(nn.Module):

    def __init__(self, train_model, if_pretrained, config, device):
        super(Segmentation, self).__init__()
        self.encoder = create_model(
            train_model=train_model,
            model_unit="encoder",
            device=device,
            if_pretrained=if_pretrained
        )
        if if_pretrained:
            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Determine the number of output neurons based on the task
        self.num_classes = config["num_classes"]
        out_features = self.num_classes

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_features, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        x = self.upsample(features)
        return x


class Segmentation_ViT(nn.Module):

    def __init__(self, encoder, encoder_output_dim):
        super(Segmentation_ViT, self).__init__()
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        
        # Decoder dimensions
        decoder_base_channels = 256
        self.decoder_base_channels = decoder_base_channels
        num_output_classes = 1
        
        # Dynamic linear projections for different input sizes (from attached code)
        self.projection_layers = nn.ModuleDict()
        
        # Spatial projection after linear projection
        self.spatial_projection = nn.Conv2d(encoder_output_dim, decoder_base_channels, kernel_size=1)
        
        # Decoder stages (your architecture)
        self.stage1 = self._make_decoder_stage(decoder_base_channels, decoder_base_channels // 2)
        self.stage2 = self._make_decoder_stage(decoder_base_channels // 2, decoder_base_channels // 4)
        self.stage3 = self._make_decoder_stage(decoder_base_channels // 4, decoder_base_channels // 8)
        self.stage4 = self._make_decoder_stage(decoder_base_channels // 8, decoder_base_channels // 16)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(decoder_base_channels // 16, num_output_classes, kernel_size=1)

    def _make_decoder_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _get_feature_map_size_and_stages(self, input_size):
        """From attached code"""
        if input_size <= 128:
            initial_feature_size = 8
            num_stages = 4
        elif input_size <= 256:
            initial_feature_size = 16
            num_stages = 4
        elif input_size <= 512:
            initial_feature_size = 32
            num_stages = 4
        else:
            initial_feature_size = 32
            num_stages = 5
            
        return initial_feature_size, num_stages
    
    def _get_or_create_projection(self, input_size):
        """From attached code - creates dynamic linear projection"""
        size_key = str(input_size)
        
        if size_key not in self.projection_layers:
            initial_feature_size, _ = self._get_feature_map_size_and_stages(input_size)
            projected_feature_dim = initial_feature_size * initial_feature_size * self.encoder_output_dim
            
            self.projection_layers[size_key] = nn.Linear(
                self.encoder_output_dim, projected_feature_dim
            ).to(next(self.parameters()).device)
            
        return self.projection_layers[size_key]

    @property
    def blocks(self):
        return self.encoder.blocks

    def no_weight_decay(self):
        return self.encoder.no_weight_decay()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Get encoder features
        features = self.encoder(x)  # Assume [B, C] for global pooled encoder
        
        # Get appropriate projection layer and feature map size
        initial_feature_size, num_stages = self._get_feature_map_size_and_stages(H)
        projection_layer = self._get_or_create_projection(H)
        
        # Project to spatial feature maps using linear layer
        features = projection_layer(features)
        features = features.reshape(B, self.encoder_output_dim, 
                                   initial_feature_size, initial_feature_size)
        
        # Project to decoder channels
        x = self.spatial_projection(features)
        
        # Apply decoder stages based on needed upsampling
        if num_stages >= 1:
            x = self.stage1(x)
        if num_stages >= 2:
            x = self.stage2(x)
        if num_stages >= 3:
            x = self.stage3(x)
        if num_stages >= 4:
            x = self.stage4(x)
        
        # Final prediction
        x = self.final_conv(x)
        
        # Ensure output matches input size exactly
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        
        return x



def create_finetune_model(train_model, which_pretraining, config, pretrained_path, device):

    if "classification" in config["task_type"]:

        if which_pretraining == "scratch_training":
            model = Classification(train_model=train_model, if_pretrained=False, config=config, device=device)

        elif which_pretraining == "imagenet_pretrained":
            model = Classification(train_model=train_model, if_pretrained=True, config=config, device=device)

        else:
            model = Classification(train_model=train_model, if_pretrained=False, config=config, device=device)
            # Load pre-trained weights if provided
            state_dict = torch.load(pretrained_path)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.encoder.load_state_dict(state_dict, strict=False)

        return model

    if "segmentation" in config["task_type"]:

        if which_pretraining == "scratch_training":
            model = Segmentation(train_model=train_model, if_pretrained=False, config=config, device=device)

        elif which_pretraining == "imagenet_pretrained":
            model = Segmentation(train_model=train_model, if_pretrained=True, config=config, device=device)

        else:
            model = Segmentation(train_model=train_model, if_pretrained=False, config=config, device=device)
            # Load pre-trained weights if provided
            state_dict = torch.load(pretrained_path)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.encoder.load_state_dict(state_dict, strict=False)

        return model


def create_finetune_model_vit(train_model, which_pretraining, drop_path, global_pool, config, pretrained_path, device, args):

    if "vit-b" in train_model:
        encoder_output_dim = 768
    elif "vit-l" in train_model:
        encoder_output_dim = 1024
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-b-16, vit-b-32, vit-l-16, vit-l-32")

    img_size = config["input_size"][0]
    if train_model == "vit-b-16":
        model = vit_customized(
            img_size=img_size, patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=12, num_heads=12,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-b-32":
        model = vit_customized(
            img_size=img_size, patch_size=32, in_chans=3,
            embed_dim=encoder_output_dim, depth=12, num_heads=12,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-l-16":
        model = vit_customized(
            img_size=img_size, patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=24, num_heads=16,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-l-32":
        model = vit_customized(
            img_size=img_size, patch_size=32, in_chans=3,
            embed_dim=encoder_output_dim, depth=24, num_heads=16,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-b-16, vit-b-32, vit-l-16, vit-l-32")
        return

    if which_pretraining == "scratch_training":
        pass

    else:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        print("\nLoad pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        if not which_pretraining == "evaluation":
            if global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if "classification" in config["task_type"]:
        model = model.to(device)
        return model

    if "segmentation" in config["task_type"]:
        model.head = nn.Identity()
        # model = Segmentation_ViT(model, encoder_output_dim)
        model = Segmentation_ViT_UNetFormer(encoder=model, encoder_output_dim=encoder_output_dim, num_classes=config["num_classes"], decoder_channels=64, window_size=8, dropout=0.1)
        model = model.to(device)
        return model
