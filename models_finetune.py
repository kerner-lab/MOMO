
import numpy as np
import random
import torch
from torch import nn
import os

from timm.models.layers import trunc_normal_

from models_pretrain import *
from models.models_vit import vit_customized
from utils.pos_embed import interpolate_pos_embed


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
        out_features = 1 if self.num_classes == 1 else 3

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

    def __init__(self, encoder):
        super(Segmentation_ViT, self).__init__()
        self.encoder = encoder

        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        # )
        latent_dim = 1024
        hidden_dim = 256
        output_channels = 1
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim // 8, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    @property
    def blocks(self):
        return self.encoder.blocks

    def no_weight_decay(self):
        return self.encoder.no_weight_decay()

    def forward(self, x):
        features = self.encoder(x)
        print(features.shape)
        x = self.upsample(features)
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

    if train_model == "vit-b-16":
        model = vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=768, depth=12, num_heads=12,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-b-32":
        model = vit_customized(
            img_size=224, patch_size=32, in_chans=3,
            embed_dim=768, depth=12, num_heads=12,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-l-16":
        model = vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=1024, depth=24, num_heads=16,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-l-32":
        model = vit_customized(
            img_size=224, patch_size=32, in_chans=3,
            embed_dim=1024, depth=24, num_heads=16,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-b-16, vit-b-32, vit-l-16, vit-l-32")
        return

    if which_pretraining == "scratch_training":
        pass

    else:
        if which_pretraining == "imagenet_pretrained":
            if "vit-b" in train_model:
                checkpoint = torch.load((os.path.join(pretrained_path, "mae_pretrain_vit_base.pth")), map_location='cpu')
            else:
                checkpoint = torch.load(os.path.join(pretrained_path, "mae_pretrain_vit_large.pth"), map_location='cpu')
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrained_path)
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
        model = Segmentation_ViT(model)
        model = model.to(device)
        return model
