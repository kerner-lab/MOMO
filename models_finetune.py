
import numpy as np
import random
import torch
from torch import nn

from models_pretrain import *


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

        # Determine the number of output neurons based on the task
        self.num_classes = config["num_classes"]
        out_features = 1 if self.num_classes == 2 else self.num_classes

        # Add a fully connected layer for classification
        self.fc = nn.Linear(num_features, out_features)

        # Add the appropriate activation function
        self.activation = nn.Sigmoid() if self.num_classes == 2 else nn.Softmax(dim=1)

    def forward(self, x):
        features = self.encoder(x)
        if len(features.shape) != 2:
            # Global average pooling
            features = torch.mean(features, dim=[2, 3])
        logits = self.fc(features)
        return self.activation(logits)


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

        # Add the appropriate activation function
        # self.activation = nn.Sigmoid() if self.num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        features = self.encoder(x)
        x = self.upsample(features)
        return x


def create_finetune_model(train_model, which_pretraining, config, pretrained_path, device):

    if "classification" in config["task_type"]:

        if which_pretraining == "scratch_training":
            model = Classification(train_model=train_model, if_pretrained=False, config=config, device=device)

        if which_pretraining == "imagenet_pretrained":
            model = Classification(train_model=train_model, if_pretrained=True, config=config, device=device)

        if which_pretraining == "finetuning":
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

        if which_pretraining == "imagenet_pretrained":
            model = Segmentation(train_model=train_model, if_pretrained=True, config=config, device=device)

        if which_pretraining == "finetuning":
            model = Segmentation(train_model=train_model, if_pretrained=False, config=config, device=device)
            # Load pre-trained weights if provided
            state_dict = torch.load(pretrained_path)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.encoder.load_state_dict(state_dict, strict=False)

        return model

