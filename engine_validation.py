
import os
import torch
import torch.nn as nn
from typing import Iterable

from collections import OrderedDict
from models_pretrain import *
from task_vectors.task_vectors import TaskVector


def model_validation_single_pretrained(output_dir: str,
                     model_path: dict, pre_trained_model: str,
                     val_dataloader: Iterable, device: torch.device):

    encoders = {}
    decoders = {}

    ### Load all individually pre-trained model
    for model, checkpoint in model_path.items():
        # Load encoder
        encoder = create_model(
            train_model=model.split("_")[-1],
            model_unit="encoder",
            device=device,
            if_pretrained=False
        )
        encoder_state_dict = torch.load(os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), f"encoder_epoch_{checkpoint}.pth"))
        if isinstance(encoder_state_dict, OrderedDict):
            encoder.load_state_dict(encoder_state_dict)
        else:
            encoder = encoder_state_dict
        encoders[model] = encoder.to(device)
        # Load decoder
        decoder = create_model(
            train_model=model.split("_")[-1],
            model_unit="decoder",
            device=device,
            if_pretrained=False
        )
        decoder_state_dict = torch.load(os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), f"decoder_epoch_{checkpoint}.pth"))
        if isinstance(decoder_state_dict, OrderedDict):
            decoder.load_state_dict(decoder_state_dict)
        else:
            decoder = decoder_state_dict
        decoders[model] = decoder.to(device)

    ### Create combined model using task_vectors
    task_vectors = [
        TaskVector(pre_trained_model, os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), "encoder_epoch_"+checkpoint+".pth"))
        for model, checkpoint in model_path.items()
    ]
    task_vector_sum = sum(task_vectors)
    combined_encoder = task_vector_sum.apply_to(pre_trained_model, model.split("_")[-1], device, scaling_coef=0.5).to(device)

    combined_model_name = "_".join([f"{model}_{checkpoint}" for model, checkpoint in model_path.items()])
    os.makedirs(os.path.join(output_dir, "combined_models"), exist_ok=True)
    torch.save(combined_encoder.state_dict(), os.path.join(output_dir, "combined_models", f"{combined_model_name}.pth"))

    ### MSE Loss for calculating reconstruction error
    mse_loss = nn.MSELoss()

    ### Initialize dictionaries to store errors for each model
    model_errors = {model_key: [] for model_key in encoders.keys()}

    with torch.no_grad():

        for model_key in encoders.keys():
            encoder = encoders[model_key]
            decoder = decoders[model_key]

            combine_errors = []
            for images in val_dataloader:
                images = images.to(device)

                # Non-combined model reconstruction
                latent = encoder(images)
                reconstruction = decoder(latent)
                error = mse_loss(reconstruction, images)
                model_errors[model_key].append(error.item())

                # Combined reconstruction
                combine_latent = combined_encoder(images)
                combine_reconstruction = decoder(combine_latent)
                combine_error = mse_loss(combine_reconstruction, images)
                combine_errors.append(combine_error.item())

            avg_combine_error = sum(combine_errors) / len(combine_errors)
            print(f"Average reconstruction error on task vector encoder and {model_key} decoder: {avg_combine_error:.4f}")

    ### Calculate and print average errors
    for model_key, errors in model_errors.items():
        avg_error = sum(errors) / len(errors)
        print(f"Average {model_key} reconstruction error: {avg_error:.4f}")


def model_validation_jointly_pretrained(output_dir: str,
                     model_path: dict, jointly_trained_model_path: str, pre_trained_model: str,
                     val_dataloader: Iterable, device: torch.device):

    encoders = {}
    decoders = {}

    ### Load jointly pre-trained model
    for model, checkpoint in jointly_trained_model_path.items():
        # Load encoder
        encoder = create_model(
            train_model=model.split("_")[-1],
            model_unit="encoder",
            device=device,
            if_pretrained=False
        )
        encoder_state_dict = torch.load(os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), f"encoder_epoch_{checkpoint}.pth"))
        if isinstance(encoder_state_dict, OrderedDict):
            encoder.load_state_dict(encoder_state_dict)
        else:
            encoder = encoder_state_dict
        encoders[model] = encoder.to(device)
        # Load decoder
        decoder = create_model(
            train_model=model.split("_")[-1],
            model_unit="decoder",
            device=device,
            if_pretrained=False
        )
        decoder_state_dict = torch.load(os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), f"decoder_epoch_{checkpoint}.pth"))
        if isinstance(decoder_state_dict, OrderedDict):
            decoder.load_state_dict(decoder_state_dict)
        else:
            decoder = decoder_state_dict
        decoders[model] = decoder.to(device)

    ### Create combined model using task_vectors
    task_vectors = [
        TaskVector(pre_trained_model, os.path.join(output_dir, "pretraining", model.split("_")[-1], "_".join(model.split("_")[:-1]), "encoder_epoch_"+checkpoint+".pth"))
        for model, checkpoint in model_path.items()
    ]
    task_vector_sum = sum(task_vectors)
    combined_encoder = task_vector_sum.apply_to(pre_trained_model, model.split("_")[-1], device, scaling_coef=0.5).to(device)

    combined_model_name = "_".join([f"{model}_{checkpoint}" for model, checkpoint in model_path.items()])
    os.makedirs(os.path.join(output_dir, "combined_models"), exist_ok=True)
    torch.save(combined_encoder.state_dict(), os.path.join(output_dir, "combined_models", f"{combined_model_name}.pth"))

    ### MSE Loss for calculating reconstruction error
    mse_loss = nn.MSELoss()

    ### Initialize dictionaries to store errors for each model
    model_errors = {model_key: [] for model_key in encoders.keys()}

    with torch.no_grad():

        for model_key in encoders.keys():
            encoder = encoders[model_key]
            decoder = decoders[model_key]

            combine_errors = []
            for images in val_dataloader:
                images = images.to(device)

                # Non-combined model reconstruction
                latent = encoder(images)
                reconstruction = decoder(latent)
                error = mse_loss(reconstruction, images)
                model_errors[model_key].append(error.item())

                # Combined reconstruction
                combine_latent = combined_encoder(images)
                combine_reconstruction = decoder(combine_latent)
                combine_error = mse_loss(combine_reconstruction, images)
                combine_errors.append(combine_error.item())

            avg_combine_error = sum(combine_errors) / len(combine_errors)
            print(f"Average reconstruction error on task vector encoder and {model_key} decoder: {avg_combine_error:.4f}")

    ### Calculate and print average errors
    for model_key, errors in model_errors.items():
        avg_error = sum(errors) / len(errors)
        print(f"Average {model_key} reconstruction error: {avg_error:.4f}")
