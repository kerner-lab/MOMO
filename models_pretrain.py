
import torch
from typing import Dict, Type

from models.resnet import ResNetEncoder, ResNetDecoder, ResNetAutoencoder
from models.squeezenet import SqueezeNetEncoder, SqueezeNetDecoder, SqueezeNetAutoencoder
from models.efficientnet import EfficientNetEncoder, EfficientNetDecoder, EfficientNetAutoencoder
from models.vit import ViTB16Encoder, ViTB32Encoder, ViTL16Encoder, ViTL32Encoder, ViTDecoder, ViTAutoencoder


MODEL_REGISTRY: Dict[str, Dict[str, Type]] = {
    "resnet34": {
        "encoder": ResNetEncoder,
        "decoder": ResNetDecoder,
        "autoencoder": ResNetAutoencoder
    },
    "squeezenet1-1": {
        "encoder": SqueezeNetEncoder,
        "decoder": SqueezeNetDecoder,
        "autoencoder": SqueezeNetAutoencoder
    },
    "efficientnet-v2-m": {
        "encoder": EfficientNetEncoder,
        "decoder": EfficientNetDecoder,
        "autoencoder": EfficientNetAutoencoder
    },
    "vit-b-16": {
        "encoder": ViTB16Encoder,
        "decoder": ViTDecoder,
        "autoencoder": ViTAutoencoder
    },
    "vit-b-32": {
        "encoder": ViTB32Encoder,
        "decoder": ViTDecoder,
        "autoencoder": ViTAutoencoder
    },
    "vit-l-16": {
        "encoder": ViTL16Encoder,
        "decoder": ViTDecoder,
        "autoencoder": ViTAutoencoder
    },
    "vit-l-32": {
        "encoder": ViTL32Encoder,
        "decoder": ViTDecoder,
        "autoencoder": ViTAutoencoder
    }
}

def create_model(
    train_model: str,
    model_unit: str,
    device: torch.device,
    if_pretrained: bool = False
) -> torch.nn.Module:
    
    """
    Create a model based on the specified architecture and type.

    Args:
        train_model: The architecture type (e.g., 'resnet34', 'squeezenet1_1')
        model_unit: The model component to train ('encoder', 'decoder', 'autoencoder')
        if_pretrained: Whether to use pretrained weights
        device: The device to put the model on

    Returns:
        The created model
    """

    if train_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {train_model}. Available types: {list(MODEL_REGISTRY.keys())}")

    if model_unit not in MODEL_REGISTRY[train_model]:
        raise ValueError(f"Unknown model component: {model_unit}. Available components: {list(MODEL_REGISTRY[train_model].keys())}")

    if model_unit == "autoencoder":
        encoder = MODEL_REGISTRY[train_model]["encoder"](if_pretrained).to(device)
        decoder = MODEL_REGISTRY[train_model]["decoder"]().to(device)
        model = MODEL_REGISTRY[train_model]["autoencoder"](encoder, decoder).to(device)
    elif model_unit == "encoder":
        model = MODEL_REGISTRY[train_model]["encoder"](if_pretrained).to(device)
    else:
        model = MODEL_REGISTRY[train_model]["decoder"]().to(device)

    return model
