
import torch
import torch.nn as nn
from typing import Dict, Type

from models.resnet import ResNetEncoder, ResNetDecoder, ResNetAutoencoder
from models.squeezenet import SqueezeNetEncoder, SqueezeNetDecoder, SqueezeNetAutoencoder
from models.efficientnet import EfficientNetEncoder, EfficientNetDecoder, EfficientNetAutoencoder
from models.mae import mae_vit_customized


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

    if "vit" in train_model:

        if train_model == "vit-b-16":
            model = mae_vit_customized(
                img_size=224, patch_size=16, in_chans=3,
                embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False
            )
        elif train_model == "vit-b-32":
            model = mae_vit_customized( 
                img_size=224, patch_size=32, in_chans=3,
                embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False
            )
        elif train_model == "vit-l-16":
            model = mae_vit_customized(
                img_size=224, patch_size=16, in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False
            )
        elif train_model == "vit-l-32":
            model = mae_vit_customized(
                img_size=224, patch_size=32, in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False
            )
        else:
            raise ValueError(f"Unknown model type: {train_model}. Available types: vit-b-16, vit-b-32, vit-l-16, vit-l-32")
            return

        # Move model to device
        model = model.to(device)

    else:
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
