
import torch
import torch.nn as nn

from models.mae import mae_vit_customized

def create_model(
    train_model: str,
    device: torch.device,
    args=None
) -> torch.nn.Module:
    
    """
    Create a model based on the specified architecture and type.

    Args:
        train_model: The architecture type (e.g., 'vit-t-16', 'vit-s-16', 'vit-b-16', 'vit-l-16')
        device: The device to put the model on

    Returns:
        The created model
    """

    if train_model == "vit-t-16":
        model = mae_vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=192, depth=12, num_heads=3,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
            args=args
        )
    elif train_model == "vit-s-16":
        model = mae_vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=384, depth=12, num_heads=6,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
            args=args
        )
    elif train_model == "vit-b-16":
        model = mae_vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
            args=args
        )
    elif train_model == "vit-l-16":
        model = mae_vit_customized(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
            args=args
        )
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-t-16, vit-s-16, vit-b-16, vit-l-16")

    ### Move model to device
    model = model.to(device)

    return model
