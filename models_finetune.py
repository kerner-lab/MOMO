

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from models_pretrain import *
from models.models_vit import vit_customized
from models.unetformer import Segmentation_ViT_UNetFormer
from utils.pos_embed import interpolate_pos_embed



def create_finetune_model_vit(train_model, which_finetuning, drop_path, global_pool, config, pretrained_path, finetuning_type, device, args):

    ### Define and embedding dimension of the encoder
    if train_model == "vit-t-16":
        encoder_output_dim = 192
    elif train_model == "vit-s-16":
        encoder_output_dim = 384
    elif train_model == "vit-b-16":
        encoder_output_dim = 768
    elif train_model == "vit-l-16":
        encoder_output_dim = 1024
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-t-16, vit-s-16, vit-b-16, vit-l-16")

    ### Define model
    if train_model == "vit-t-16":
        model = vit_customized(
            img_size=config["input_size"][0], patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=12, num_heads=3,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-s-16":
        model = vit_customized(
            img_size=config["input_size"][0], patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=12, num_heads=6,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-b-16":
        model = vit_customized(
            img_size=config["input_size"][0], patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=12, num_heads=12,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    elif train_model == "vit-l-16":
        model = vit_customized(
            img_size=config["input_size"][0], patch_size=16, in_chans=3,
            embed_dim=encoder_output_dim, depth=24, num_heads=16,
            drop_path_rate=drop_path, global_pool=global_pool, num_classes=config["num_classes"]
        )
    else:
        raise ValueError(f"Unknown model type: {train_model}. Available types: vit-b-16, vit-b-32, vit-l-16, vit-l-32")

    if which_finetuning == "scratch_training":
        ### Do nothing to train model from scratch
        pass

    else:
        ### Load pre-trained weights if provided
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        print("\nLoad pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        ### Interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        ### Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        ### Manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if "classification" in config["task_type"]:
        if finetuning_type == "lp":
            if which_finetuning in ["imagenet_pretrained", "mae_imagenet_pretrained", "checkpoint"]:
                for param in model.parameters():
                    param.requires_grad = False

                if hasattr(model, 'head'):
                    for param in model.head.parameters():
                        param.requires_grad = True
                if hasattr(model, 'fc_norm'):
                    for param in model.fc_norm.parameters():
                        param.requires_grad = True

        model = model.to(device)
        return model

    if "segmentation" in config["task_type"]:
        model.head = nn.Identity()
        if finetuning_type == "lp":
            if which_finetuning in ["imagenet_pretrained", "mae_imagenet_pretrained", "checkpoint"]:
                for param in model.parameters():
                    param.requires_grad = False

        model = Segmentation_ViT_UNetFormer(encoder=model, encoder_output_dim=encoder_output_dim, num_classes=config["num_classes"], decoder_channels=64, window_size=8, dropout=0.1)

        ''' TODO
        ### Ensure decoder parameters remain trainable (they should be by default)
        ### But let's explicitly check - decoder parameters include:
        ### - projection_layers (ModuleDict)
        ### - spatial_projection
        ### - stage1, stage2, stage3, stage4
        ### - final_conv

        decoder_params = []
        for name, param in model.named_parameters():
            if not name.startswith('encoder'):  ### Everything that's not the encoder
                decoder_params.append(param)

        for param in decoder_params:
            param.requires_grad = True
        '''

        model = model.to(device)
        return model
