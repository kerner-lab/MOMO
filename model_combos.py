
### Importing Libraries
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from collections import OrderedDict
import torchvision.models as torchModels

from models_pretrain import *


### Parse command line arguments
def get_args_parser():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--model_dir", type=str,
                           default="/scratch-edge/large-mars-model/models/task_arithmetic/pretraining", required=False)
    argparser.add_argument("--imagenet_model_dir", type=str,
                           default="/scratch-edge/large-mars-model/models/task_arithmetic/imagenet", required=False)
    argparser.add_argument("--train_model", type=str, default="resnet34", required=False,
                            help="Available choices: resnet34, squeezenet1-1, efficientnet-v2-m, vit-b-16, vit-b-32, vit-l-16, vit-l-32")
    argparser.add_argument("--ctx_checkpoint", type=str, default="500", required=True)
    argparser.add_argument("--hirise_checkpoint", type=str, default="500", required=True)
    argparser.add_argument("--themis_checkpoint", type=str, default="500", required=True)
    argparser.add_argument("--output_dir_normal", type=str,
                            default="/scratch-edge/large-mars-model/models/task_arithmetic/customized_models_normal", required=False)
    argparser.add_argument("--output_dir_task_vectors", type=str,
                            default="/scratch-edge/large-mars-model/models/task_arithmetic/customized_models_task_vectors", required=False)
    argparser.add_argument("--output_dir_try", type=str,
                            default="/scratch-edge/large-mars-model/models/task_arithmetic/customized_models_try", required=False)

    return argparser


### Initializing Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Helper Functions
def modelDictToTensor(model_state_dict: OrderedDict) -> (torch.Tensor, list):
    """
    Convert model weights (OrderedDict) to a flattened tensor and save metadata.
    Args:
        model_state_dict (OrderedDict): Model weights.
    Returns:
        torch.Tensor: Flattened tensor of weights.
        list: Metadata containing layer names and shapes.
    """
    layer_weights = []
    metadata = []

    for name, weights in model_state_dict.items():
        # Save layer name and original shape
        metadata.append((name, weights.shape))
        layer_weights.append(torch.flatten(weights))

    return torch.cat(layer_weights), metadata


def tensorToModelDict(tensor: torch.Tensor, metadata: list) -> OrderedDict:
    """
    Convert flattened tensor back into OrderedDict using metadata.
    Args:
        tensor (torch.Tensor): Flattened tensor of weights.
        metadata (list): Metadata containing layer names and shapes.
    Returns:
        OrderedDict: Reconstructed model weights.
    """
    reconstructed_dict = OrderedDict()
    start = 0

    for name, shape in metadata:
        # Compute the number of elements in this layer (convert to int)
        num_elements = int(torch.prod(torch.tensor(shape)))

        # Extract the corresponding portion of the tensor
        layer_weights = tensor[start : start + num_elements]

        # Reshape to original shape and add to OrderedDict
        reconstructed_dict[name] = layer_weights.view(shape)
        start += num_elements

    return reconstructed_dict


### Model combo function
def getMagnitude(m: torch.tensor) -> torch.float:
    val = torch.sum(torch.pow(m, 2))**0.5
    return val.item()

def modelCombo1(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor, k:float = 1) -> torch.Tensor:
    m1_mag = getMagnitude(m1)
    m2_mag = getMagnitude(m2)
    m3_mag = getMagnitude(m3)
    final = k * ((m1_mag + m2_mag + m3_mag)/3) * ((m1/m1_mag) + (m2/m2_mag) + (m3/m3_mag))/3
    return final

def modelCombo2(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor, k:float = 1) -> torch.Tensor:
    final = k * (m1 + m2 + m3) / 3
    return final

def modelCombo3(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor, k:float = 1) -> torch.Tensor:
    m1_mag = getMagnitude(m1)
    m2_mag = getMagnitude(m2)
    m3_mag = getMagnitude(m3)
    final = k * ((m1 * (m2_mag + m3_mag)) + (m2 * (m3_mag + m1_mag)) + (m3 * (m2_mag + m1_mag)))/(2* (m1_mag + m2_mag + m3_mag))
    return final


def main(args):

    model_name = "hirise_" + args.train_model + "_" + args.hirise_checkpoint + "_ctx_" + args.train_model + "_" + args.ctx_checkpoint + "_themis_" + args.train_model + "_" + args.themis_checkpoint

    ### Load models
    ctx_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "ctx", f"encoder_epoch_{args.ctx_checkpoint}.pth"))
    hirise_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "hirise", f"encoder_epoch_{args.hirise_checkpoint}.pth"))
    themis_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "themis", f"encoder_epoch_{args.themis_checkpoint}.pth"))

    ctx_model, metadata = modelDictToTensor(ctx_checkpoint)
    hirise_model, metadata = modelDictToTensor(hirise_checkpoint)
    themis_model, metadata = modelDictToTensor(themis_checkpoint)

    ''' Without task vectors '''
    ### Customized Model Combo 1
    modified_model = modelCombo1(ctx_model, hirise_model, themis_model)
    reconstructed_model = tensorToModelDict(modified_model, metadata)

    encoder = create_model(
        train_model=args.train_model,
        model_unit="encoder",
        device=DEVICE,
        if_pretrained=False
    )
    encoder.load_state_dict(reconstructed_model)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir_normal, model_name+"_mc1.pth"))

    ### Customized Model Combo 2
    modified_model = modelCombo2(ctx_model, hirise_model, themis_model)
    reconstructed_model = tensorToModelDict(modified_model, metadata)

    encoder = create_model(
        train_model=args.train_model,
        model_unit="encoder",
        device=DEVICE,
        if_pretrained=False
    )
    encoder.load_state_dict(reconstructed_model)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir_normal, model_name+"_mc2.pth"))

    ''' With task vectors '''
    ### Load pre-trained ImageNet version
    base_model = torch.load(os.path.join(args.imagenet_model_dir, f"pre_trained_{args.train_model}.pth"))
    imagenet_model, metadata = modelDictToTensor(base_model)

    ### Customized Model Combo 1
    modified_model = imagenet_model + modelCombo1((ctx_model - imagenet_model), (hirise_model - imagenet_model), (themis_model - imagenet_model))
    reconstructed_model = tensorToModelDict(modified_model, metadata)

    encoder = create_model(
        train_model=args.train_model,
        model_unit="encoder",
        device=DEVICE,
        if_pretrained=False
    )
    encoder.load_state_dict(reconstructed_model)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir_task_vectors, model_name+"_mc1.pth"))

    ### Customized Model Combo 2
    modified_model = imagenet_model + modelCombo1((ctx_model - imagenet_model), (hirise_model - imagenet_model), (themis_model - imagenet_model))
    reconstructed_model = tensorToModelDict(modified_model, metadata)

    encoder = create_model(
        train_model=args.train_model,
        model_unit="encoder",
        device=DEVICE,
        if_pretrained=False
    )
    encoder.load_state_dict(reconstructed_model)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir_task_vectors, model_name+"_mc2.pth"))


    ''' Pruned averaging model '''
    ctx_es_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "ctx", f"encoder_epoch_{289}.pth"))
    hirise_es_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "hirise", f"encoder_epoch_{151}.pth"))
    themis_es_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "themis", f"encoder_epoch_{193}.pth"))

    ctx_es_model, metadata = modelDictToTensor(ctx_es_checkpoint)
    hirise_es_model, metadata = modelDictToTensor(hirise_es_checkpoint)
    themis_es_model, metadata = modelDictToTensor(themis_es_checkpoint)

    ctx_final_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "ctx", f"encoder_epoch_{500}.pth"))
    hirise_final_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "hirise", f"encoder_epoch_{500}.pth"))
    themis_final_checkpoint = torch.load(os.path.join(args.model_dir, args.train_model, "themis", f"encoder_epoch_{500}.pth"))

    ctx_final_model, metadata = modelDictToTensor(ctx_final_checkpoint)
    hirise_final_model, metadata = modelDictToTensor(hirise_final_checkpoint)
    themis_final_model, metadata = modelDictToTensor(themis_final_checkpoint)

    ### Customized Model Combo 4
    # modified_model = modelCombo2((ctx_model+ctx_es_model)/2, (hirise_model+hirise_es_model)/2, (themis_model+themis_es_model)/2)
    # model_name = f"hirise_ctx_themis_{args.train_model}.pth"
    modified_model = modelCombo2((ctx_model+ctx_es_model+ctx_final_model)/3, (hirise_model+hirise_es_model+hirise_final_model)/3, (themis_model+themis_es_model)/3)
    model_name = f"hirise_ctx_themis_{args.train_model}_all_3.pth"
    reconstructed_model = tensorToModelDict(modified_model, metadata)

    encoder = create_model(
        train_model=args.train_model,
        model_unit="encoder",
        device=DEVICE,
        if_pretrained=False
    )
    encoder.load_state_dict(reconstructed_model)
    torch.save(encoder.state_dict(), os.path.join(args.output_dir_try, model_name))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)


# python model_combos.py --train_model resnet34 --ctx_checkpoint 500 --hirise_checkpoint 11 --themis_checkpoint 17
# python model_combos.py --train_model efficientnet-v2-m --ctx_checkpoint 500 --hirise_checkpoint 6 --themis_checkpoint 29
# python model_combos.py --train_model vit-b-16 --ctx_checkpoint 500 --hirise_checkpoint 9 --themis_checkpoint 171
