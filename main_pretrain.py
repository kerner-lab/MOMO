
import argparse
import os
import random
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from data_processing import prepare_dataloaders
from engine_pretrain import model_training
from engine_validation import model_validation_single_pretrained, model_validation_jointly_pretrained
from models_pretrain import create_model
from utils import *


### Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def parse_dict(arg_string):
    """ Parses a string of key=value,key=value into a dictionary. """
    # Split by commas to separate key-value pairs
    items = arg_string.split(',')

    # Split each pair by the ':' and construct the dictionary
    parsed_dict = {}
    for item in items:
        key, value = item.split(':')
        parsed_dict[key] = value

    return parsed_dict


### Parse command line arguments
def get_args_parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--if_training", default=False, required=False, action="store_true")
    argparser.add_argument("--if_validation", default=False, required=False, action="store_true")

    # Pre-training args
    argparser.add_argument("--data_dir", type=str, default="data", required=False)
    argparser.add_argument("--which_instrument", "--list", type=str, default="", required=False)
    argparser.add_argument("--val_split", type=float, default=0.1, required=False)
    argparser.add_argument("--train_model", type=str, default="resnet34", required=False,
                            help="Available choices: resnet34, squeezenet1-1, efficientnet-v2-m, vit-b-16, vit-b-32, vit-l-16, vit-l-32")
    argparser.add_argument("--imagenet_pretrain", default=False, required=False, action="store_true")
    argparser.add_argument("--backbone_weight", type=str, default="imagenet", required=False)
    argparser.add_argument("--num_epochs", type=int, default=500, required=False)
    argparser.add_argument("--batch_size", type=int, default=256, required=False)

    argparser.add_argument("--output_dir", type=str, default="models", required=False)

    argparser.add_argument("--wandb_enabled", default=False, required=False, action="store_true",
                        help="True value of this parameter assumes that you have wandb account")
    argparser.add_argument("--wandb_entity", type=str, default="mpurohi3", required=False,
                            help="Provide Wandb entity where plots will be available")
    argparser.add_argument("--wandb_project", type=str, default="LMM", required=False,
                            help="Provide Wandb project name for plots")

    # Validation args
    argparser.add_argument("--model_paths", type=str, default="", required=False,
                        help="Dictionary of pre-trained model paths for validation")
    argparser.add_argument("--jointly_trained_model_path", type=str, default="", required=False,
                           help="Path of pre-trained model which is pre-trained on multiiple instruments")
    argparser.add_argument("--pretrained_model_path", type=str, default="", required=False,
                           help="Path of pre-trained model (ImageNet, COCO)")
    argparser.add_argument("--val_data", type=str, default="", required=False,
                           help="Which data/instrument for validation")
    return argparser


def main(args):

    ### Check if either if_training or if_validation is True
    if not (args.if_training or args.if_validation):
        raise AssertionError("Either --if_training or --if_validation must be set to True!!")

    ### Check device type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Model training
    if args.if_training:

        ### Initialize data and model directory, unique name of current run
        args.which_instrument = [item for item in args.which_instrument.split(', ')]
        name_of_run = "_".join([each_instrument.lower() for each_instrument in args.which_instrument])
        output_dir = os.path.join(args.output_dir, "pretraining", args.train_model, name_of_run)
        os.makedirs(output_dir, exist_ok=True)

        ### Prepare dataloaders
        train_dataloader, val_dataloader = prepare_dataloaders(
            args.data_dir,
            args.which_instrument,
            args.val_split,
            args.batch_size
        )

        ### Initialize model, loss, optimizer and wandb
        model = create_model(
            train_model=args.train_model,
            model_unit="autoencoder",
            device=device,
            if_pretrained=args.imagenet_pretrain
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        if args.wandb_enabled:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=name_of_run + "_" + args.train_model,
                config={
                        "Training data samples": len(train_dataloader),
                        "Validation data samples": len(val_dataloader),
                        "Model": args.train_model,
                        "Epochs": args.num_epochs,
                        "Batch syize": args.batch_size,
                        "Optimizer": optimizer,
                        "Loss": criterion,
                        "Model path": output_dir
                    }
            )

        ### Training model
        model_training(
            model,
            train_dataloader, val_dataloader,
            args.num_epochs, device, output_dir,
            criterion, optimizer, wandb, args
        )

    ### Model validation
    if args.if_validation:
        assert args.model_paths, "Error: model_path must be provided for validation."
        assert len(args.model_paths) >= 2, "Error: model_path must contain path of at least two models."
        assert args.val_data, "Error: val_data (instrument) must be provided for validation."

        ### Prepare val dataloader
        _, val_dataloader = prepare_dataloaders(
            args.data_dir,
            [args.val_data],
            args.val_split,
            args.batch_size
        )

        print("Validation instrument -", args.val_data)
        print("-"*30)

        model_paths = parse_dict(args.model_paths)
        jointly_trained_model_path = parse_dict(args.jointly_trained_model_path)
        model_validation_single_pretrained(
            args.output_dir,
            model_paths, args.pretrained_model_path,
            val_dataloader, device
        )
        print("-"*30)
        model_validation_jointly_pretrained(
            args.output_dir,
            model_paths, jointly_trained_model_path, args.pretrained_model_path,
            val_dataloader, device
        )


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
