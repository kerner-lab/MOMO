
import argparse
import os
import random
import time
import wandb
import timm
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_processing import prepare_dataloaders
from engine_merging import create_combined_encoder
from engine_pretrain import model_training, model_training_vit
from models_pretrain import create_model

import utils.lr_sched as lr_sched
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.seed import seed_everything


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
    argparser.add_argument("--if_merging", default=False, required=False, action="store_true")

    # Pre-training and output directory
    argparser.add_argument("--data_dir", type=str, default="data", required=False)
    argparser.add_argument("--data_df", type=str, default="", required=False)
    argparser.add_argument("--which_instrument", "--list", type=str, default="", required=False)
    argparser.add_argument("--val_split", type=float, default=0.1, required=False)
    argparser.add_argument("--output_dir", type=str, default="models", required=False)

    # Model parameters
    argparser.add_argument("--train_model", type=str, default="resnet34", required=False,
                            help="Available choices: resnet34, squeezenet1-1, efficientnet-v2-m, vit-t-16, vit-s-16, vit-b-16, vit-l-16")
    argparser.add_argument("--if_pretrained", default=False, required=False, action="store_true")
    argparser.add_argument("--backbone_weight", type=str, default="imagenet", required=False)
    argparser.add_argument("--vit_pretrained_checkpoint_path", type=str, default="", required=False)

    # Hyperparameters
    argparser.add_argument("--num_epochs", type=int, default=500, required=False)
    argparser.add_argument("--batch_size", type=int, default=256, required=False)
    argparser.add_argument("--evaluation_interval", type=int, default=100000, required=False)
    argparser.add_argument("--num_workers", type=int, default=4, required=False)
    argparser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    argparser.add_argument('--learning_rate', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    argparser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    argparser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    argparser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    argparser.add_argument('--accum_iter', type=int, default=1, help='Accumulate gradient iterations (for training, not validatio)')
    argparser.add_argument('--mask_ratio', default=0.75, type=float, help='Mask`ing ratio (percentage of removed patches) only for ViT models.')
    argparser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    argparser.set_defaults(pin_mem=True)

    # MAE parameters
    argparser.add_argument('--combined_loss', default=False, required=False, action='store_true', help='use combined loss')
    argparser.add_argument('--mse_weight', type=float, default=1.0, help='weight for mse loss')
    argparser.add_argument('--ssim_weight', type=float, default=0.15, help='weight for ssim loss')
    argparser.add_argument('--lpips_weight', type=float, default=0.5, help='weight for lpips loss')
    argparser.add_argument('--grad_weight', type=float, default=0.2, help='weight for gradient loss')

    # Wandb parameters
    argparser.add_argument("--wandb_enabled", default=False, required=False, action="store_true",
                        help="True value of this parameter assumes that you have wandb account")
    argparser.add_argument("--wandb_entity", type=str, default="mpurohi3", required=False,
                            help="Provide Wandb entity where plots will be available")
    argparser.add_argument("--wandb_project", type=str, default="LMM", required=False,
                            help="Provide Wandb project name for plots")

    # Merging args
    argparser.add_argument("--model_combinations", type=str, default="", required=False,
                        help="Dictionary of pre-trained model paths for validation")
    argparser.add_argument("--suffix", type=str, default="", required=False)
    argparser.add_argument("--which_merging_technique", type=str, default="task_vectors", required=False,
                        help="Which merging technique to merge models. Available choices: task_vectors, magmax")
    argparser.add_argument("--pretrained_model_path", type=str, default="", required=False,
                           help="Path of pre-trained model")
    argparser.add_argument("--scaling_coef", type=float, default=0.5, required=False,
                        help="Scaling coefficient for merging techniques")

    return argparser


def main(args):

    ### Check if either if_training or if_merging is True
    if not (args.if_training or args.if_merging):
        raise AssertionError("Either --if_training or --if_merging must be set to True!!")

    ### Check device type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Model training
    if args.if_training:

        ### Initialize data and model directory, unique name of current run
        args.which_instrument = [item for item in args.which_instrument.split(', ')]
        name_of_run = "_".join([each_instrument.lower() for each_instrument in args.which_instrument]) + "_" + str(args.if_pretrained)
        args.name_of_run = name_of_run
        output_dir = os.path.join(args.output_dir, "pretraining", args.train_model, name_of_run)
        print("Created output directory: ", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        ### Prepare dataloaders
        train_dataloader, val_dataloader = prepare_dataloaders(
            args.data_dir,
            args.data_df,
            args.if_pretrained,
            args.batch_size,
            args.num_workers,
            args.pin_mem
        )

        ### Initialize model, loss, optimizer and wandb
        model = create_model(
            train_model=args.train_model,
            model_unit="autoencoder",
            device=device,
            if_pretrained=args.if_pretrained,
            args=args
        )

        ### Initialize optimizer and loss scaler
        if "vit" in args.train_model:
            eff_batch_size = args.batch_size * args.accum_iter

            if args.learning_rate is None:
                args.learning_rate = args.blr * eff_batch_size / 256

            param_groups = lr_sched.add_weight_decay(model, args.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
            loss_scaler = NativeScaler()

            misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
            criterion = loss_scaler

        else:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        ### Initialize wandb
        if args.wandb_enabled:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.train_model + "_" + name_of_run,
                config={
                        "Training data samples": len(train_dataloader),
                        "Validation data samples": len(val_dataloader),
                        "Model": args.train_model,
                        "Epochs": args.num_epochs,
                        "Batch size": args.batch_size,
                        "Optimizer": optimizer,
                        "Loss": criterion,
                        "Model path": output_dir
                    }
            )

        ### Training model
        if "vit" in args.train_model:
            model_training_vit(
                model,
                train_dataloader, val_dataloader,
                args.num_epochs, device, output_dir,
                loss_scaler, optimizer, wandb, args
            )
        else:
            model_training(
                model,
                train_dataloader, val_dataloader,
                args.num_epochs, device, output_dir,
                criterion, optimizer, wandb, args
            )

    ### Model Merging
    if args.if_merging:
        model_combinations = parse_dict(args.model_combinations)
        _ = create_combined_encoder(
            model_combinations, args.train_model, args.pretrained_model_path,
            args.which_merging_technique, args.output_dir,
            args.suffix, device, args.scaling_coef, args
        )


if __name__ == '__main__':
    seed_everything(42)
    args = get_args_parser()
    args = args.parse_args()
    main(args)
