import argparse
import json
import os
import pandas as pd
import random
from tqdm import tqdm
# import wandb

import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from datasets_finetune.dataset_factory import DatasetFactory
from engine_finetune import *
from models_finetune import create_finetune_model
from utils import *


def get_args_parser():

    argparser = argparse.ArgumentParser(description="Fine-tuning script for all types of tasks")
    argparser.add_argument("--dataset", type=str, required=True,
                           help="Dataset name",
                           choices=["martian_frost", "hirise_landmark", "domars16", "atmospheric_dust_edr", "atmospheric_dust_rdr", "conequest"])
    argparser.add_argument("--balance_data", default=False, required=False, action="store_true")

    argparser.add_argument("--train_model", type=str, default="resnet34", required=False,
                            help="Available choices: resnet34, squeezenet1-1, efficientnet-v2-m, vit-b-16, vit-b-32, vit-l-16, vit-l-32")
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--num_epochs", type=int, default=10)
    argparser.add_argument("--learning_rate", type=float, default=0.0001)
    argparser.add_argument("--which_pretraining", type=str, default=None, required=True,
                           choices=["imagenet_pretrained", "scratch_training", "finetuning"])
    argparser.add_argument("--encoder_checkpoint", type=str, default=None, required=False,
                           help="For finetuning, please provide path of the weights for encoder")
    argparser.add_argument("--output_dir", type=str, default=None, required=True,
                           help="path where to save")

    argparser.add_argument("--wandb_enabled", default=False, required=False, action="store_true",
                            help="True value of this parameter assumes that you have wandb account")
    argparser.add_argument("--wandb_entity", type=str, default="mpurohi3", required=False,
                            help="Provide Wandb entity where plots will be available")
    argparser.add_argument("--wandb_project", type=str, default="LMM_finetuning", required=False,
                            help="Provide Wandb project name for plots")
    
    argparser.add_argument("--patience", type=int, default=5, required=False,
                            help="Number of epochs to wait for improvement before early stopping")
    argparser.add_argument("--metrics_dir", type=str, default="metrics", required=False,
                            help="path where to save metrics")
       

    return argparser



class CombinedLoss(nn.Module):

    def __init__(self, num_classes, weight_dice=0.5, weight_bce=0.5):
        """
        Initializes the combined loss function.
        Args:
            num_classes (int): Number of output classes.
            weight_dice (float): Weight for the Dice Loss component.
            weight_bce (float): Weight for the BCEWithLogitsLoss or CrossEntropyLoss component.
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

        # Define individual losses
        self.dice_loss_fn = smp.losses.DiceLoss(mode="binary" if num_classes == 1 else "multiclass", from_logits=True)
        self.bce_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Computes the combined loss.
        Args:
            logits (torch.Tensor): Raw model outputs (logits).
            targets (torch.Tensor): Ground truth targets.
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice_loss = self.dice_loss_fn(logits, targets)
        bce_loss = self.bce_loss_fn(logits, targets)  # CrossEntropyLoss directly uses logits

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + \
                        (self.weight_bce * bce_loss)

        return combined_loss


def create_transforms(train_model, is_training=True):

    if "vit" in train_model:
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    else:
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

def main(args):

    ### Set seed
    DEFAULT_SEED = random.randint(0, 2**32 - 1)
    seed_everything(DEFAULT_SEED)

    ### Check device type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Initializing output directory and unique name of current run
    if (args.which_pretraining == "imagenet_pretrained") or (args.which_pretraining == "scratch_training"):
        pretrained_model = args.which_pretraining
        pretraining_configuration = "-"
        name_of_run = "balance_data_" + str(args.balance_data) + "_" + args.which_pretraining
    else:
        assert args.encoder_checkpoint is not None, "Path of pretrained encoder checkpoint must be provided for finetuning."
        assert os.path.exists(args.encoder_checkpoint), f"Encoder checkpoint path does not exist: {args.encoder_checkpoint}"
        type_of_model = args.encoder_checkpoint.split("/")[-2]
        if (type_of_model == "combined_models") or ("customized_models" in type_of_model):
            pretraining_configuration = args.encoder_checkpoint.split("/")[-1].replace("_"+args.train_model, "")
        else:
            pretraining_configuration = args.encoder_checkpoint.split("/")[-2] + "_" + args.encoder_checkpoint.split("/")[-1].split(".")[0].split("_")[-1]
        pretrained_model = pretraining_configuration
        name_of_run = "balance_data_" + str(args.balance_data) + "_" + pretraining_configuration

    output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, name_of_run)
    os.makedirs(output_dir, exist_ok=True)

    ### Load and update config
    with open("datasets_finetune/datasets_config.json", "r") as config_file:
        all_configs = json.load(config_file)
    if args.dataset not in all_configs:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    config = all_configs[args.dataset]
    config["balance"] = args.balance_data

    ### Create transforms
    train_transform = create_transforms(args.train_model, is_training=True)
    val_transform = create_transforms(args.train_model, is_training=False)

    ### Create dataset and dataloaders using the factory
    dataset = DatasetFactory.create_dataset(args.dataset, config, train_transform, val_transform, args)
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))

    ### Create model
    model = create_finetune_model(args.train_model, args.which_pretraining, config, args.encoder_checkpoint, device)
    model = model.to(device)

    ### Create loss function based on the task type
    if "classification" in config["task_type"]:
        if config["num_classes"] == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    if "segmentation" in config["task_type"]:
        criterion = CombinedLoss(num_classes=config["num_classes"])

    ### Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # if args.wandb_enabled:
    #   wandb.init(
    #       entity=args.wandb_entity,
    #       project=args.wandb_project,
    #       name=args.dataset + "_" + name_of_run + "_" + args.train_model,
    #       config={
    #           "Dataset": args.dataset,
    #           "Model": args.train_model,
    #           "Training data samples": len(train_dataloader),
    #           "Validation data samples": len(val_dataloader),
    #           "Pre-trained Model": pretrained_model,
    #           "Epochs": args.num_epochs,
    #           "Batch size": args.batch_size,
    #           "Optimizer": optimizer,
    #           "Loss": criterion,
    #           "Model path": output_dir
    #       }
    #   )

    ### Train model
    for epoch in tqdm(range(args.num_epochs), desc=name_of_run):

        if "classification" in config["task_type"]:
            model, stop_early = training_model_classification(model, train_dataloader, val_dataloader,
                                   optimizer, device,
                                   epoch, config["num_classes"], output_dir,
                                   criterion, args)
            # if (epoch == 0) or ((epoch+1) % 5 == 0):
            # early stopping - patience -> 10
            # use best epoch for metrics
            # accuracy, precision, recall, f1score = evaluate_model_classification(model=model, test_dataloader=test_dataloader,
            #                                                                         device=device, output_dir=None, config=config)
            if stop_early:
                print("Early stopping triggered.")
                break

        if "segmentation" in config["task_type"]:
            model = training_model_segmentation(model, train_dataloader, val_dataloader,
                                                optimizer, device,
                                                epoch, config["num_classes"], output_dir,
                                                criterion, args)

            if (epoch == 0) or ((epoch+1) % 5 == 0):
                pixel_iou, pixel_accuracy, pixel_precision, pixel_recall, pixel_dice = evaluate_model_segmentation(model=model, test_dataloader=test_dataloader,
                                                                                                                   device=device, output_dir=None, config=config)

    # Final evaluation on test set using best model
    if "classification" in config["task_type"]:
        # Load the best model for final evaluation
        best_model_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("\nFinal evaluation on test set using best model...")
            accuracy, precision, recall, f1score = evaluate_model_classification(
                model=model, 
                test_dataloader=test_dataloader,
                device=device, 
                output_dir=output_dir, 
                config=config,
                model_name = args.train_model,
                dataset_name = args.dataset,
                checkpoint_name = args.encoder_checkpoint.split("/")[-1],
                metrics_dir = args.metrics_dir
            )
        else:
            print("Warning: No best model found for final evaluation")
            accuracy, precision, recall, f1score = 0, 0, 0, 0

    ### Save results in CSV file
    if "classification" in config["task_type"]:
        result_csv_path = os.path.join("results", f"results_classification_ci.csv")
        if os.path.exists(result_csv_path):
            result_df = pd.read_csv(result_csv_path)
        else:
            result_df = pd.DataFrame(columns=[
                "Downstream Task",
                "Training type",
                "Train Model",
                "Pre-training configuration",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ])
        current_result = [
            args.dataset,
            args.which_pretraining,
            args.train_model,
            pretraining_configuration,
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1score, 4)
        ]
        result_df.loc[len(result_df)] = current_result
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        result_df.to_csv(result_csv_path, index=False)

    if "segmentation" in config["task_type"]:
        result_csv_path = os.path.join("results", f"results_segmentation.csv")
        if os.path.exists(result_csv_path):
            result_df = pd.read_csv(result_csv_path)
        else:
            result_df = pd.DataFrame(columns=[
                "Downstream Task",
                "Training type",
                "Train Model",
                "Pre-training configuration",
                "Pixel IoU",
                "Pixel Accuracy",
                "Pixel Precision",
                "Pixel Recall",
                "Pixel Dice"
            ])
        current_result = [
            args.dataset,
            args.which_pretraining,
            args.train_model,
            pretraining_configuration,
            pixel_iou,
            pixel_accuracy,
            pixel_precision,
            pixel_recall,
            pixel_dice
        ]
        result_df.loc[len(result_df)] = current_result
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        result_df.to_csv(result_csv_path, index=False)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
