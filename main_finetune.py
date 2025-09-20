
import argparse
import datetime
import json
import os
import wandb
import warnings

import torch
from torch import nn

from datasets_finetune.dataset_factory import DatasetFactory
from engine_finetune import *
from models_finetune import create_finetune_model, create_finetune_model_vit

from utils.losses import WeightedCombinedLoss, compute_class_weights, CombinedLoss
import utils.lr_decay as lrd
from utils.seed import seed_everything
from utils.transforms import create_transforms

# Suppress warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_warn_always(False)


def get_args_parser():

    argparser = argparse.ArgumentParser(description="Fine-tuning script for all types of tasks")

    # Dataset and paths
    argparser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    argparser.add_argument("--dataset", type=str, required=True, help="Dataset name",
                           choices=["mb-frost_cls", "mb-landmark_cls", "mb-domars16k", "mb-atmospheric_dust_cls_edr", "mb-atmospheric_dust_cls_rdr", "mb-change_cls_ctx", "mb-change_cls_hirise"
                                    "mb-conequest_seg", "mb-crater_binary_seg", "mb-mmls", "mb-boulder_seg"])
    argparser.add_argument("--balance_data", default="default", required=False, type=str,
                           choices=["loss_reweight", "under_sample", "over_sample"])
    argparser.add_argument("--few_shot", type=str, default=None, required=False,
                           help="Few shot dataset name", choices=["1_shot", "2_shot", "5_shot", "10_shot", "15_shot", "20_shot"])
    argparser.add_argument("--partition", type=str, default=None, required=False,
                           help="Partition dataset name",
                           choices=["0.01x_partition", "0.02x_partition", "0.05x_partition", "0.10x_partition", "0.20x_partition", "0.25x_partition", "0.50x_partition"])

    # Finetuning parameters
    argparser.add_argument("--which_pretraining", type=str, default=None, required=True,
                           choices=["imagenet_pretrained", "scratch_training", "finetuning", "evaluation"])
    argparser.add_argument("--encoder_checkpoint", type=str, default=None, required=False,
                           help="For finetuning, please provide path of the weights for encoder")

    # Paths
    argparser.add_argument("--output_dir", type=str, default=None, required=False,
                           help="path where to save")
    argparser.add_argument("--metrics_dir", type=str, default="metrics", required=False,
                            help="path where to save metrics")

    # Model and hyperparameters
    argparser.add_argument("--seed", type=int, default=42, required=False)
    argparser.add_argument("--train_model", type=str, default="resnet34", required=False,
                            choices=["resnet34", "squeezenet1-1", "efficientnet-v2-m", "vit-b-16", "vit-b-32", "vit-l-16", "vit-l-32"])

    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--num_epochs", type=int, default=100)
    argparser.add_argument("--patience", type=int, default=5, required=False,
                            help="Number of epochs to wait for improvement before early stopping")

    argparser.add_argument("--drop_path", type=float, default=0.0, required=False)
    argparser.add_argument("--global_pool", default=True, required=False, action="store_true")
    argparser.add_argument("--lr", type=float, default=5e-3)
    argparser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    argparser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    argparser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    argparser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    argparser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    argparser.add_argument('--max_norm', type=float, default=1.0, help='max norm for gradient clipping')
    argparser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    argparser.set_defaults(pin_mem=True)

    # Data parameters
    argparser.add_argument("--use_positive_only_conequest", default=False, required=False, action="store_true",
                            help="Use negative samples only in ConeQuest")

    # wandb
    argparser.add_argument("--wandb_enabled", default=False, required=False, action="store_true",
                            help="True value of this parameter assumes that you have wandb account")
    argparser.add_argument("--wandb_entity", type=str, default="mpurohi3", required=False,
                            help="Provide Wandb entity where plots will be available")
    argparser.add_argument("--wandb_project", type=str, default="LMM_finetuning", required=False,
                            help="Provide Wandb project name for plots")

    return argparser



def main(args):

    ### Check device type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Initializing output directory and unique name of current run
    if args.which_pretraining in ["imagenet_pretrained", "scratch_training"]:
        if (args.which_pretraining == "imagenet_pretrained") and ("vit" in args.train_model) and (args.encoder_checkpoint is None):
            raise ValueError("Path of ImageNet pretrained checkpoint must be provided for finetuning ViT models.")
        pretraining_configuration = "-"
        args.name_of_run = f"{args.which_pretraining}_{args.balance_data}"
    elif args.which_pretraining == "finetuning":
        assert args.encoder_checkpoint is not None, "Path of pretrained encoder checkpoint must be provided for finetuning."
        path_parts = args.encoder_checkpoint.split("/")
        checkpoint_name, type_of_model = path_parts[-1], path_parts[-2]
        if "model_merging" in type_of_model:
            pretraining_configuration = type_of_model.replace("model_merging_", "") + "_" + checkpoint_name.replace(".pth", "")
        else:
            pretraining_configuration = checkpoint_name.replace(".pth", "")
        args.name_of_run = f"{pretraining_configuration}_{args.balance_data}"
    else:
        args.name_of_run = args.encoder_checkpoint.split("/")[2]
        output_dir = args.output_dir

    ### Load and update config
    with open("datasets_finetune/datasets_config.json", "r") as config_file:
        all_configs = json.load(config_file)
    if args.dataset not in all_configs:
        raise ValueError(f"Add dataset information of {args.dataset} in datasets_finetune/datasets_config.json")
    config = all_configs[args.dataset]
    config["balance"] = args.balance_data if args.balance_data is not None else "default"

    ### Create transforms, datasets and dataloaders
    train_transform = create_transforms(args.train_model, config["task_type"], is_training=True)
    val_transform = create_transforms(args.train_model, config["task_type"], is_training=False)

    dataset = DatasetFactory.create_dataset(args.dataset, config, train_transform, val_transform, args)
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()

    ### Create model
    if "vit" in args.train_model:
        model = create_finetune_model_vit(args.train_model, args.which_pretraining, args.drop_path, args.global_pool, config, args.encoder_checkpoint, device, args)
    else:
        model = create_finetune_model(args.train_model, args.which_pretraining, config, args.encoder_checkpoint, device)
    model = model.to(device)

    if args.which_pretraining != "evaluation":
        if args.few_shot:
            output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, args.few_shot + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        elif args.partition:
            output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, args.partition + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(output_dir, exist_ok=True)

        ### Save arguments as JSON
        args_dict = vars(args)
        args_json_path = os.path.join(output_dir, "args.json")
        with open(args_json_path, "w") as f:
            json.dump(args_dict, f, indent=4)

        ### Create loss function based on the task type
        if "classification" in config["task_type"]:
            if args.balance_data == "loss_reweight":
                class_weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        if "segmentation" in config["task_type"]:
            if args.balance_data == "loss_reweight":
                class_weights = compute_class_weights(train_dataloader, config["num_classes"])
                criterion = WeightedCombinedLoss(
                    num_classes=config["num_classes"],
                    class_weights=class_weights,
                    weight_dice=0.3,
                    weight_bce=0.7
                )
            else:
                criterion = CombinedLoss(num_classes=config["num_classes"])
        criterion = criterion.to(device)

        ### Create optimizer and scaler
        param_groups = lrd.param_groups_lrd(model, args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        scaler = torch.cuda.amp.GradScaler()

        ### Initialize wandb
        if args.wandb_enabled:
            if args.few_shot:
                wandb_name = args.dataset + "_" + args.name_of_run + "_" + args.train_model + "_" + args.few_shot
            elif args.partition:
                wandb_name = args.dataset + "_" + args.name_of_run + "_" + args.train_model + "_" + args.partition
            else:
                wandb_name = args.dataset + "_" + args.name_of_run + "_" + args.train_model
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=wandb_name,
                config={
                    "Dataset": args.dataset,
                    "Balance data": args.balance_data,
                    "Model": args.train_model,
                    "Training data samples": len(train_dataloader),
                    "Validation data samples": len(val_dataloader),
                    "Pre-trained Model": pretraining_configuration,
                    "Epochs": args.num_epochs,
                    "Patience": args.patience,
                    "Batch size": args.batch_size,
                    "Optimizer": optimizer,
                    "Loss": criterion,
                    "Output dir": output_dir,
                    "Learning rate": args.lr,
                    "Min learning rate": args.min_lr,
                    "Warmup epochs": args.warmup_epochs,
                    "Weight decay": args.weight_decay,
                    "Layer decay": args.layer_decay,
                    "No of training samples": len(train_dataloader)*args.batch_size
                }
            )

        ### Train model
        if "classification" in config["task_type"]:
            if args.few_shot:
                result_csv_path = os.path.join("results", f"{args.dataset}_few_shot_results.csv")
            elif args.partition:
                result_csv_path = os.path.join("results", f"{args.dataset}_partition_results.csv")
            else:
                result_csv_path = os.path.join("results", f"{args.dataset}_results.csv")
            model = training_model_classification(
                model, train_dataloader, val_dataloader,
                optimizer, device,
                output_dir, args.patience, scaler,
                args.name_of_run, criterion, args
            )
            eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5 = evaluate_model_classification(
                model, test_dataloader,
                device, result_csv_path,
                args.balance_data, config,
                pretraining_configuration, output_dir, args.name_of_run,
                len(train_dataloader)*args.batch_size, args
            )

        if "segmentation" in config["task_type"]:
            if args.few_shot:
                result_csv_path = os.path.join("results", f"{args.dataset}_few_shot_results.csv")
            elif args.partition:
                result_csv_path = os.path.join("results", f"{args.dataset}_partition_results.csv")
            else:
                result_csv_path = os.path.join("results", f"{args.dataset}_results.csv")
            model = training_model_segmentation(
                model, train_dataloader, val_dataloader,
                optimizer, device, config["num_classes"],
                output_dir, args.patience,
                scaler, args.name_of_run, criterion, args
            )
            pixel_iou, pixel_accuracy, pixel_recall, pixel_precision, pixel_dice, object_precision, object_recall = evaluate_model_segmentation(
                model=model, test_dataloader=test_dataloader,
                device=device, output_dir=output_dir,
                result_csv_path=result_csv_path, config=config,
                pretraining_configuration=pretraining_configuration, args=args
            )

    ### Evaluate model
    else:
        if args.encoder_checkpoint is None:
            raise ValueError("Output directory must be provided for evaluation.")

        if "classification" in config["task_type"]:
            result_csv_path = os.path.join("results", f"{args.dataset}_results_classification_evaluate.csv")
            eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5 = evaluate_model_classification(
                model, test_dataloader,
                device, result_csv_path,
                "", config, args.name_of_run,
                output_dir, args.name_of_run,
                len(train_dataloader)*args.batch_size, args
            )

        if "segmentation" in config["task_type"]:
            result_csv_path = os.path.join("results", f"{args.dataset}_results_segmentation_evaluate.csv")
            pixel_iou, pixel_accuracy, pixel_recall, pixel_precision, pixel_dice, object_precision, object_recall = evaluate_model_segmentation(
                model=model, output_dir=args.output_dir,
                test_dataloader=test_dataloader, device=device,
                result_csv_path=result_csv_path, config=config,
                pretraining_configuration=pretraining_configuration, args=args
            )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    seed_everything(args.seed)
    main(args)
    torch.cuda.empty_cache()
