
import argparse
import datetime
import json
import os
import random
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


# python main_finetune.py --data_dir /data/hkerner/mpurohi3/MarsBench/NewDatasets --dataset mb-mmls --balance_data loss_reweight --which_finetuning scratch_training --output_dir /scratch/bgajera2/Mirali/ --batch_size 32 --num_epochs 1


def get_args_parser():

    argparser = argparse.ArgumentParser(description="Fine-tuning script for all types of tasks")

    # Seed
    argparser.add_argument("--random_seed_per_run", default=False, required=False, action="store_true",
                            help="True value of this parameter assumes that you want to use a random seed for each run")

    # Dataset and paths
    argparser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    argparser.add_argument("--dataset", type=str, required=True, help="Dataset name",
                           choices=["mb-frost_cls", "mb-landmark_cls", "mb-domars16k", "mb-atmospheric_dust_cls_edr", "mb-atmospheric_dust_cls_rdr", "mb-change_cls_ctx", "mb-change_cls_hirise",
                                    "mb-conequest_seg", "mb-crater_binary_seg", "mb-mmls", "mb-boulder_seg", "mb-crater_multi_seg"])
    argparser.add_argument("--balance_data", default="default", required=False, type=str,
                           choices=["default", "loss_reweight", "under_sample", "over_sample"])
    argparser.add_argument("--few_shot", type=str, default=None, required=False,
                           help="Few shot dataset name only for classification tasks", choices=["1_shot", "2_shot", "5_shot", "10_shot", "15_shot", "20_shot"])
    argparser.add_argument("--partition", type=str, default=None, required=False,
                           help="Partition dataset name",
                           choices=["0.01x_partition", "0.02x_partition", "0.05x_partition", "0.10x_partition", "0.20x_partition", "0.25x_partition", "0.50x_partition"])

    # Finetuning parameters
    argparser.add_argument("--which_finetuning", type=str, default=None, required=True,
                           choices=["imagenet_pretrained", "scratch_training", "checkpoint"])
    argparser.add_argument("--finetuning_type", type=str, default="ft", required=False,
                           help="For finetuning, please provide the type of finetuning: lp for linear probing, ft for full finetuning",
                           choices=["lp", "ft"])
    argparser.add_argument("--encoder_checkpoint", type=str, default=None, required=False,
                           help="For finetuning, please provide path of the weights for encoder")
    argparser.add_argument("--normalize", type=str, default="HiRISE_CTX_THEMIS", required=False,
                           help="For finetuning, please provide the name of the pretrained model",
                           choices=["HiRISE", "CTX", "THEMIS", "HiRISE_CTX_THEMIS"])

    # Paths
    argparser.add_argument("--output_dir", type=str, default=None, required=False,
                           help="path where to save")
    argparser.add_argument("--metrics_dir", type=str, default="", required=False,
                            help="path where to save metrics")

    # Model and hyperparameters
    argparser.add_argument("--train_model", type=str, default="vit-b-16", required=False,
                            choices=["resnet34", "squeezenet1-1", "efficientnet-v2-m", "vit-t-16", "vit-s-16", "vit-b-16", "vit-l-16"])

    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--num_epochs", type=int, default=100)
    argparser.add_argument("--patience", type=int, default=5, required=False,
                            help="Number of epochs to wait for improvement before early stopping")

    argparser.add_argument("--drop_path", type=float, default=0.0, required=False)
    argparser.add_argument("--global_pool", default=True, required=False, action="store_true")
    argparser.add_argument("--learning_rate", type=float, default=1e-3)
    argparser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    argparser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')
    argparser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    argparser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    argparser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    argparser.add_argument('--max_norm', type=float, default=1.0, help='max norm for gradient clipping')
    argparser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    argparser.set_defaults(pin_mem=True)

    # Segmentation hyperparameters
    argparser.add_argument("--weight_dice", type=float, default=0.5, required=False,
                           help="Weight for dice loss")
    argparser.add_argument("--weight_ce", type=float, default=0.3, required=False,
                           help="Weight for cross entropy loss")
    argparser.add_argument("--weight_boundary", type=float, default=0.2, required=False,
                           help="Weight for boundary loss")
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
    if args.which_finetuning in ["imagenet_pretrained", "scratch_training"]:
        if (args.which_finetuning == "imagenet_pretrained") and ("vit" in args.train_model) and (args.encoder_checkpoint is None):
            raise ValueError("Path of ImageNet pretrained checkpoint must be provided for finetuning ViT models.")
        pretraining_configuration = "-"
        args.name_of_run = f"{args.which_finetuning}_{args.balance_data}"
    elif args.which_finetuning == "checkpoint":
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

    ### Load and update config
    with open("datasets_finetune/datasets_config.json", "r") as config_file:
        all_configs = json.load(config_file)
    if args.dataset not in all_configs:
        raise ValueError(f"Add dataset information of {args.dataset} in datasets_finetune/datasets_config.json")
    config = all_configs[args.dataset]
    config["balance"] = args.balance_data if args.balance_data is not None else "default"

    ### Create transforms, datasets and dataloaders
    train_transform = create_transforms(args.dataset, args.which_finetuning, args.normalize, is_training=True)
    val_transform = create_transforms(args.dataset, args.which_finetuning, args.normalize, is_training=False)

    dataset = DatasetFactory.create_dataset(config, train_transform, val_transform, args)
    train_dataloader, no_of_samples = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    test_dataloader = dataset.get_test_dataloader()

    ### Create model
    if "vit" in args.train_model:
        model = create_finetune_model_vit(args.train_model, args.which_finetuning, args.drop_path, args.global_pool, config, args.encoder_checkpoint, args.finetuning_type, device, args)
    else:
        model = create_finetune_model(args.train_model, args.which_finetuning, config, args.encoder_checkpoint, device)
    model = model.to(device)

    ### Create output and metrics directories
    if args.few_shot:
        current_output_folder = args.few_shot + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        metrics_dir = os.path.join(args.output_dir, "metrics", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        args.data_configuration = args.few_shot
    elif args.partition:
        current_output_folder = args.partition + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        metrics_dir = os.path.join(args.output_dir, "metrics", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        args.data_configuration = args.partition
    else:
        current_output_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_dir, "finetune", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        metrics_dir = os.path.join(args.output_dir, "metrics", args.train_model, args.dataset, args.name_of_run, current_output_folder)
        args.data_configuration = "full"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print(f"Output directory: {output_dir}\n")

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
        class_weights = compute_class_weights(train_dataloader, config["num_classes"])
        if args.balance_data == "loss_reweight":
            criterion = WeightedCombinedLoss(
                num_classes=config["num_classes"],
                class_weights=class_weights,
                weight_dice=args.weight_dice,
                weight_ce=args.weight_ce,
                weight_boundary=args.weight_boundary
            )
        else:
            criterion = CombinedLoss(num_classes=config["num_classes"])
    criterion = criterion.to(device)

    ### Create optimizer and scaler
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
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
                "Learning rate": args.learning_rate,
                "Min learning rate": args.min_lr,
                "Warmup epochs": args.warmup_epochs,
                "Weight decay": args.weight_decay,
                "Layer decay": args.layer_decay,
                "No of training samples": no_of_samples
            }
        )

    ### Train classification model
    if "classification" in config["task_type"]:
        model = training_model_classification(
            model, train_dataloader, val_dataloader,
            optimizer, device,
            output_dir, args.patience, scaler,
            args.name_of_run, criterion, args
        )
        eval_accuracy, eval_precision, eval_recall, eval_f1score, eval_acc1, eval_acc5 = evaluate_model_classification(model, test_dataloader, device, config, args.name_of_run, metrics_dir)

        ### Save classification results
        if args.few_shot:
            result_csv_path = os.path.join("results", f"{args.dataset}_few_shot_results.csv")
        elif args.partition:
            result_csv_path = os.path.join("results", f"{args.dataset}_partition_results.csv")
        else:
            result_csv_path = os.path.join("results", f"{args.dataset}_seed_results.csv")
        if os.path.exists(result_csv_path):
            result_df = pd.read_csv(result_csv_path)
        else:
            result_df = pd.DataFrame(columns=[
                "Downstream Task",  "Train Model", "Training type", "Pre-training configuration", "Finetuning type", "balance_data", "data_configuration", "no_of_training_samples",
                "Accuracy", "Precision", "Recall", "F1-Score", "Top-1 Accuracy", "Top-5 Accuracy", "batch_size", "num_epochs", "patience",
                "drop_path", "global_pool", "lr", "min_lr", "weight_decay", "layer_decay", "warmup_epochs", "max_norm", "accum_iter", "output_folder"
            ])
        current_result = [
            args.dataset, args.train_model, args.which_finetuning, pretraining_configuration, args.finetuning_type, args.balance_data, args.data_configuration, no_of_samples,
            round(eval_accuracy, 4), round(eval_precision, 4), round(eval_recall, 4), round(eval_f1score, 4),
            round(eval_acc1, 4), round(eval_acc5, 4), args.batch_size, args.num_epochs, args.patience,
            args.drop_path, args.global_pool, args.learning_rate, args.min_lr, args.weight_decay, args.layer_decay,
            args.warmup_epochs, args.max_norm, args.accum_iter, current_output_folder
        ]
        result_df.loc[len(result_df)] = current_result
        result_df.to_csv(result_csv_path, index=False)

    ### Train segmentation model
    if "segmentation" in config["task_type"]:
        model = training_model_segmentation(
            model, train_dataloader, val_dataloader,
            optimizer, device, config["num_classes"],
            output_dir, args.patience,
            scaler, args.name_of_run, criterion,
            class_weights, args
        )
        pixel_iou, pixel_accuracy, pixel_recall, pixel_precision, pixel_dice, object_precision, object_recall, object_f1, mean_ap, mean_ap_50, mean_ap_75, pixel_ap_mean = evaluate_model_segmentation(
            model=model, test_dataloader=test_dataloader,
            device=device, output_dir=output_dir,
            config=config, class_weights=class_weights, args=args
        )

        ### Save segmentation results
        if args.few_shot:
            result_csv_path = os.path.join("results", f"{args.dataset}_few_shot_results.csv")
        elif args.partition:
            result_csv_path = os.path.join("results", f"{args.dataset}_partition_results.csv")
        else:
            result_csv_path = os.path.join("results", f"{args.dataset}_results.csv")
        if os.path.exists(result_csv_path):
            result_df = pd.read_csv(result_csv_path)
        else:
            result_df = pd.DataFrame(columns=[
                "Downstream Task", "Train Model", "Training type", "Pre-training configuration", "Finetuning type", "balance_data", "data_configuration", "no_of_training_samples",
                "Pixel IoU", "Pixel Accuracy", "Pixel Precision", "Pixel Recall", "Pixel Dice", "Object Precision", "Object Recall", "Object F1-Score",
                "Instance mAP", "Instance mAP@0.5", "Instance mAP@0.75", "Pixel-based AP",
                "batch_size", "num_epochs", "patience", "drop_path", "global_pool", "lr", "min_lr", "weight_decay", "layer_decay",
                "warmup_epochs", "max_norm", "accum_iter", "weight_dice", "weight_ce", "weight_boundary", "use_positive_only_conequest", "output_folder"
            ])
        current_result = [
            args.dataset, args.train_model, args.which_finetuning, pretraining_configuration, args.finetuning_type, args.balance_data, args.data_configuration, no_of_samples,
            pixel_iou, pixel_accuracy, pixel_precision, pixel_recall, pixel_dice, object_precision, object_recall, object_f1,
            mean_ap, mean_ap_50, mean_ap_75, pixel_ap_mean,
            args.batch_size, args.num_epochs, args.patience, args.drop_path, args.global_pool, args.learning_rate, args.min_lr,
            args.weight_decay, args.layer_decay, args.warmup_epochs, args.max_norm, args.accum_iter, args.weight_dice, args.weight_ce, args.weight_boundary, args.use_positive_only_conequest,
            current_output_folder
        ]
        result_df.loc[len(result_df)] = current_result
        result_df.to_csv(result_csv_path, index=False)


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.random_seed_per_run:
        args.seed = random.randint(0, 2**32 - 1)
    else:
        args.seed = 42
    seed_everything(args.seed)

    main(args)
    torch.cuda.empty_cache()
