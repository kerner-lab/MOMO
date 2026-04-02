
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


def create_transforms(dataset, which_finetuning, is_training=True, seed=None):

    if which_finetuning == "scratch_training":
        with open("datasets_finetune/datasets_config.json", "r") as f:
            DATASETS_CONFIG = json.load(f)
        mean = DATASETS_CONFIG[dataset]["mean"]
        std = DATASETS_CONFIG[dataset]["std"]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.33),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], seed=seed)
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
