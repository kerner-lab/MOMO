
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


def create_transforms(dataset, which_finetuning, normalize, is_training=True):

    with open("utils/statistics.json", "r") as f:
        INSTRUMENT_STATS = json.load(f)

    with open("datasets_finetune/datasets_config.json", "r") as f:
        DATASETS_CONFIG = json.load(f)

    if which_finetuning == "scratch_training":
        mean = DATASETS_CONFIG[dataset]["mean"]
        std = DATASETS_CONFIG[dataset]["std"]
    elif which_finetuning == "imagenet_pretrained":
        mean = INSTRUMENT_STATS["ImageNet"]["mean"]
        std = INSTRUMENT_STATS["ImageNet"]["std"]
    else:
        mean = INSTRUMENT_STATS[normalize]["mean"]
        std = INSTRUMENT_STATS[normalize]["std"]

    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.33),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
