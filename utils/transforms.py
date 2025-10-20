
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


def create_transforms(train_model, task_type, which_pretraining, is_training=True):

    with open("utils/statistics.json", "r") as f:
        INSTRUMENT_STATS = json.load(f)

    if which_pretraining == "ImageNet":
        mean = INSTRUMENT_STATS["ImageNet"]["mean"]
        std = INSTRUMENT_STATS["ImageNet"]["std"]
    else:
        mean = INSTRUMENT_STATS[which_pretraining]["mean"]
        std = INSTRUMENT_STATS[which_pretraining]["std"]

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
