
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
from typing import List
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    
    def __init__(self, data_dir, data_df, if_pretrained: bool, transform=None):
        self.df = data_df.reset_index(drop=True)
        self.transform = transform
        self.if_pretrained = if_pretrained

        self.image_paths = []
        self.gmom_units = []

        for _, row in data_df.iterrows():
            image_path = os.path.join(data_dir, row["which_instrument"], "processed_data", row["GMoM_Unit_acronym"], row["Filename"])
            self.image_paths.append(image_path)
            self.gmom_units.append(row["GMoM_Unit_acronym"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, gmom_unit = self.image_paths[index], self.gmom_units[index]

        # Load image
        arr = np.array(Image.open(image_path)).astype(np.float32)
        arr_normalized = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        if self.if_pretrained:
            image = Image.fromarray(arr_normalized.astype(np.uint8)).convert("RGB")
        else:
            image = Image.fromarray(arr_normalized.astype(np.uint8)).convert("L")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, gmom_unit


def prepare_dataloaders(
    data_dir: str,
    data_df: str,
    if_pretrained: bool,
    which_instrument: List[str],
    batch_size: int,
    num_workers: int,
    pin_mem: bool
):

    ### Define transforms
    instrument_key = ", ".join(which_instrument)

    with open("utils/statistics.json", "r") as f:
        INSTRUMENT_STATS = json.load(f)

    if if_pretrained:
        mean = INSTRUMENT_STATS["ImageNet"]["mean"]
        std = INSTRUMENT_STATS["ImageNet"]["std"]
    else:
        mean = INSTRUMENT_STATS[instrument_key]["mean"]
        std = INSTRUMENT_STATS[instrument_key]["std"]

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std),
    ])

    ### Read data
    data_df = pd.read_csv(data_df)
    train_df = data_df[data_df["Split"]=="train"]
    val_df = data_df[data_df["Split"]=="val"]

    ### Create datasets
    dataset_train = CustomImageDataset(data_dir=data_dir, data_df=train_df, if_pretrained=if_pretrained, transform=train_transforms)
    dataset_val = CustomImageDataset(data_dir=data_dir, data_df=val_df, if_pretrained=if_pretrained, transform=val_transforms)

    ### Create dataloaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)

    train_dataloader = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    return train_dataloader, val_dataloader
