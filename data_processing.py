
import os
from PIL import Image
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    
    def __init__(self, data_list, use_grayscale=False, transform=None):
        self.transform = transform
        self.use_grayscale = use_grayscale

        class_names = sorted(list(set([item[1] for item in data_list])))
        self.classes = class_names
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        self.samples = [(path, self.class_to_idx[class_name]) for path, class_name in data_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        # Load image
        if self.use_grayscale:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, target


def prepare_dataloaders(
    data_dir: str,
    which_instrument: List[str],
    val_split: float,
    train_model: str,
    if_pretrained: bool,
    use_grayscale: bool,
    batch_size: int,
    num_workers: int,
    pin_mem: bool
):

    ### Initialize train and val size
    train_split = 1 - val_split

    if len(which_instrument) == 1:
        ### Single instrument processing
        instrument = which_instrument[0]
        if instrument == "HiRISE":
            folder_path = os.path.join(data_dir, "hirise-tiles")
        elif instrument == "CTX":
            folder_path = os.path.join(data_dir, "ctx-tiles")
        elif instrument == "THEMIS":
            folder_path = os.path.join(data_dir, "themis-tiles")
        else:
            raise ValueError(f"Unsupported instrument: {instrument}")

        all_filelist = os.listdir(folder_path)
        all_filelist = [os.path.join(folder_path, filename) for filename in all_filelist]
        train_size, val_size = int(train_split * len(all_filelist)), int(val_split * len(all_filelist))
        random.shuffle(all_filelist)
        train_list = all_filelist[:train_size]
        val_list = all_filelist[train_size: train_size + val_size]

    else:
        ### Multiple instruments processing
        train_list = []
        val_list = []

        for instrument in which_instrument:
            if instrument == "HiRISE":
                folder_path = os.path.join(data_dir, "hirise-tiles")
            elif instrument == "CTX":
                folder_path = os.path.join(data_dir, "ctx-tiles")
            elif instrument == "THEMIS":
                folder_path = os.path.join(data_dir, "themis-tiles")
            else:
                raise ValueError(f"Unsupported instrument: {instrument}")

            all_filelist = os.listdir(folder_path)
            all_filelist = [os.path.join(folder_path, filename) for filename in all_filelist]
            train_size, val_size = int(train_split * len(all_filelist)), int(val_split * len(all_filelist))
            random.shuffle(all_filelist)
            train_list = all_filelist[:train_size]
            val_list = all_filelist[train_size: train_size + val_size]
            train_list.extend(train_list)
            val_list.extend(val_list)

    ### Define transformations
    if if_pretrained:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_list = [(item, "class_0") for item in train_list]
    val_list = [(item, "class_0") for item in val_list]

    dataset_train = CustomImageDataset(data_list=train_list, use_grayscale=use_grayscale, transform=train_transforms)
    dataset_val = CustomImageDataset(data_list=val_list, use_grayscale=use_grayscale, transform=val_transforms)

    ### Create dataloaders
    if "vit" in train_model:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
        )

    else:
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size)

    return train_dataloader, val_dataloader