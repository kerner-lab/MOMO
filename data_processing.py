
import os
from PIL import Image
import random
from typing import List

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    
    def __init__(self, file_list, transform=None):
        self.image_files = [f for f in file_list if f.endswith((".tif", ".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def prepare_dataloaders(
    data_dir: str,
    which_instrument: List[str],
    val_split: float,
    batch_size: int
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

    # print(f"Total training samples: {len(train_list)}")
    # print(f"Total validation samples: {len(val_list)}")

    ### Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    val_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    ### Create datasets
    train_dataset = ImageDataset(file_list=train_list, transform=train_transforms)
    val_dataset = ImageDataset(file_list=val_list, transform=val_transforms)

    ### Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
