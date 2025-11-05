

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
import tifffile
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .base_dataset import BaseDataset


class CustomDataset(Dataset):

    def __init__(self, data_dir: str, df: pd.DataFrame, transform: transforms.Compose):
        self.data_dir = data_dir
        self.df = df
        self.transform = transform

        self.image_paths = []
        self.masks_paths = []
        self.filenames = []

        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, "data", row["split"], "images", row["file_id"])
            mask_path = os.path.join(data_dir, "data", row["split"], "masks", row["file_id"])
            self.image_paths.append(image_path)
            self.masks_paths.append(mask_path)
            self.filenames.append(row["file_id"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.masks_paths[idx]

        if "mmls" in self.data_dir:
            image = tifffile.imread(image_path)[:, :, [2, 1, 0]]
            mask = tifffile.imread(mask_path)
        else:
            # image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = Image.open(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        arr = np.array(image).astype(np.float32)
        arr_normalized = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        image = np.array(Image.fromarray(arr_normalized.astype(np.uint8)).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = np.expand_dims(mask, axis=0)

        mask = mask.astype(np.float32)

        return image, mask, self.filenames[idx]


class SegmentationDataset(BaseDataset):

    def __init__(self, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args):
        super().__init__(config, train_transform, val_transform)
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.partition = args.partition
        self.train_model = args.train_model
        self.batch_size = args.batch_size
        self.pin_mem = args.pin_mem
        self.use_positive_only_conequest = args.use_positive_only_conequest
        self.args = args

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _get_data_df(self, data_dir, split) -> pd.DataFrame:
        img_files = os.listdir(os.path.join(data_dir, split, "images"))
        data_df = pd.DataFrame({"file_id": img_files, "split": split})
        return data_df

    def _prepare_data(self):

        train_df = self._get_data_df(os.path.join(self.data_dir, "data"), "train")
        val_df = self._get_data_df(os.path.join(self.data_dir, "data"), "val")
        test_df = self._get_data_df(os.path.join(self.data_dir, "data"), "test")

        if self.partition:
            ratio = float(self.partition.split("x_")[0])
            n_samples = int(np.ceil(len(train_df) * ratio))
            train_df = train_df.sample(n=n_samples, random_state=self.args.seed).reset_index(drop=True)

        if ("conequest" in self.data_dir) and self.use_positive_only_conequest:
            metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
            metadata_df = metadata_df.loc[metadata_df["Number of Cones"]!=0]
            valid_names = metadata_df[["Patch Id"]].rename(columns={"Patch Id": "file_id"})
            train_df = train_df.merge(valid_names, on="file_id")
            val_df = val_df.merge(valid_names, on="file_id")
            test_df = test_df.merge(valid_names, on="file_id")

        return train_df, val_df, test_df

    def _create_dataset(self, df: pd.DataFrame, is_training: bool) -> Dataset:
        return CustomDataset(self.data_dir, df, self.train_transform if is_training else self.val_transform)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._create_dataset(self.train_df, is_training=True)
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=sampler_train,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
            pin_memory=self.pin_mem
        )
        return train_dataloader, self.train_df.shape[0]

    def get_val_dataloader(self) -> DataLoader:
        val_dataset = self._create_dataset(self.val_df, is_training=False)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset, sampler=sampler_val,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
            pin_memory=self.pin_mem
        )
        return val_dataloader

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = self._create_dataset(self.test_df, is_training=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=self.pin_mem
        )
        return test_dataloader
