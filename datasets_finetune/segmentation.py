

import cv2
import numpy as np
import os
import pandas as pd
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_dir, "data", row["split"], "images", row["file_id"])
        label_path = os.path.join(self.data_dir, "data", row["split"], "masks", row["file_id"])
        filename = row["file_id"]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=0)

        if self.transform:
            image = self.transform(image)

        return image, label, filename


class SegmentationDataset(BaseDataset):

    def __init__(self, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args):
        super().__init__(config, train_transform, val_transform)
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.partition = args.partition
        self.train_model = args.train_model
        self.batch_size = args.batch_size
        self.use_positive_only_conequest = args.use_positive_only_conequest

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _get_data_df(self, data_dir, split) -> pd.DataFrame:
        img_files = os.listdir(os.path.join(data_dir, split, "images"))
        data_df = pd.DataFrame({"file_id": img_files, "split": split})
        return data_df

    def _prepare_data(self):

        if self.partition:
            data_df = pd.read_csv(os.path.join(self.data_dir, "partitions", f"{self.partition}.csv"))
            train_df = data_df[data_df["split"]=="train"]
            val_df = data_df[data_df["split"]=="val"]
            test_df = data_df[data_df["split"]=="test"]
        else:
            train_df = self._get_data_df(os.path.join(self.data_dir, "data"), "train")
            val_df = self._get_data_df(os.path.join(self.data_dir, "data"), "val")
            test_df = self._get_data_df(os.path.join(self.data_dir, "data"), "test")

        if ("conequest" in self.data_dir) and self.use_positive_only_conequest:
            data_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
            data_df = data_df.loc[data_df["Number of Cones"]!=0]
            valid_names = data_df["Patch Id"]
            train_df = train_df.merge(valid_names, on="Patch Id")
            val_df = val_df.merge(valid_names, on="Patch Id")
            test_df = test_df.merge(valid_names, on="Patch Id")

        return train_df, val_df, test_df

    def _create_dataset(self, data_dir: str, df: pd.DataFrame, is_training: bool) -> Dataset:
        return CustomDataset(data_dir, df, self.train_transform if is_training else self.val_transform)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._create_dataset(self.data_dir, self.train_df, is_training=True)
        if "vit" in self.train_model:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, sampler=sampler_train,
                batch_size=self.batch_size,
                pin_memory=self.pin_mem,
                drop_last=True,
            )
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return train_dataloader

    def get_val_dataloader(self) -> DataLoader:
        val_dataset = self._create_dataset(self.data_dir, self.val_df, is_training=False)
        if "vit" in self.train_model:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, sampler=sampler_val,
                batch_size=self.batch_size,
                pin_memory=self.pin_mem,
                drop_last=True,
            )
        else:
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return val_dataloader

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = self._create_dataset(self.data_dir, self.test_df, is_training=False)
        return DataLoader(test_dataset, batch_size=1, shuffle=False)

