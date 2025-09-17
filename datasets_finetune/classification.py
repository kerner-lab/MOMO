
import cv2
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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
        image_path = os.path.join(self.data_dir, "data", row["split"], row["feature_name"], row["file_id"])
        label = row["label"]
        filename = image_path.split("/")[-1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label, filename


class ClassificationDataset(BaseDataset):

    def __init__(self, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args):
        super().__init__(config, train_transform, val_transform)
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.few_shot = args.few_shot
        self.partition = args.partition
        self.balance = args.balance_data
        self.train_model = args.train_model
        self.batch_size = args.batch_size
        self.pin_mem = args.pin_mem

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _prepare_data(self):

        if self.few_shot:
            data_df = pd.read_csv(os.path.join(self.data_dir, "few_shot", f"{self.few_shot}.csv"))
        elif self.partition:
            data_df = pd.read_csv(os.path.join(self.data_dir, "partitions", f"{self.partition}.csv"))
        else:
            data_df = pd.read_csv(os.path.join(self.data_dir, "annotation.csv"))

        train_df = data_df[data_df["split"]=="train"]
        val_df = data_df[data_df["split"]=="val"]
        test_df = data_df[data_df["split"]=="test"]

        if self.balance == "under_sample":
            min_samples = train_df['label'].value_counts().min()
            train_df = train_df.groupby('label').apply(lambda x: x.sample(n=min_samples, random_state=42)).reset_index(drop=True)
        elif self.balance == "over_sample":
            ros = RandomOverSampler(random_state=42)
            # Only use label column for y
            X_resampled, y_resampled = ros.fit_resample(train_df.drop(columns=['label']), train_df['label'])
            # Combine the resampled features with the resampled labels
            train_df = pd.concat([X_resampled, pd.Series(y_resampled, name='label')], axis=1)
            # Ensure labels are still in the correct range
            train_df['label'] = train_df['label'].astype(int)

        return train_df, val_df, test_df

    def _create_dataset(self, data_dir: str, df: pd.DataFrame, is_training: bool) -> Dataset:
        return CustomDataset(self.data_dir, df, self.train_transform if is_training else self.val_transform)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._create_dataset(self.data_dir, self.train_df, is_training=True)
        if "vit" in self.train_model:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, sampler=sampler_train,
                batch_size=self.batch_size,
                pin_memory=self.pin_mem
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
                pin_memory=self.pin_mem
            )
        else:
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return val_dataloader

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = self._create_dataset(self.data_dir, self.test_df, is_training=False)
        return DataLoader(test_dataset, batch_size=1, shuffle=False)

    def get_class_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df['label']),
            y=self.train_df['label']
        )
        class_weight_dict = dict(zip(np.unique(self.train_df['label']), class_weights))
        print(class_weight_dict)
        return class_weights
