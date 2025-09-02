
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .base_dataset import BaseDataset

class CustomDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        label = row["label"]
        filename = image_path.split("/")[-1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # label = torch.tensor(label, dtype=torch.long)

        return image, label, filename


class MartianFrostDataset(BaseDataset):

    def __init__(self, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args):
        super().__init__(config, train_transform, val_transform)
        self.data_dir = config["data_dir"]
        self.num_classes = config["num_classes"]
        self.task_type = config["task_type"]
        self.balance = config["balance"]
        self.batch_size = args.batch_size

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _prepare_data(self):
        train_list, val_list, test_list = [], [], []

        for file_name, data_list in [("train_source_images.txt", train_list),
                                     ("val_source_images.txt", val_list),
                                     ("test_source_images.txt", test_list)]:
            with open(os.path.join(self.data_dir, file_name)) as f:
                data_list.extend([line.strip() for line in f.readlines()])

        data_dir = os.path.join(self.data_dir, "data")
        filepath_list, feature_list, set_list = [], [], []

        for each_folder in os.listdir(data_dir):
            if each_folder == ".DS_Store":
                continue
            parent_folder = "_".join(each_folder.split("_")[0:3])
            if parent_folder in train_list:
                current_set = "train"
            elif parent_folder in val_list:
                current_set = "val"
            elif parent_folder in test_list:
                current_set = "test"
            
            class_name = os.listdir(os.path.join(data_dir, each_folder, "tiles"))[0]
            for each_file in os.listdir(os.path.join(data_dir, each_folder, "tiles", class_name)):
                filepath_list.append(os.path.join(data_dir, each_folder, "tiles", class_name, each_file))
                feature_list.append(class_name)
                set_list.append(current_set)

        data_dict = {"filepath": filepath_list, "feature": feature_list, "set": set_list}
        data_df = pd.DataFrame(data_dict)
        data_df = data_df[data_df["filepath"] != ".DS_Store"]

        le = LabelEncoder()
        le.fit(data_df["feature"])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        # print(le_name_mapping)
        data_df["label"] = le.transform(data_df["feature"])

        train_df = data_df[data_df["set"] == "train"]
        if self.balance:
            # Balance the train_df based on label column
            min_samples = train_df['label'].value_counts().min()
            train_df = train_df.groupby('label').apply(lambda x: x.sample(n=min_samples, random_state=42)).reset_index(drop=True)
        val_df = data_df[data_df["set"] == "val"]
        test_df = data_df[data_df["set"] == "test"]

        return train_df, val_df, test_df

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._create_dataset(self.train_df, is_training=True)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def get_val_dataloader(self) -> DataLoader:
        val_dataset = self._create_dataset(self.val_df, is_training=False)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def get_test_dataloader(self) -> DataLoader:
        test_dataset = self._create_dataset(self.test_df, is_training=False)
        return DataLoader(test_dataset, batch_size=1, shuffle=False)

    def _create_dataset(self, df: pd.DataFrame, is_training: bool) -> Dataset:
        return CustomDataset(df, self.train_transform if is_training else self.val_transform)
