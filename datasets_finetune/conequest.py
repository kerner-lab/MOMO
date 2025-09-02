
import cv2
import numpy as np
import os
import pandas as pd
from typing import Dict, Any

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
        image_path = row["input_path"]
        label_path = row["mask_path"]
        filename = image_path.split("/")[-1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=0)

        if self.transform:
            image = self.transform(image)

        return image, label, filename


class ConeQuestDataset(BaseDataset):

    def __init__(self, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args):
        super().__init__(config, train_transform, val_transform)
        self.data_dir = config["data_dir"]
        self.num_classes = config["num_classes"]
        self.task_type = config["task_type"]
        self.balance = config["balance"]
        self.batch_size = args.batch_size

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _prepare_data(self):

        # filepath_list, feature_list, set_list = [], [], []

        data_df = pd.read_csv(os.path.join(self.data_dir, "ConeQuest_data.csv"))
        data_df = data_df.loc[data_df["Number of Cones"]!=0]

        data_df["input_path"] = data_df.apply(lambda x: os.path.join(self.data_dir, 
                                                                    x["Patch Id"].split("_")[0] + "_" + x["Patch Id"].split("_")[1], 
                                                                    "input_dir",
                                                                    x["Patch Id"]), axis=1)
        data_df["mask_path"] = data_df.apply(lambda x: os.path.join(self.data_dir,
                                                                   x["Patch Id"].split("_")[0] + "_" + x["Patch Id"].split("_")[1], 
                                                                   "output_dir",
                                                                   x["Patch Id"]), axis=1)

        train_df = data_df.loc[data_df["BM-1 Set"]=="train"]
        val_df = data_df.loc[data_df["BM-1 Set"]=="val"]
        test_df = data_df.loc[data_df["BM-1 Set"]=="test"]

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
