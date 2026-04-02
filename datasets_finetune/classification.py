
import numpy as np
import os
import pandas as pd
from PIL import Image

from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .base_dataset import BaseDataset
from utils.stratified_partition import stratified_sample_with_redistribution


class CustomDataset(Dataset):

    def __init__(self, data_dir: str, df, transform: transforms.Compose):
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.filenames = []

        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, "data", row["split"], row["feature_name"], row["file_id"])
            self.image_paths.append(image_path)
            self.labels.append(row["label"])
            self.filenames.append(row["file_id"])

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        if "change_cls" in self.data_dir:
            image_id_before = self.image_paths[idx]
            arr = np.array(Image.open(image_id_before)).astype(np.float32)
            arr_normalized = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            layer_before = np.array(Image.fromarray(arr_normalized.astype(np.uint8)))
            layer_zero = np.zeros((layer_before.shape[0], layer_before.shape[1]), dtype=layer_before.dtype)

            image_id_after = self.image_paths[idx].replace("before", "after")
            arr = np.array(Image.open(image_id_after)).astype(np.float32)
            arr_normalized = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            layer_after = np.array(Image.fromarray(arr_normalized.astype(np.uint8)))
            image = np.stack([layer_zero, layer_after, layer_before], axis=-1)
        else:
            arr = np.array(Image.open(self.image_paths[idx])).astype(np.float32)
            arr_normalized = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            image = np.array(Image.fromarray(arr_normalized.astype(np.uint8)).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, self.labels[idx], self.filenames[idx]



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
        self.args = args

        self.train_df, self.val_df, self.test_df = self._prepare_data()

    def _prepare_data(self):

        data_df = pd.read_csv(os.path.join(self.data_dir, "annotation.csv"))

        train_df = data_df[data_df["split"]=="train"]
        val_df = data_df[data_df["split"]=="val"]
        test_df = data_df[data_df["split"]=="test"]

        if self.few_shot:
            n_shots = int(self.few_shot.split("_")[0])
            train_df = (train_df.groupby('label', group_keys=False)
                       .apply(lambda x: x.sample(n=n_shots, random_state=self.args.seed)).reset_index(drop=True))
        elif self.partition:
            ratio = float(self.partition.split("x_")[0])
            train_df = stratified_sample_with_redistribution(train_df, "feature_name", ratio, seed=self.args.seed)
        else:
            pass

        if self.balance == "under_sample":
            min_samples = train_df['label'].value_counts().min()
            train_df = (train_df.groupby('label', group_keys=False)
                       .apply(lambda x: x.sample(n=min_samples, random_state= self.args.seed)).reset_index(drop=True))
        elif self.balance == "over_sample":
            X = train_df.drop(columns=['label'])
            y = train_df['label']
            ros = RandomOverSampler(random_state=self.args.seed)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            train_df = pd.concat([X_resampled, pd.Series(y_resampled, name='label')], axis=1)
            train_df['label'] = train_df['label'].astype('int32')

        return train_df, val_df, test_df

    def _create_dataset(self, df: pd.DataFrame, is_training: bool) -> Dataset:
        return CustomDataset(self.data_dir, df, self.train_transform if is_training else self.val_transform)

    def _worker_init_fn(self, worker_id):
        """Seed each DataLoader worker for reproducible augmentations (albumentations uses numpy/random)."""
        import random
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._create_dataset(self.train_df, is_training=True)
        g = torch.Generator()
        g.manual_seed(self.args.seed)
        sampler_train = torch.utils.data.RandomSampler(train_dataset, generator=g)
        # persistent_workers=False for reproducibility
        train_dataloader = DataLoader(
            train_dataset, sampler=sampler_train,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=False,
            pin_memory=self.pin_mem,
            prefetch_factor=2,
            generator=g,
            worker_init_fn=self._worker_init_fn
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
            pin_memory=self.pin_mem,
            prefetch_factor=2
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
            pin_memory=self.pin_mem,
            prefetch_factor=2
        )
        return test_dataloader

    def get_class_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df['label']),
            y=self.train_df['label']
        )
        # class_weight_dict = dict(zip(np.unique(self.train_df['label']), class_weights))
        # print(class_weight_dict)
        return class_weights
