from typing import Dict, Any
from torchvision import transforms
from .base_dataset import BaseDataset

from .classification import HiRISELandmarkDataset, DoMars16Dataset, AtmosphericDustDataset, MartianFrostDataset
from .conequest import ConeQuestDataset

class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name: str, config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args) -> BaseDataset:
        if dataset_name == "martian_frost":
            return MartianFrostDataset(config, train_transform, val_transform, args)
        elif dataset_name == "hirise_landmark":
            return HiRISELandmarkDataset(config, train_transform, val_transform, args)
        elif dataset_name == "domars16":
            return DoMars16Dataset(config, train_transform, val_transform, args)
        elif (dataset_name == "atmospheric_dust_edr") or (dataset_name == "atmospheric_dust_rdr"):
            return AtmosphericDustDataset(config, train_transform, val_transform, args)
        elif dataset_name == "conequest":
            return ConeQuestDataset(config, train_transform, val_transform, args)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
