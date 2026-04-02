
from typing import Dict, Any
from torchvision import transforms
from .base_dataset import BaseDataset

from .classification import ClassificationDataset
from .segmentation import SegmentationDataset

class DatasetFactory:
    @staticmethod
    def create_dataset(config: Dict[str, Any], train_transform: transforms.Compose, val_transform: transforms.Compose, args) -> BaseDataset:
        # Get task type from config to determine which dataset class to use
        task_type = config.get("task_type", "").lower()
        
        if "segmentation" in task_type:
            return SegmentationDataset(config, train_transform, val_transform, args)
        elif "classification" in task_type:
            return ClassificationDataset(config, train_transform, val_transform, args)
        else:
            raise ValueError(f"Unknown task type: {task_type}. Expected 'segmentation' or 'classification' in task_type.")
