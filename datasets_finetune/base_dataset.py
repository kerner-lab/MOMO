# This file should NOT be removed since it serves as a critical base class that enforces
# a consistent interface across all dataset implementations.
#
# It is actively used in:
# 1. datasets_finetune/conequest.py - Inherits from BaseDataset
# 2. datasets_finetune/dataset_factory.py - Uses BaseDataset as return type
# 3. Likely other dataset classes also inherit from it (martian_frost.py, hirise_landmark.py etc)
#    as seen in dataset_factory.py imports
#
# Removing this would require:
# 1. Duplicating the dataloader interface in each dataset class
# 2. Losing the guarantee that all datasets implement the required methods
# 3. Making the codebase more prone to errors and inconsistencies
#
# Recommendation: Keep this base class to maintain clean architecture and type safety

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision import transforms

class BaseDataset(ABC):
    def __init__(self, config: dict, train_transform: transforms.Compose, val_transform: transforms.Compose):
        self.config = config
        self.train_transform = train_transform
        self.val_transform = val_transform

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader:
        pass