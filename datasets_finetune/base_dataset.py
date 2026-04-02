
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