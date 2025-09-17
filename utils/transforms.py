
import torchvision.transforms as transforms


def create_transforms(train_model, is_training=True):

    if "vit" in train_model:
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # Very small rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Mild color changes
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),  # Mild cropping
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.05)),  # Light erasing
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    else:
        if is_training:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
