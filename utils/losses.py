import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def compute_class_weights(dataset_loader, num_classes, method='inverse_frequency'):
    """
    Compute class weights from the dataset to handle class imbalance.
    
    Args:
        dataset_loader: DataLoader for the dataset
        num_classes (int): Number of classes
        method (str): Method to compute weights ('inverse_frequency' or 'effective_number')
    
    Returns:
        torch.Tensor: Class weights
    """
    print("Computing class weights from dataset...")
    
    # Unified approach for both binary and multiclass
    class_counts = torch.zeros(num_classes) 
    
    for _, targets, _ in dataset_loader:
        for c in range(num_classes): 
            class_counts[c] += (targets == c).sum().item()
    
    total_samples = class_counts.sum()
    
    if method == 'inverse_frequency':
        weights = total_samples / (num_classes * class_counts) 
        # Avoid infinite weights for classes with 0 samples
        weights = torch.where(class_counts == 0, torch.tensor(1.0), weights)
    else:  # effective_number
        beta = 0.9999
        effective_nums = (1 - beta**class_counts) / (1 - beta)
        weights = (1 - beta) / effective_nums
        weights = weights / weights.sum() * num_classes  # Normalize 
    
    print(f"Class weights: {weights}")
    print(f"Class distribution: {class_counts}")
    
    return weights


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes edges between objects.
    Works for both binary and multiclass segmentation.
    """
    def __init__(self, num_classes=2, boundary_weight=4.0):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: Logits of shape (B, C, H, W)
            target: Target of shape (B, H, W) with class indices
        """
        # Ensure target has correct shape
        if target.dim() == 4:
            target = target.squeeze(1)
        
        # Compute boundaries for each class
        boundaries = torch.zeros(target.shape[0], self.num_classes, target.shape[1], target.shape[2], 
                                device=target.device)
        
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        
        for c in range(self.num_classes):
            # Create binary mask for class c
            class_mask = (target == c).float().unsqueeze(1)
            
            # Erode the mask
            class_eroded = F.conv2d(class_mask, kernel, padding=1)
            class_eroded = (class_eroded == 9).float()
            
            # Boundary is original - eroded
            boundaries[:, c:c+1, :, :] = class_mask - class_eroded
        
        # Compute CrossEntropy loss per pixel
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Weight the loss based on boundaries (any class boundary)
        boundary_mask = boundaries.sum(dim=1)  # (B, H, W)
        weighted_loss = ce_loss * (1 + self.boundary_weight * boundary_mask)
        
        return weighted_loss.mean()


class WeightedCombinedLoss(nn.Module):

    def __init__(self, num_classes, class_weights=None, weight_dice=0.5, weight_ce=0.5, weight_boundary=0.0):
        """
        Initializes the weighted combined loss function.
        Args:
            num_classes (int): Number of output classes.
            class_weights (torch.Tensor or list): Weights for each class. 
                                                 For binary: [background_weight, foreground_weight]
                                                 For multiclass: [class0_weight, class1_weight, ...]
            weight_dice (float): Weight for the Dice Loss component.
            weight_ce (float): Weight for the CrossEntropyLoss component.
            weight_boundary (float): Weight for the Boundary Loss component (0 to disable).
        """
        super(WeightedCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_boundary = weight_boundary

        # Convert class_weights to tensor if provided
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = class_weights
        else:
            self.class_weights = None

        # Define individual losses with class weights
        # Since we're using multi-channel outputs for both binary (2 channels) and multiclass (N channels),
        # we use "multiclass" mode for DiceLoss in all cases
        self.dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        
        # CrossEntropyLoss works for both binary and multiclass
        # TODO: verify reduction
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        
        # Boundary loss (for both binary and multiclass if enabled)
        if weight_boundary > 0:
            self.boundary_loss_fn = BoundaryLoss(num_classes=num_classes)
        else:
            self.boundary_loss_fn = None

    def forward(self, logits, targets):
        """
        Computes the weighted combined loss.
        Args:
            logits (torch.Tensor): Raw model outputs (logits).
            targets (torch.Tensor): Ground truth targets.
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Ensure targets are of correct shape and dtype
        if targets.dim() == 4:
            targets = targets.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        targets = targets.long()  # Ensure long type for cross-entropy and dice

        dice_loss = self.dice_loss_fn(logits, targets)
        ce_loss = self.ce_loss_fn(logits, targets)

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + (self.weight_ce * ce_loss)
        
        # Add boundary loss if enabled
        if self.boundary_loss_fn is not None and self.weight_boundary > 0:
            boundary_loss = self.boundary_loss_fn(logits, targets)
            combined_loss = combined_loss + (self.weight_boundary * boundary_loss)
        
        return combined_loss



class CombinedLoss(nn.Module):

    def __init__(self, num_classes, weight_dice=0.5, weight_ce=0.5):
        """
        Initializes the combined loss function.
        Args:
            num_classes (int): Number of output classes.
            weight_dice (float): Weight for the Dice Loss component.
            weight_ce (float): Weight for the CrossEntropyLoss component.
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        # Define individual losses
        # Since we're using multi-channel outputs for both binary (2 channels) and multiclass (N channels),
        # we use "multiclass" mode for DiceLoss in all cases
        self.dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

        # CrossEntropyLoss works for both binary and multiclass
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Computes the combined loss.
        Args:
            logits (torch.Tensor): Raw model outputs (logits).
            targets (torch.Tensor): Ground truth targets.
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Ensure targets are of correct shape and dtype
        if targets.dim() == 4:
            targets = targets.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        targets = targets.long()  # Ensure long type for cross-entropy and dice

        dice_loss = self.dice_loss_fn(logits, targets)
        ce_loss = self.ce_loss_fn(logits, targets)

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + (self.weight_ce * ce_loss)

        return combined_loss