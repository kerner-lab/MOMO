
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
    
    if num_classes == 2:
        # Binary segmentation
        positive_pixels = 0
        total_pixels = 0
        
        for _, targets, _ in dataset_loader:
            total_pixels += targets.numel()
            positive_pixels += (targets > 0).sum().item()
        
        negative_pixels = total_pixels - positive_pixels
        
        if method == 'inverse_frequency':
            # Inverse frequency weighting
            pos_weight = total_pixels / (2 * positive_pixels) if positive_pixels > 0 else 1.0
            neg_weight = total_pixels / (2 * negative_pixels) if negative_pixels > 0 else 1.0
            weights = torch.tensor([neg_weight, pos_weight])
            
        else:  # effective_number
            beta = 0.9999
            effective_num_neg = (1 - beta**negative_pixels) / (1 - beta)
            effective_num_pos = (1 - beta**positive_pixels) / (1 - beta)
            weights = torch.tensor([1/effective_num_neg, 1/effective_num_pos])
            weights = weights / weights.sum() * 2  # Normalize
    
    else:
        # Multiclass segmentation
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
    print(f"Class distribution: {class_counts if num_classes > 2 else f'Negative: {total_pixels - positive_pixels}, Positive: {positive_pixels}'}")
    
    return weights


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes edges between objects.
    """
    def __init__(self, boundary_weight=4.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target):
        # Ensure target is float and has correct shape
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()
        
        # Create 3x3 kernel for morphological operations
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        
        # Erode target to get inner region
        target_eroded = F.conv2d(target, kernel, padding=1)
        target_eroded = (target_eroded == 9).float()
        
        # Boundary is original - eroded
        boundary = target - target_eroded
        
        # Compute BCE with logits (autocast-safe)
        # Note: pred should be logits (not sigmoid applied)
        boundary_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') #TODO: update bce
        boundary_loss = boundary_loss * (1 + self.boundary_weight * boundary)
        
        return boundary_loss.mean()




class WeightedCombinedLoss(nn.Module):

    def __init__(self, num_classes, class_weights=None, weight_dice=0.5, weight_bce=0.5, weight_boundary=0.0):
        """
        Initializes the weighted combined loss function.
        Args:
            num_classes (int): Number of output classes.
            class_weights (torch.Tensor or list): Weights for each class. 
                                                 For binary: [background_weight, foreground_weight]
                                                 For multiclass: [class0_weight, class1_weight, ...]
            weight_dice (float): Weight for the Dice Loss component.
            weight_bce (float): Weight for the BCEWithLogitsLoss or CrossEntropyLoss component.
            weight_boundary (float): Weight for the Boundary Loss component (0 to disable).
        """
        super(WeightedCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce #TODO: update bce
        self.weight_boundary = weight_boundary

        # Convert class_weights to tensor if provided
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.class_weights = class_weights
        else:
            self.class_weights = None

        # Define individual losses with class weights
        if num_classes == 2:
            # Binary segmentation
            self.dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

            # For binary BCE, we'll handle weighting manually since BCEWithLogitsLoss
            # doesn't support class weights the same way
            self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none') #TODO: update bce

        else:
            # Multiclass segmentation
            self.dice_loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
            self.bce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean') #TODO: update bce
        
        # Boundary loss (only for binary segmentation)
        if weight_boundary > 0 and num_classes == 2:
            self.boundary_loss_fn = BoundaryLoss()
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
        dice_loss = self.dice_loss_fn(logits, targets)
        
        if self.num_classes == 2:
            # Binary case - apply manual weighting
            bce_loss_raw = self.bce_loss_fn(logits, targets) #TODO: update bce
            
            if self.class_weights is not None:
                # Apply class weights manually for binary case
                weight_map = torch.where(targets == 1, 
                                       self.class_weights[1].to(targets.device), 
                                       self.class_weights[0].to(targets.device))
                bce_loss = (bce_loss_raw * weight_map).mean() #TODO: update bce
            else:
                bce_loss = bce_loss_raw.mean()
        else:
            # Multiclass case - CrossEntropyLoss handles weighting automatically
            bce_loss = self.bce_loss_fn(logits, targets) #TODO: update bce

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + (self.weight_bce * bce_loss)
        
        # Add boundary loss if enabled
        if self.boundary_loss_fn is not None and self.weight_boundary > 0:
            boundary_loss = self.boundary_loss_fn(logits, targets)
            combined_loss = combined_loss + (self.weight_boundary * boundary_loss)
        
        return combined_loss


# Alternative: Focal Loss for extreme imbalance
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2, weight_dice=0.5, weight_focal=0.5):
        super(FocalDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        
        self.dice_loss = smp.losses.DiceLoss(
            mode="binary" if num_classes == 2 else "multiclass", 
            from_logits=True
        )
        
        if num_classes == 2:
            self.focal_loss = smp.losses.FocalLoss(mode="binary", alpha=alpha, gamma=gamma)
        else:
            self.focal_loss = smp.losses.FocalLoss(mode="multiclass", alpha=alpha, gamma=gamma)
    
    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return (self.weight_dice * dice_loss) + (self.weight_focal * focal_loss)



class CombinedLoss(nn.Module):

    def __init__(self, num_classes, weight_dice=0.5, weight_bce=0.5):
        """
        Initializes the combined loss function.
        Args:
            num_classes (int): Number of output classes.
            weight_dice (float): Weight for the Dice Loss component.
            weight_bce (float): Weight for the BCEWithLogitsLoss or CrossEntropyLoss component.
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce #TODO: update bce

        # Define individual losses
        self.dice_loss_fn = smp.losses.DiceLoss(mode="binary" if num_classes == 2 else "multiclass", from_logits=True)
        self.bce_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss() #TODO: update bce

    def forward(self, logits, targets):
        """
        Computes the combined loss.
        Args:
            logits (torch.Tensor): Raw model outputs (logits).
            targets (torch.Tensor): Ground truth targets.
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice_loss = self.dice_loss_fn(logits, targets)
        bce_loss = self.bce_loss_fn(logits, targets)  # CrossEntropyLoss directly uses logits #TODO: update bce

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + \
                        (self.weight_bce * bce_loss)

        return combined_loss
