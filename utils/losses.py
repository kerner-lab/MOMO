
import torch.nn as nn
import segmentation_models_pytorch as smp


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
        self.weight_bce = weight_bce

        # Define individual losses
        self.dice_loss_fn = smp.losses.DiceLoss(mode="binary" if num_classes == 1 else "multiclass", from_logits=True)
        self.bce_loss_fn = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

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
        bce_loss = self.bce_loss_fn(logits, targets)  # CrossEntropyLoss directly uses logits

        # Combine the losses
        combined_loss = (self.weight_dice * dice_loss) + \
                        (self.weight_bce * bce_loss)

        return combined_loss
