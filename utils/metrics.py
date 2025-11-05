
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage import measure


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def _squeeze_to_2d(mask):
    """
    Convert mask to 2D by squeezing all singleton dimensions.
    """
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask.squeeze()
    return mask


def get_object_level_metrics(y_true, y_pred, iou_threshold=0.5):
    """
    Compute object-level metrics (TP, FP, FN) using connected component analysis.
    
    Args:
        y_true (np.ndarray): Ground truth mask (can be 2D or 3D with channel dimension)
        y_pred (np.ndarray): Predicted mask (can be 2D or 3D with channel dimension)
        iou_threshold (float): IoU threshold for matching objects
    
    Returns:
        tuple: (true_positives, false_positives, false_negatives)
    """
    
    y_true = _squeeze_to_2d(y_true)
    y_pred = _squeeze_to_2d(y_pred)
    
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    
    # Label connected components
    true_labeled, num_true = ndimage.label(y_true)
    pred_labeled, num_pred = ndimage.label(y_pred)
    
    if num_true == 0 and num_pred == 0:
        return 0, 0, 0
    elif num_true == 0:
        return 0, num_pred, 0
    elif num_pred == 0:
        return 0, 0, num_true
    
    # Compute IoU matrix between all object pairs
    iou_matrix = np.zeros((num_true, num_pred))
    for i in range(1, num_true + 1):
        true_mask = true_labeled == i
        for j in range(1, num_pred + 1):
            pred_mask = pred_labeled == j
            iou_matrix[i-1, j-1] = compute_iou(true_mask, pred_mask)
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    # Count matches above threshold
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            matched_pairs.append((i, j))
    
    tp = len(matched_pairs)
    fp = num_pred - tp
    fn = num_true - tp
    
    return tp, fp, fn


def compute_object_metrics(y_true, y_pred, iou_threshold=0.5, num_classes=2):
    """
    Compute object-level precision, recall, and F1 score.

    Args:
        y_true (np.ndarray): Ground truth mask (binary or multi-class)
        y_pred (np.ndarray): Predicted mask (binary or multi-class)
        iou_threshold (float): IoU threshold for matching
        num_classes (int): Number of classes (2 for binary, >2 for multi-class)

    Returns:
        dict: Metrics including precision, recall, f1, tp, fp, fn
    """

    if num_classes == 2:
        # Binary segmentation - use existing logic
        tp, fp, fn = get_object_level_metrics(y_true, y_pred, iou_threshold)
    else:
        # Multi-class segmentation - compute metrics per class and average
        total_tp, total_fp, total_fn = 0, 0, 0

        for class_id in range(1, num_classes):  # Skip background (class 0)
            # Create binary masks for current class
            y_true_binary = (y_true == class_id).astype(np.uint8)
            y_pred_binary = (y_pred == class_id).astype(np.uint8)

            tp, fp, fn = get_object_level_metrics(y_true_binary, y_pred_binary, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        tp, fp, fn = total_tp, total_fp, total_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compute_batch_object_metrics(y_true_batch, y_pred_batch, iou_threshold=0.5):
    """
    Compute object-level metrics for a batch of images.
    
    Args:
        y_true_batch (np.ndarray): Batch of ground truth masks (can be [B, H, W] or [B, C, H, W])
        y_pred_batch (np.ndarray): Batch of predicted masks (can be [B, H, W] or [B, C, H, W])
        iou_threshold (float): IoU threshold for matching
    
    Returns:
        dict: Averaged metrics across batch
    """
    
    y_true_batch = np.asarray(y_true_batch)
    y_pred_batch = np.asarray(y_pred_batch)
    
    # Remove batch dimension if present
    if y_true_batch.ndim == 4:  # [B, C, H, W]
        y_true_batch = y_true_batch.squeeze(0)
        y_pred_batch = y_pred_batch.squeeze(0)
    
    batch_size = y_true_batch.shape[0] if y_true_batch.ndim == 3 else 1
    total_tp, total_fp, total_fn = 0, 0, 0
    
    if y_true_batch.ndim == 3:  # [B, H, W]
        for i in range(batch_size):
            tp, fp, fn = get_object_level_metrics(y_true_batch[i], y_pred_batch[i], iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
    else:  # [H, W] or [C, H, W]
        tp, fp, fn = get_object_level_metrics(y_true_batch, y_pred_batch, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compute_map_segmentation(labels_np, pred_np, posterior_np, num_classes, 
                             iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                             confidence_thresholds=None):
    """
    Compute mAP for instance segmentation.
    
    Args:
        labels_np: Ground truth mask (H, W) with class indices
        pred_np: Predicted mask (H, W) with class indices
        posterior_np: Class probabilities (num_classes, H, W)
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds for mAP calculation
        confidence_thresholds: Optional list of confidence thresholds
    
    Returns:
        Dictionary with mAP metrics
    """
    
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # Extract connected components (instances) from ground truth and predictions
    gt_instances = []
    pred_instances = []
    
    # Process each class (skip background class 0)
    for class_idx in range(1, num_classes):
        # Ground truth instances for this class
        gt_mask_class = (labels_np == class_idx).astype(np.uint8)
        gt_labeled = measure.label(gt_mask_class, connectivity=2)
        
        for region in measure.regionprops(gt_labeled):
            gt_instances.append({
                'class': class_idx,
                'mask': (gt_labeled == region.label),
                'area': region.area,
                'bbox': region.bbox
            })
        
        # Predicted instances for this class
        pred_mask_class = (pred_np == class_idx).astype(np.uint8)
        pred_labeled = measure.label(pred_mask_class, connectivity=2)
        
        # Get confidence scores for each predicted instance
        for region in measure.regionprops(pred_labeled):
            instance_mask = (pred_labeled == region.label)
            # Average confidence score over the instance pixels
            confidence = posterior_np[class_idx][instance_mask].mean()
            
            pred_instances.append({
                'class': class_idx,
                'mask': instance_mask,
                'area': region.area,
                'bbox': region.bbox,
                'confidence': confidence
            })
    
    # Sort predictions by confidence (descending)
    pred_instances.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate AP for each IoU threshold
    aps = []
    
    for iou_threshold in iou_thresholds:
        # Track which ground truth instances have been matched
        gt_matched = [False] * len(gt_instances)
        
        # Lists to store precision and recall values
        precisions = []
        recalls = []
        
        tp = 0
        fp = 0
        
        for pred_idx, pred in enumerate(pred_instances):
            # Find best matching ground truth instance
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_instances):
                # Only match same class
                if pred['class'] != gt['class']:
                    continue
                
                # Calculate IoU
                intersection = np.logical_and(pred['mask'], gt['mask']).sum()
                union = np.logical_or(pred['mask'], gt['mask']).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if this prediction matches a ground truth
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(gt_instances) if len(gt_instances) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation or all-point interpolation
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    
    # Calculate mAP as mean of APs across IoU thresholds
    map_score = np.mean(aps)
    
    # Also calculate AP at specific IoU thresholds (like COCO metrics)
    map_50 = aps[0] if len(aps) > 0 else 0  # AP@0.5
    map_75 = aps[5] if len(aps) > 5 else 0  # AP@0.75
    
    return {
        'mAP': map_score,
        'mAP@0.5': map_50,
        'mAP@0.75': map_75,
        'AP_per_iou': dict(zip(iou_thresholds, aps))
    }


def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision using 11-point interpolation.
    """
    if len(recalls) == 0:
        return 0.0
    
    # Add sentinel values at the beginning and end
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for threshold in np.arange(0, 1.1, 0.1):
        # Find recalls that are >= threshold
        valid_indices = [i for i, r in enumerate(recalls) if r >= threshold]
        if len(valid_indices) > 0:
            ap += max([precisions[i] for i in valid_indices])
    
    ap /= 11.0
    return ap


def compute_pixel_based_map(labels_np, posterior_np, num_classes,
                            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                            confidence_thresholds=None):
    """
    Compute mAP using pixel-level predictions at different confidence thresholds.
    This is simpler but less accurate than instance-based mAP.
    
    Args:
        labels_np: Ground truth mask (H, W) with class indices
        posterior_np: Class probabilities (num_classes, H, W)
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds
        confidence_thresholds: List of confidence thresholds to evaluate
    
    Returns:
        Dictionary with mAP metrics
    """
    if confidence_thresholds is None:
        confidence_thresholds = np.arange(0.1, 1.0, 0.05)
    
    aps_per_class = []
    
    # Calculate AP for each class (skip background)
    for class_idx in range(1, num_classes):
        precisions = []
        recalls = []
        
        gt_mask = (labels_np == class_idx)
        gt_positive = gt_mask.sum()
        
        if gt_positive == 0:
            continue
        
        # Evaluate at different confidence thresholds
        for conf_threshold in confidence_thresholds:
            pred_mask = posterior_np[class_idx] >= conf_threshold
            
            tp = np.logical_and(pred_mask, gt_mask).sum()
            fp = np.logical_and(pred_mask, ~gt_mask).sum()
            fn = np.logical_and(~pred_mask, gt_mask).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP for this class
        ap = calculate_ap(recalls, precisions)
        aps_per_class.append(ap)
    
    # Calculate mAP
    map_score = np.mean(aps_per_class) if len(aps_per_class) > 0 else 0.0
    
    return {
        'mAP': map_score,
        'AP_per_class': aps_per_class,
        'num_classes_evaluated': len(aps_per_class)
    }
