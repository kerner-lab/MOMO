
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


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


def compute_object_metrics(y_true, y_pred, iou_threshold=0.5):
    """
    Compute object-level precision, recall, and F1 score.
    
    Args:
        y_true (np.ndarray): Binary ground truth mask
        y_pred (np.ndarray): Binary predicted mask
        iou_threshold (float): IoU threshold for matching
    
    Returns:
        dict: Metrics including precision, recall, f1, tp, fp, fn
    """
    
    tp, fp, fn = get_object_level_metrics(y_true, y_pred, iou_threshold)
    
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




'''
import rasterio.features
import shapely.geometry

def get_object_level_metrics(y_true, y_pred, iou_threshold=0.5):
    """
    Get object level metrics for a single mask / prediction pair.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.
        iou_threshold (float, optional): IoU threshold for matching predictions to ground truths. Defaults to 0.5.

    Returns
        tuple (int, int, int): Number of true positives, false positives, and false negatives.
    """

    if iou_threshold < 0.5:
        raise ValueError(
            "iou_threshold must be greater than 0.5"
        )  # If we go lower than 0.5 then it is possible for a single prediction to match with multiple ground truths and we have to do de-duplication
    y_true_shapes = []
    for geom, val in rasterio.features.shapes(y_true):
        if val == 1:
            y_true_shapes.append(shapely.geometry.shape(geom))

    y_pred_shapes = []
    for geom, val in rasterio.features.shapes(y_pred):
        if val == 1:
            y_pred_shapes.append(shapely.geometry.shape(geom))

    tps = 0
    fns = 0
    tp_is = set()  # keep track of which of the true shapes are true positives
    tp_js = set()  # keep track of which of the predicted shapes are true positives
    fn_is = set()  # keep track of which of the true shapes are false negatives
    matched_js = set()
    for i, y_true_shape in enumerate(y_true_shapes):
        matching_j = None
        for j, y_pred_shape in enumerate(y_pred_shapes):
            if y_true_shape.intersects(y_pred_shape):
                intersection = y_true_shape.intersection(y_pred_shape)
                union = y_true_shape.union(y_pred_shape)
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    matching_j = j
                    matched_js.add(j)
                    tp_js.add(j)
                    break
        if matching_j is not None:
            tp_is.add(i)
            tps += 1
        else:
            fn_is.add(i)
            fns += 1
    fps = len(y_pred_shapes) - len(matched_js)
    fp_js = (
        set(range(len(y_pred_shapes))) - matched_js
    )  # compute which of the predicted shapes are false positives

    # Create masks of the true positives, false positives, and false negatives
    # tp_i_mask = rasterio.features.rasterize([y_true_shapes[i] for i in tp_is], out_shape= y_true.shape)
    # tp_j_mask = rasterio.features.rasterize([y_pred_shapes[j] for j in tp_js], out_shape= y_pred.shape)
    # fp_j_mask = rasterio.features.rasterize([y_pred_shapes[j] for j in fp_js], out_shape= y_pred.shape)
    # fn_i_mask = rasterio.features.rasterize([y_true_shapes[i] for i in fn_is], out_shape= y_true.shape)

    return (tps, fps, fns)
'''