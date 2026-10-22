"""Evaluation Metrics for Medical Image Segmentation.

Implements:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard)
- Sensitivity (Recall)
- Precision (Positive Predictive Value)
- False Negative Rate (FNR)
- 95th Percentile Hausdorff Distance (HD95)

Explicitly handles empty ground truth without silent fabrication of 0 or 1.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
import cv2


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Computes the 95th percentile bidirectional Hausdorff Distance (HD95) in

    pixels.

    Args:
      pred: 2D binary numpy array {0, 1}
      gt: 2D binary numpy array {0, 1}

    Returns:
      HD95 distance in pixels, or np.nan if either mask is empty.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # If either mask is empty, Hausdorff distance is mathematically undefined
    if not np.any(pred_bool) or not np.any(gt_bool):
        return np.nan

    # Both non-empty: extract boundary coordinates using contour extraction
    pred_contours, _ = cv2.findContours(
        pred_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    gt_contours, _ = cv2.findContours(
        gt_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return np.nan

    pts_pred = np.vstack(
        [c.squeeze(axis=1) for c in pred_contours if len(c) > 0]
    )
    pts_gt = np.vstack([c.squeeze(axis=1) for c in gt_contours if len(c) > 0])

    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return np.nan

    # Pairwise Euclidean distances
    dists = cdist(pts_pred, pts_gt, metric="euclidean")

    # Directed minimum distances
    min_d_pred_to_gt = np.min(dists, axis=1)  # distance from each pred to gt
    min_d_gt_to_pred = np.min(dists, axis=0)  # distance from each gt to pred

    # 95th percentiles
    hd95_pred_to_gt = float(np.percentile(min_d_pred_to_gt, 95))
    hd95_gt_to_pred = float(np.percentile(min_d_gt_to_pred, 95))

    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, Union[float, bool, int]]:
    """Calculates all clinical segmentation metrics between prediction and

    ground truth.

    Handles empty ground truth slices explicitly.

    Returns dict containing:
      is_empty_gt: bool
      pred_empty: bool
      tp, fp, fn, tn: int
      dice: float or np.nan (if empty GT)
      iou: float or np.nan (if empty GT)
      sensitivity: float or np.nan (if empty GT)
      precision: float or np.nan (if empty GT)
      fnr: float or np.nan (if empty GT)
      hd95: float or np.nan
      false_positive_rate_empty: float or np.nan (if empty GT, indicates whether pred fired)
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)

    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))

    is_empty_gt = bool(np.sum(g) == 0)
    pred_empty = bool(np.sum(p) == 0)

    # Explicit handling of empty ground truth (no-tumor slices)
    if is_empty_gt:
        return {
            "is_empty_gt": True,
            "pred_empty": pred_empty,
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "tn": tn,
            "dice": np.nan,  # Never silently fabricate 1.0 or 0.0
            "iou": np.nan,
            "sensitivity": np.nan,
            "precision": np.nan,
            "fnr": np.nan,
            "hd95": np.nan,
            "false_alarm": not pred_empty,
            "fp_pixel_count": fp,
        }

    # Standard metrics for slices containing tumor (GT > 0)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = float(tp) / (tp + fp + fn + eps)
    sensitivity = float(tp) / (tp + fn + eps)
    precision = float(tp) / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    fnr = float(fn) / (tp + fn + eps)
    hd95_val = compute_hd95(p, g)

    return {
        "is_empty_gt": False,
        "pred_empty": pred_empty,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "fnr": float(fnr),
        "hd95": float(hd95_val),
        "false_alarm": False,
        "fp_pixel_count": fp,
    }
