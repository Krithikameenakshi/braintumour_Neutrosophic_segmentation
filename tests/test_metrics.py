"""Unit tests for segmentation metrics module."""

import numpy as np
import pytest
from src.metrics import compute_metrics, compute_hd95


def test_metrics_perfect_overlap():
    """Identical prediction and GT should have perfect scores and HD95 = 0."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 1

    res = compute_metrics(mask, mask)
    assert not res["is_empty_gt"]
    assert res["dice"] == pytest.approx(1.0, abs=1e-5)
    assert res["iou"] == pytest.approx(1.0, abs=1e-5)
    assert res["sensitivity"] == pytest.approx(1.0, abs=1e-5)
    assert res["precision"] == pytest.approx(1.0, abs=1e-5)
    assert res["fnr"] == pytest.approx(0.0, abs=1e-5)
    assert res["hd95"] == pytest.approx(0.0, abs=1e-5)


def test_metrics_disjoint_masks():
    """Completely non-overlapping masks."""
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[5:15, 5:15] = 1

    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[35:45, 35:45] = 1

    res = compute_metrics(pred, gt)
    assert not res["is_empty_gt"]
    assert res["dice"] == pytest.approx(0.0, abs=1e-5)
    assert res["iou"] == pytest.approx(0.0, abs=1e-5)
    assert res["sensitivity"] == pytest.approx(0.0, abs=1e-5)
    assert res["precision"] == pytest.approx(0.0, abs=1e-5)
    assert res["fnr"] == pytest.approx(1.0, abs=1e-5)
    assert res["hd95"] > 20.0  # large distance


def test_metrics_partial_overlap():
    """Known partial overlap."""
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[0:10, 0:10] = 1  # area 100

    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:15, 0:10] = 1  # area 100, overlap is 5:10, 0:10 = area 50

    res = compute_metrics(pred, gt)
    # TP = 50, FP = 50, FN = 50
    # Dice = 2*50 / (100 + 100) = 0.5
    # IoU = 50 / 150 = 1/3
    assert res["dice"] == pytest.approx(0.5, abs=1e-4)
    assert res["iou"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert res["sensitivity"] == pytest.approx(0.5, abs=1e-4)
    assert res["precision"] == pytest.approx(0.5, abs=1e-4)
    assert res["fnr"] == pytest.approx(0.5, abs=1e-4)


def test_empty_ground_truth_explicit_handling():
    """Verifies that empty ground truth is NOT silently treated as Dice=0 or 1."""
    empty_gt = np.zeros((40, 40), dtype=np.uint8)
    empty_pred = np.zeros((40, 40), dtype=np.uint8)

    # 1. True negative (no tumor, correctly predicted empty)
    res_tn = compute_metrics(empty_pred, empty_gt)
    assert res_tn["is_empty_gt"] is True
    assert res_tn["pred_empty"] is True
    assert np.isnan(res_tn["dice"])  # NaN, not 1.0 or 0.0!
    assert res_tn["false_alarm"] is False
    assert res_tn["fp_pixel_count"] == 0

    # 2. False positive (no tumor, but model predicted a false region)
    noisy_pred = np.zeros((40, 40), dtype=np.uint8)
    noisy_pred[10:15, 10:15] = 1

    res_fp = compute_metrics(noisy_pred, empty_gt)
    assert res_fp["is_empty_gt"] is True
    assert res_fp["pred_empty"] is False
    assert np.isnan(res_fp["dice"])
    assert res_fp["false_alarm"] is True
    assert res_fp["fp_pixel_count"] == 25
