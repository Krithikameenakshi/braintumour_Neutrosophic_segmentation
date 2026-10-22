"""Baseline Brain Tumor Segmentation Algorithms.

Implements the 5 benchmark classical approaches:
1. Otsu Thresholding
2. k-means Clustering (k=3)
3. Fuzzy C-Means (FCM, c=3, m=2.0)
4. Plain Guo-Cheng Neutrosophic Thresholding
5. Tufail-style NS ROI (Fixed S-function on T, I = norm(SD_7x7)*norm(Sobel), Tt=0.9, Tf=0.1, Otsu for I)

For all multi-region methods, the brightest connected component is retained.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from scipy.cluster.vq import kmeans2
from skimage.filters import threshold_otsu
from skimage.measure import label
import cv2

from src.ns import neutrosophic_transform


def keep_brightest_connected_component(
    binary_mask: np.ndarray,
    intensity_img: np.ndarray,
) -> np.ndarray:
    """Retains the connected component having the highest average intensity.

    Args:
      binary_mask: 2D binary array {0, 1}
      intensity_img: 2D float image array in [0, 1]

    Returns:
      2D binary array {0, 1} with only the brightest component.
    """
    labeled, num_features = label(binary_mask > 0, return_num=True, connectivity=2)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    best_mean_val = -1.0
    best_lbl = 0

    for lbl in range(1, num_features + 1):
        comp = labeled == lbl
        mean_val = float(np.mean(intensity_img[comp]))
        if mean_val > best_mean_val:
            best_mean_val = mean_val
            best_lbl = lbl

    return (labeled == best_lbl).astype(np.uint8)


def baseline_otsu(g: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """1.

    Otsu thresholding within brain parenchyma.
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return np.zeros_like(g, dtype=np.uint8)

    pixels = g[valid]
    thresh = threshold_otsu(pixels)
    candidate_mask = (g > thresh) & valid
    return keep_brightest_connected_component(candidate_mask, g)


def baseline_kmeans(
    g: np.ndarray,
    brain_mask: np.ndarray,
    k: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """2.

    k-means clustering (k=3) within brain parenchyma.
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return np.zeros_like(g, dtype=np.uint8)

    pixels = g[valid].astype(np.float64)

    # Initialize deterministic centroids (20th, 50th, 80th percentiles)
    init_centroids = np.percentile(pixels, np.linspace(20, 80, k))
    centroids, labels = kmeans2(
        pixels, init_centroids, iter=20, minit="matrix", seed=seed
    )

    # Highest centroid is the hyperintense tumor class
    tumor_cluster = np.argmax(centroids)
    tumor_pixels = labels == tumor_cluster

    pred_mask = np.zeros_like(g, dtype=np.uint8)
    pred_mask[valid] = tumor_pixels.astype(np.uint8)

    return keep_brightest_connected_component(pred_mask, g)


def baseline_fuzzy_c_means(
    g: np.ndarray,
    brain_mask: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    max_iter: int = 100,
    error: float = 0.005,
    eps: float = 1e-8,
) -> np.ndarray:
    """3.

    Fuzzy C-Means (FCM) clustering within brain parenchyma.
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return np.zeros_like(g, dtype=np.uint8)

    X = g[valid].astype(np.float64)  # shape (N,)
    N = len(X)

    # Initialize centers from percentiles
    centers = np.percentile(X, np.linspace(20, 85, n_clusters))

    p = 2.0 / (m - 1.0)

    for _ in range(max_iter):
        # Distance matrix (N, C)
        dists = np.abs(X[:, None] - centers[None, :]) + eps

        # Membership update (N, C)
        inv_dists = dists ** (-p)
        sum_inv = np.sum(inv_dists, axis=1, keepdims=True)
        U = inv_dists / sum_inv

        # Center update
        u_m = U**m
        new_centers = np.sum(u_m * X[:, None], axis=0) / (
            np.sum(u_m, axis=0) + eps
        )

        if np.max(np.abs(new_centers - centers)) < error:
            centers = new_centers
            break
        centers = new_centers

    # Hard cluster assignment to maximum membership
    labels = np.argmax(U, axis=1)
    tumor_cluster = np.argmax(centers)

    pred_mask = np.zeros_like(g, dtype=np.uint8)
    pred_mask[valid] = (labels == tumor_cluster).astype(np.uint8)

    return keep_brightest_connected_component(pred_mask, g)


def baseline_guo_cheng_ns(
    g: np.ndarray,
    brain_mask: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """4.

    Plain Guo-Cheng Neutrosophic Thresholding.
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return np.zeros_like(g, dtype=np.uint8)

    T, I, F = neutrosophic_transform(g, window_size=window_size)

    tau_T = threshold_otsu(T[valid])
    tau_I = threshold_otsu(I[valid])

    candidate_mask = (T > tau_T) & (I < tau_I) & valid
    return keep_brightest_connected_component(candidate_mask, g)


def apply_s_function(
    x: np.ndarray,
    PL: float = 0.1,
    PM: float = 0.5,
    PR: float = 0.9,
    eps: float = 1e-8,
) -> np.ndarray:
    """Standard piecewise S-function for contrast enhancement."""
    denom = max(eps, PR - PL)
    out = np.zeros_like(x, dtype=np.float64)

    # Region 1: x <= PL -> 0
    # Region 2: PL < x <= PM
    r2 = (x > PL) & (x <= PM)
    out[r2] = 2.0 * (((x[r2] - PL) / denom) ** 2)

    # Region 3: PM < x < PR
    r3 = (x > PM) & (x < PR)
    out[r3] = 1.0 - 2.0 * (((x[r3] - PR) / denom) ** 2)

    # Region 4: x >= PR -> 1
    r4 = x >= PR
    out[r4] = 1.0

    return np.clip(out, 0.0, 1.0)


def baseline_tufail_ns(
    g: np.ndarray,
    brain_mask: np.ndarray,
    sd_window: int = 7,
    sobel_ksize: int = 3,
    tt: float = 0.9,
    tf: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    """5.

    Tufail-style NS ROI.

    Formula:
      - S-function on T (fixed Tt=0.9, Tf=0.1)
      - I = norm(SD_7x7) * norm(Sobel)
      - Otsu for I
      - Tumor ROI: (T_S >= 0.9) & (I < Otsu(I)) & M_b
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return np.zeros_like(g, dtype=np.uint8)

    g_f = g.astype(np.float64)

    # 1. T from local mean
    gbar = ndi.uniform_filter(g_f, size=5, mode="reflect")
    g_min = float(np.min(gbar[valid]))
    g_max = float(np.max(gbar[valid]))
    T = (
        (gbar - g_min) / (g_max - g_min + eps)
        if (g_max - g_min) >= eps
        else np.zeros_like(gbar)
    )

    # S-function on T with fixed Tf=0.1, Tt=0.9
    tm = 0.5 * (tf + tt)
    T_S = apply_s_function(T, PL=tf, PM=tm, PR=tt)

    # 2. I = norm(SD_7x7) * norm(Sobel)
    gx = cv2.Sobel(g_f, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(g_f, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_mag = np.sqrt(gx**2 + gy**2)

    mean_sq = ndi.uniform_filter(g_f**2, size=sd_window, mode="reflect")
    sq_mean = ndi.uniform_filter(g_f, size=sd_window, mode="reflect") ** 2
    local_sd = np.sqrt(np.maximum(0.0, mean_sq - sq_mean))

    g_sobel_min, g_sobel_max = float(np.min(grad_mag[valid])), float(
        np.max(grad_mag[valid])
    )
    sd_min, sd_max = float(np.min(local_sd[valid])), float(
        np.max(local_sd[valid])
    )

    norm_sobel = (
        (grad_mag - g_sobel_min) / (g_sobel_max - g_sobel_min + eps)
        if (g_sobel_max - g_sobel_min) >= eps
        else np.zeros_like(grad_mag)
    )
    norm_sd = (
        (local_sd - sd_min) / (sd_max - sd_min + eps)
        if (sd_max - sd_min) >= eps
        else np.zeros_like(local_sd)
    )

    I_tufail = np.clip(norm_sobel, 0.0, 1.0) * np.clip(norm_sd, 0.0, 1.0)

    # Otsu threshold for I
    tau_I = threshold_otsu(I_tufail[valid])

    # Tumor ROI
    candidate_mask = (T_S >= tt) & (I_tufail < tau_I) & valid
    return keep_brightest_connected_component(candidate_mask, g)
