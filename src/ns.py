"""Neutrosophic Set (NS) Transformation and Entropy Module.

Implements the verified Guo & Sengur (2013) Neutrosophic domain mapping:
  gbar = mean_{w x w}(g)
  T = (gbar - gbar_min) / (gbar_max - gbar_min)
  delta = |g - gbar|
  I = (delta - delta_min) / (delta_max - delta_min)
  F = 1 - T

Entropy:
  En(X) = -sum p_X(l) * ln(p_X(l)) computed over a 256-bin histogram within M_b.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi


def neutrosophic_transform(
    g: np.ndarray,
    window_size: int = 5,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Neutrosophic Set (T, I, F) transformation (Guo & Sengur

    2013).

    Args:
      g: 2D float image array with values in [0, 1].
      window_size: Size w of the w x w local averaging window (default: 5).
      eps: Small constant to prevent division by zero when min == max.

    Returns:
      T: Truth-membership map in [0, 1]
      I: Indeterminacy-membership map in [0, 1]
      F: Falsehood-membership map in [0, 1]
    """
    g_f = g.astype(np.float64)

    # 1. Local window mean gbar
    gbar = ndi.uniform_filter(
        g_f, size=window_size, mode="reflect"
    )  # mean_{w x w}(g)

    # 2. T-membership: normalized mean intensity
    gbar_min = float(np.min(gbar))
    gbar_max = float(np.max(gbar))
    denom_t = gbar_max - gbar_min
    if denom_t < eps:
        t_map = np.zeros_like(gbar)
    else:
        t_map = (gbar - gbar_min) / (denom_t + eps)

    # 3. I-membership: normalized absolute local deviation (indeterminacy)
    delta = np.abs(g_f - gbar)
    delta_min = float(np.min(delta))
    delta_max = float(np.max(delta))
    denom_i = delta_max - delta_min
    if denom_i < eps:
        i_map = np.zeros_like(delta)
    else:
        i_map = (delta - delta_min) / (denom_i + eps)

    # 4. F-membership: complement of truth
    t_map = np.clip(t_map, 0.0, 1.0)
    i_map = np.clip(i_map, 0.0, 1.0)
    f_map = np.clip(1.0 - t_map, 0.0, 1.0)

    return t_map, i_map, f_map


def compute_entropy(
    x: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bins: int = 256,
) -> float:
    """Computes Shannon entropy En(X) over a 256-bin histogram within mask M_b.

    Formula:
      En(X) = -sum_{l} p_X(l) * ln(p_X(l))

    Args:
      x: 2D array (e.g. T, I, or F map).
      mask: Optional binary mask (e.g. brain mask M_b).
      bins: Number of histogram bins (default: 256).

    Returns:
      Entropy value in nats (natural log).
    """
    if mask is not None:
        values = x[mask > 0]
    else:
        values = x.ravel()

    if len(values) == 0:
        return 0.0

    # Values for T, I, F are strictly in [0.0, 1.0]
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total_count = np.sum(hist)
    if total_count == 0:
        return 0.0

    # Discrete probability distribution
    probs = hist / total_count
    # Consider only non-zero probability bins
    nz_probs = probs[probs > 0]

    entropy = -float(np.sum(nz_probs * np.log(nz_probs)))
    return entropy


def enhanced_neutrosophic_transform(
    g: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    window_size: int = 5,
    sd_window: int = 7,
    sobel_ksize: int = 3,
    w_delta: float = 0.5,
    w_gradient_sd: float = 0.5,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Enhanced Multi-Scale & Gradient-Aware Neutrosophic Transform.

    Overcomes standard Guo-Sengur limitations via:
    1. Brain-restricted localized min-max scaling (eliminating background/skull bias)
    2. Dual-cue Indeterminacy: fuses local intensity deviation |g - gbar|
       with joint edge-texture discontinuity norm(Sobel(g)) * norm(SD_7x7(g)).

    Returns:
      T: Truth map in [0, 1]
      I: Enhanced Indeterminacy map in [0, 1]
      F: Falsehood map (1 - T) in [0, 1]
    """
    import cv2
    g_f = g.astype(np.float64)

    # 1. Local mean gbar
    gbar = ndi.uniform_filter(g_f, size=window_size, mode="reflect")

    # Determine normalization bounds
    if brain_mask is not None and np.sum(brain_mask) > 10:
        valid_pixels = brain_mask > 0
        gbar_min = float(np.min(gbar[valid_pixels]))
        gbar_max = float(np.max(gbar[valid_pixels]))
    else:
        gbar_min = float(np.min(gbar))
        gbar_max = float(np.max(gbar))

    denom_t = gbar_max - gbar_min
    if denom_t < eps:
        t_map = np.zeros_like(gbar)
    else:
        t_map = (gbar - gbar_min) / (denom_t + eps)
    t_map = np.clip(t_map, 0.0, 1.0)

    # 2. Indeterminacy Component A: normalized absolute deviation |g - gbar|
    delta = np.abs(g_f - gbar)
    if brain_mask is not None and np.sum(brain_mask) > 10:
        delta_min = float(np.min(delta[valid_pixels]))
        delta_max = float(np.max(delta[valid_pixels]))
    else:
        delta_min = float(np.min(delta))
        delta_max = float(np.max(delta))

    denom_d = delta_max - delta_min
    delta_norm = (delta - delta_min) / (denom_d + eps) if denom_d >= eps else np.zeros_like(delta)
    delta_norm = np.clip(delta_norm, 0.0, 1.0)

    # 3. Indeterminacy Component B: Joint Gradient & Texture SD Landscape
    # B1: Sobel gradient magnitude
    gx = cv2.Sobel(g_f, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(g_f, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # B2: Local Standard Deviation SD_7x7
    mean_sq = ndi.uniform_filter(g_f**2, size=sd_window, mode="reflect")
    sq_mean = ndi.uniform_filter(g_f, size=sd_window, mode="reflect") ** 2
    local_sd = np.sqrt(np.maximum(0.0, mean_sq - sq_mean))

    # Normalize within brain mask
    if brain_mask is not None and np.sum(brain_mask) > 10:
        g_min, g_max = float(np.min(grad_mag[valid_pixels])), float(np.max(grad_mag[valid_pixels]))
        sd_min, sd_max = float(np.min(local_sd[valid_pixels])), float(np.max(local_sd[valid_pixels]))
    else:
        g_min, g_max = float(np.min(grad_mag)), float(np.max(grad_mag))
        sd_min, sd_max = float(np.min(local_sd)), float(np.max(local_sd))

    g_norm = (grad_mag - g_min) / (g_max - g_min + eps) if (g_max - g_min) >= eps else np.zeros_like(grad_mag)
    sd_norm = (local_sd - sd_min) / (sd_max - sd_min + eps) if (sd_max - sd_min) >= eps else np.zeros_like(local_sd)

    joint_edge = np.clip(g_norm, 0.0, 1.0) * np.clip(sd_norm, 0.0, 1.0)
    if brain_mask is not None and np.sum(brain_mask) > 10:
        je_min = float(np.min(joint_edge[valid_pixels]))
        je_max = float(np.max(joint_edge[valid_pixels]))
    else:
        je_min = float(np.min(joint_edge))
        je_max = float(np.max(joint_edge))

    denom_je = je_max - je_min
    joint_edge_norm = (joint_edge - je_min) / (denom_je + eps) if denom_je >= eps else np.zeros_like(joint_edge)
    joint_edge_norm = np.clip(joint_edge_norm, 0.0, 1.0)

    # 4. Hybrid Indeterminacy fusion
    i_map = w_delta * delta_norm + w_gradient_sd * joint_edge_norm
    i_map = np.clip(i_map, 0.0, 1.0)

    # 5. Falsehood
    f_map = np.clip(1.0 - t_map, 0.0, 1.0)

    return t_map, i_map, f_map
