"""Preprocessing Module for Brain MRI Segmentation.

Implements:
1. Percentile-based intensity normalization: g = clip((I - P1)/(P99 - P1), 0, 1)
2. Head mask extraction via Otsu thresholding + largest connected component + hole filling
3. Brain parenchyma mask (M_b) via distance-to-background transform (erosion of scalp fat)
4. Anisotropic diffusion (Perona-Malik) preceded by 3x3 median filtering
"""

from typing import Dict, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label


def normalize_intensity(
    img: np.ndarray,
    p1: float = 1.0,
    p99: float = 99.0,
) -> np.ndarray:
    """Normalizes image intensities between 1st and 99th percentiles.

    Formula: g = clip((I - P1) / (P99 - P1), 0.0, 1.0)
    """
    img_f = img.astype(np.float64)
    # Consider only non-zero pixels if possible, or entire slice
    non_zero = img_f[img_f > 0]
    if len(non_zero) > 100:
        val_p1 = np.percentile(non_zero, p1)
        val_p99 = np.percentile(non_zero, p99)
    else:
        val_p1 = np.percentile(img_f, p1)
        val_p99 = np.percentile(img_f, p99)

    denom = val_p99 - val_p1
    if denom <= 1e-8:
        return np.zeros_like(img_f, dtype=np.float64)

    g = (img_f - val_p1) / denom
    return np.clip(g, 0.0, 1.0)


def extract_head_mask(g: np.ndarray) -> np.ndarray:
    """Extracts binary head mask H.

    Formula: H = fill_holes(largest_connected_component(g > Otsu(g)))
    """
    # Compute Otsu on non-zero pixels to avoid background black dominance
    nz = g[g > 0]
    if len(nz) < 50:
        return np.zeros_like(g, dtype=bool)

    thresh = threshold_otsu(nz)
    binary = g > thresh

    # Connected components
    labeled, num_features = label(binary, return_num=True, connectivity=2)
    if num_features == 0:
        return np.zeros_like(g, dtype=bool)

    # Largest connected component
    counts = np.bincount(labeled.flat)
    counts[0] = 0  # ignore background
    largest_label = counts.argmax()
    lcc = labeled == largest_label

    # Fill holes
    head_mask = ndi.binary_fill_holes(lcc)
    return head_mask


def extract_brain_mask(
    head_mask: np.ndarray,
    r_s_ratio: float = 0.05,
) -> np.ndarray:
    """Extracts intracranial brain parenchyma mask M_b by eroding outer scalp

    margin.

    Formula:
      r_s = r_s_ratio * width
      M_b = {x in H : distance_to_background(x) > r_s}
    """
    if not np.any(head_mask):
        return np.zeros_like(head_mask, dtype=bool)

    height, width = head_mask.shape
    r_s = float(r_s_ratio * width)

    # Euclidean distance transform from background (where head_mask == False)
    dist = ndi.distance_transform_edt(head_mask)
    brain_mask = dist > r_s

    # Keep largest connected component inside M_b to prevent disconnected artifacts
    labeled, num_features = label(brain_mask, return_num=True, connectivity=2)
    if num_features > 0:
        counts = np.bincount(labeled.flat)
        counts[0] = 0
        if len(counts) > 1 and counts.max() > 0:
            brain_mask = labeled == counts.argmax()

    return brain_mask


def perona_malik_diffusion(
    g: np.ndarray,
    kappa: float = 0.05,
    lambda_val: float = 0.15,
    n_iter: int = 10,
    conductance_option: int = 1,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion filter with 3x3 median prefiltering.

    Formula:
      g = median3x3(g)
      g^{t+1} = g^t + lambda * (cN*dN + cS*dS + cE*dE + cW*dW)
      where c(grad) = exp(-(grad/kappa)^2) [option 1]
           or c(grad) = 1 / (1 + (grad/kappa)^2) [option 2]
    """
    # 3x3 median filter pre-smoothing
    u = ndi.median_filter(g.astype(np.float64), size=3)

    for _ in range(n_iter):
        # 4-neighbor spatial differences
        dn = np.zeros_like(u)
        ds = np.zeros_like(u)
        de = np.zeros_like(u)
        dw = np.zeros_like(u)

        dn[:-1, :] = u[1:, :] - u[:-1, :]  # North (upward gradient)
        ds[1:, :] = u[:-1, :] - u[1:, :]  # South (downward gradient)
        de[:, :-1] = u[:, 1:] - u[:, :-1]  # East (rightward gradient)
        dw[:, 1:] = u[:, :-1] - u[:, 1:]  # West (leftward gradient)

        # Conductance coefficients
        if conductance_option == 1:
            cn = np.exp(-((dn / kappa) ** 2))
            cs = np.exp(-((ds / kappa) ** 2))
            ce = np.exp(-((de / kappa) ** 2))
            cw = np.exp(-((dw / kappa) ** 2))
        else:
            cn = 1.0 / (1.0 + (dn / kappa) ** 2)
            cs = 1.0 / (1.0 + (ds / kappa) ** 2)
            ce = 1.0 / (1.0 + (de / kappa) ** 2)
            cw = 1.0 / (1.0 + (dw / kappa) ** 2)

        # Discrete diffusion update step
        u += lambda_val * (cn * dn + cs * ds + ce * de + cw * dw)

    return np.clip(u, 0.0, 1.0)


def preprocess_image(
    img: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs complete preprocessing pipeline according to config parameters.

    Returns:
      g_norm: normalized original image [0, 1]
      head_mask: binary head mask (H)
      brain_mask: binary brain parenchyma mask (M_b)
      g_denoised: Perona-Malik smoothed image [0, 1]
    """
    p_cfg = cfg.get("preprocess", {})
    p1 = p_cfg.get("p1", 1.0)
    p99 = p_cfg.get("p99", 99.0)
    r_s_ratio = p_cfg.get("skull_strip", {}).get("r_s_ratio", 0.05)

    pm_cfg = p_cfg.get("perona_malik", {})
    kappa = pm_cfg.get("kappa", 0.05)
    lambda_val = pm_cfg.get("lambda_val", 0.15)
    n_iter = pm_cfg.get("n_iter", 10)
    opt = pm_cfg.get("conductance_function", 1)

    # 1. Percentile normalization
    g_norm = normalize_intensity(img, p1=p1, p99=p99)

    # 2. Head mask H
    head_mask = extract_head_mask(g_norm)

    # 3. Brain mask M_b
    brain_mask = extract_brain_mask(head_mask, r_s_ratio=r_s_ratio)

    # 4. Anisotropic diffusion (Perona-Malik with median 3x3)
    g_denoised = perona_malik_diffusion(
        g_norm,
        kappa=kappa,
        lambda_val=lambda_val,
        n_iter=n_iter,
        conductance_option=opt,
    )

    return g_norm, head_mask, brain_mask, g_denoised
