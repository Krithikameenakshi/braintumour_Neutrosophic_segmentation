"""Full Neutrosophic Brain Tumor Segmentation Pipeline (Phase 3).

Step-by-step modular architecture:
- Step A: Indeterminacy reduction via Directional Alpha-Mean Filtering (DMF)
  NOTE: Exact window shapes are UNVERIFIED in historical literature; implemented
  with a cleanly swappable isotropic mean fallback option.
- Step B: Adaptive Multi-Otsu S-Function contrast stretching
- Step C: Dual-polarity seed extraction
- Step D: Candidate scoring (intensity, indeterminacy, solidity, bilateral asymmetry)
  and rejection threshold for no-tumor cases.
- Step E: Boundary refinement via marker-controlled watershed on I-landscape
- Step F: Morphological post-processing
- Step G: Clinical outputs, boundary uncertainty readout, and confidence scoring
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing, binary_opening, binary_erosion
from skimage.segmentation import watershed
import cv2

from src.ns import neutrosophic_transform, compute_entropy, enhanced_neutrosophic_transform
from src.preprocess import preprocess_image


# =====================================================================
# STEP A: Indeterminacy Reduction with Directional Alpha-Mean (DMF)
# =====================================================================
def directional_alpha_mean_filtering(
    T: np.ndarray,
    I: np.ndarray,
    brain_mask: np.ndarray,
    sigma: float = 0.01,
    window_size: int = 5,
    max_iter: int = 10,
    convergence_tol: float = 0.01,
    fallback_isotropic: bool = False,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Step A: Indeterminacy reduction via Directional Alpha-Mean Filtering (DMF).

    Formula:
      alpha = Otsu(I | M_b)
      T_{k+1} = T_k if I_k < alpha, else DMF(T_k)
      recompute I from T_{k+1}
      stop when |dEn(I)| / En(I) < 0.01 or max_iter iterations.

    NOTE (UNVERIFIED): Exact window shapes for DMF vary across publications.
    Implemented here with horizontal/vertical edge-aligned kernels and an
    explicit, clearly swappable isotropic fallback option.
    """
    valid = brain_mask > 0
    if np.sum(valid) < 50:
        return T.copy(), I.copy(), 1.0 - T, 0

    T_curr = T.copy().astype(np.float64)
    I_curr = I.copy().astype(np.float64)
    en_prev = compute_entropy(I_curr, mask=brain_mask, bins=256)
    iterations_run = 0

    for it in range(max_iter):
        iterations_run = it + 1
        # 1. Indeterminacy threshold alpha
        valid_I = I_curr[valid]
        if np.max(valid_I) - np.min(valid_I) < 1e-4:
            break
        alpha = threshold_otsu(valid_I)

        # 2. DMF update on T
        indeterminate_mask = (I_curr >= alpha) & valid
        if not np.any(indeterminate_mask):
            break

        if fallback_isotropic:
            # Fallback isotropic 3x3 mean
            T_filtered = ndi.uniform_filter(T_curr, size=3, mode="reflect")
            T_next = np.where(indeterminate_mask, T_filtered, T_curr)
        else:
            # Directional alpha-mean filtering
            # Compute gradients Gh and Gv
            gh = cv2.Sobel(T_curr, cv2.CV_64F, 1, 0, ksize=3)
            gv = cv2.Sobel(T_curr, cv2.CV_64F, 0, 1, ksize=3)
            diff_grad = np.abs(np.abs(gh) - np.abs(gv))

            # Isotropic condition: ||Gh| - |Gv|| <= sigma
            iso_mean = ndi.uniform_filter(T_curr, size=3, mode="reflect")
            # Directional along edge:
            # If |Gh| > |Gv|, horizontal gradient is dominant -> edge is vertical -> average along column
            vert_mean = ndi.convolve1d(T_curr, weights=[1/3, 1/3, 1/3], axis=0, mode="reflect")
            # If |Gv| > |Gh|, vertical gradient is dominant -> edge is horizontal -> average along row
            horiz_mean = ndi.convolve1d(T_curr, weights=[1/3, 1/3, 1/3], axis=1, mode="reflect")

            edge_mean = np.where(np.abs(gh) > np.abs(gv), vert_mean, horiz_mean)
            dmf_val = np.where(diff_grad <= sigma, iso_mean, edge_mean)

            T_next = np.where(indeterminate_mask, dmf_val, T_curr)

        T_next = np.clip(T_next, 0.0, 1.0)

        # 3. Recompute I from T_{k+1}
        T_bar = ndi.uniform_filter(T_next, size=window_size, mode="reflect")
        delta = np.abs(T_next - T_bar)
        d_min = float(np.min(delta[valid]))
        d_max = float(np.max(delta[valid]))
        denom_d = d_max - d_min
        if denom_d > eps:
            I_next = np.clip((delta - d_min) / denom_d, 0.0, 1.0)
        else:
            I_next = np.zeros_like(delta)

        # 4. Check convergence on En(I)
        en_curr = compute_entropy(I_next, mask=brain_mask, bins=256)
        if en_prev > eps and abs(en_curr - en_prev) / en_prev < convergence_tol:
            T_curr = T_next
            I_curr = I_next
            break

        T_curr = T_next
        I_curr = I_next
        en_prev = en_curr

    F_final = np.clip(1.0 - T_curr, 0.0, 1.0)
    return T_curr, I_curr, F_final, iterations_run


# =====================================================================
# STEP B: Adaptive S-Function Contrast Enhancement
# =====================================================================
def apply_piecewise_s_function(
    x: np.ndarray,
    PL: float,
    PM: float,
    PR: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Evaluates the standard piecewise cubic S-function:

      S(x) = 0 for x <= PL
      2 * ((x - PL)/(PR - PL))^2 for PL < x <= PM
      1 - 2 * ((x - PR)/(PR - PL))^2 for PM < x < PR
      1 for x >= PR
    """
    denom = max(eps, PR - PL)
    out = np.zeros_like(x, dtype=np.float64)

    r2 = (x > PL) & (x <= PM)
    out[r2] = 2.0 * (((x[r2] - PL) / denom) ** 2)

    r3 = (x > PM) & (x < PR)
    out[r3] = 1.0 - 2.0 * (((x[r3] - PR) / denom) ** 2)

    r4 = x >= PR
    out[r4] = 1.0

    return np.clip(out, 0.0, 1.0)


def adaptive_s_function(
    T_star: np.ndarray,
    brain_mask: np.ndarray,
    multi_otsu_classes: int = 4,
    fallback_classes: int = 3,
    min_class_ratio: float = 0.02,
    pm_candidates: int = 16,
    guard_area_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Step B: Adaptive S-function on T* with Multi-Otsu and Entropy Maximization.

    Returns:
      T_S: Contrast-enhanced Truth map
      F_S: Falsehood map (1 - T_S)
      PL, PM, PR: Fitted parameters
    """
    valid = brain_mask > 0
    t_vals = T_star[valid]
    if len(t_vals) < 50:
        return T_star.copy(), 1.0 - T_star, 0.1, 0.5, 0.9

    # 1. Multi-Otsu thresholding
    try:
        thresholds = threshold_multiotsu(t_vals, classes=multi_otsu_classes)
        # Check if any class has < min_class_ratio (2%) of pixels
        bins = [-np.inf] + list(thresholds) + [np.inf]
        counts, _ = np.histogram(t_vals, bins=bins)
        if np.any(counts / len(t_vals) < min_class_ratio):
            thresholds = threshold_multiotsu(t_vals, classes=fallback_classes)
    except Exception:
        thresholds = threshold_multiotsu(t_vals, classes=fallback_classes)

    if len(thresholds) >= 3:
        PL = float(thresholds[1])  # t2
        PR = float(thresholds[2])  # t3
        t3_val = float(thresholds[2])
    elif len(thresholds) == 2:
        PL = float(thresholds[0])
        PR = float(thresholds[1])
        t3_val = float(thresholds[1])
    else:
        PL = 0.2
        PR = 0.8
        t3_val = 0.8

    # 2. Search for PM maximizing En(S(T*; PL, PM, PR))
    cand_pms = np.linspace(PL + 1e-4, PR - 1e-4, pm_candidates)
    best_pm = cand_pms[len(cand_pms) // 2]
    best_entropy = -1.0

    for pm in cand_pms:
        s_cand = apply_piecewise_s_function(t_vals, PL, pm, PR)
        en = compute_entropy(s_cand, bins=256)
        if en > best_entropy:
            best_entropy = en
            best_pm = pm

    PM = best_pm
    T_S = apply_piecewise_s_function(T_star, PL, PM, PR)

    # 3. Guard: If object seed covers >35% of brain, set PL = t3 and recompute
    seed_area = np.sum((T_S >= 0.9) & valid)
    brain_area = np.sum(valid)
    if brain_area > 0 and (seed_area / brain_area) > guard_area_ratio:
        PL = t3_val
        PR = min(1.0, t3_val + 0.15)
        PM = 0.5 * (PL + PR)
        T_S = apply_piecewise_s_function(T_star, PL, PM, PR)

    F_S = np.clip(1.0 - T_S, 0.0, 1.0)
    return T_S, F_S, PL, PM, PR


# =====================================================================
# STEP C: Seed Extraction & Dual Polarity
# =====================================================================
def extract_seeds(
    T_S: np.ndarray,
    I_star: np.ndarray,
    brain_mask: np.ndarray,
    ts_high: float = 0.9,
    ts_low: float = 0.1,
    bg_erosion_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step C: Seed extraction.

    O = (T_S >= ts_high) & (I* < tau_I) & M_b
    B = erode((T_S <= ts_low) & (I* < tau_I), 2px) | ~M_b
    E = I* >= tau_I
    """
    valid = brain_mask > 0
    valid_I = I_star[valid]
    if len(valid_I) < 50:
        return np.zeros_like(T_S, dtype=bool), np.zeros_like(T_S, dtype=bool), np.zeros_like(T_S, dtype=bool)

    tau_I = threshold_otsu(valid_I)

    # Object seeds O
    O = (T_S >= ts_high) & (I_star < tau_I) & valid

    # Background seeds B
    low_indeterminate_bg = (T_S <= ts_low) & (I_star < tau_I) & valid
    if bg_erosion_px > 0:
        eroded_bg = binary_erosion(low_indeterminate_bg, disk(bg_erosion_px))
    else:
        eroded_bg = low_indeterminate_bg
    B = eroded_bg | (~valid)

    # Indeterminate / Edge region E
    E = (I_star >= tau_I) & valid

    return O, B, E


# =====================================================================
# STEP D: Candidate Scoring & Rejection
# =====================================================================
def score_and_select_candidates(
    O: np.ndarray,
    T_S: np.ndarray,
    I_star: np.ndarray,
    g_norm: np.ndarray,
    brain_mask: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """Step D: Candidate Component Scoring, Bilateral Asymmetry, and Rejection.

    Filters candidates by size [0.05%, 35%].
    Computes Score(C) = 0.35*mean(T_S) + 0.25*(1 - mean(I*)) + 0.20*Solidity + 0.20*g_sym*Asym.
    If max score < tau_rej, returns empty mask (no tumor).
    """
    d_cfg = cfg.get("pipeline", {}).get("step_d_candidate_scoring", {})
    min_area_ratio = d_cfg.get("min_area_ratio", 0.0005)
    max_area_ratio = d_cfg.get("max_area_ratio", 0.35)
    w_t = d_cfg.get("w_t", 0.35)
    w_i = d_cfg.get("w_i", 0.25)
    w_solidity = d_cfg.get("w_solidity", 0.20)
    w_sym = d_cfg.get("w_sym", 0.20)
    sym_corr_thresh = d_cfg.get("sym_corr_thresh", 0.85)
    tau_rej = d_cfg.get("tau_rej", 0.40)
    top_cand_ratio = d_cfg.get("top_candidate_ratio", 0.80)

    h, w = g_norm.shape
    total_img_area = h * w
    min_px = int(min_area_ratio * total_img_area)
    max_px = int(max_area_ratio * total_img_area)

    labeled_O, num_features = label(O > 0, return_num=True, connectivity=2)
    if num_features == 0:
        return np.zeros_like(O, dtype=np.uint8), 0.0, []

    # Hemispheric Symmetry Evaluation:
    # Check left-right (horizontal mirror) and up-down symmetry
    mirror_h = np.fliplr(g_norm)
    valid_b = brain_mask > 0
    if np.sum(valid_b) > 100:
        g_vals = g_norm[valid_b]
        m_vals = mirror_h[valid_b]
        # Correlation
        std_g = np.std(g_vals)
        std_m = np.std(m_vals)
        if std_g > 1e-4 and std_m > 1e-4:
            corr_val = float(np.corrcoef(g_vals, m_vals)[0, 1])
        else:
            corr_val = 0.0
    else:
        corr_val = 0.0

    g_sym = 1.0 if corr_val >= sym_corr_thresh else 0.0

    # Renormalize weights if g_sym == 0
    if g_sym == 0.0:
        base_sum = w_t + w_i + w_solidity
        w_t_eff = w_t / base_sum
        w_i_eff = w_i / base_sum
        w_sol_eff = w_solidity / base_sum
        w_sym_eff = 0.0
    else:
        w_t_eff = w_t
        w_i_eff = w_i
        w_sol_eff = w_solidity
        w_sym_eff = w_sym

    # Brain absolute difference map
    abs_diff = np.abs(g_norm - mirror_h)
    mean_brain_diff = float(np.mean(abs_diff[valid_b])) if np.sum(valid_b) > 0 else 1.0
    mean_brain_diff = max(1e-4, mean_brain_diff)

    props = regionprops(labeled_O)
    candidate_records = []

    for prop in props:
        area = prop.area
        if area < min_px or area > max_px:
            continue

        comp_mask = labeled_O == prop.label
        mean_t = float(np.mean(T_S[comp_mask]))
        mean_i = float(np.mean(I_star[comp_mask]))
        solidity = float(prop.solidity)

        # Asymmetry
        mean_comp_diff = float(np.mean(abs_diff[comp_mask]))
        asym = mean_comp_diff / mean_brain_diff
        asym_clipped = min(2.0, asym)  # prevent extreme outliers

        score = (
            w_t_eff * mean_t
            + w_i_eff * (1.0 - mean_i)
            + w_sol_eff * solidity
            + w_sym_eff * g_sym * asym_clipped
        )

        candidate_records.append({
            "label": prop.label,
            "area": area,
            "mean_T": mean_t,
            "mean_I": mean_i,
            "solidity": solidity,
            "asym": asym,
            "score": score,
            "bbox": prop.bbox,
        })

    if not candidate_records:
        return np.zeros_like(O, dtype=np.uint8), 0.0, []

    # Sort candidates by score descending
    candidate_records.sort(key=lambda c: c["score"], reverse=True)
    best_score = candidate_records[0]["score"]

    # Rejection Threshold check for healthy/no-tumor slices
    if best_score < tau_rej:
        return np.zeros_like(O, dtype=np.uint8), best_score, candidate_records

    # Keep top candidate and any other candidate scoring >= top_cand_ratio * max
    retained_mask = np.zeros_like(O, dtype=np.uint8)
    score_cutoff = top_cand_ratio * best_score

    for cand in candidate_records:
        if cand["score"] >= score_cutoff:
            retained_mask[labeled_O == cand["label"]] = 1

    return retained_mask, best_score, candidate_records


# =====================================================================
# STEP E: Boundary Refinement via Marker-Controlled Watershed
# =====================================================================
def marker_controlled_watershed(
    C_star: np.ndarray,
    B: np.ndarray,
    O: np.ndarray,
    I_star: np.ndarray,
    g_norm: np.ndarray,
    brain_mask: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """Step E: Boundary refinement by marker-controlled watershed on I-landscape.

    L = 0.5 * (I* + Ghat), Ghat = smoothed norm(SD_7x7) * norm(Sobel)
    Markers: label 1 = erode(C*, 1px); label 2 = B | (O minus C*)
    R = basin of label 1.
    """
    if not np.any(C_star):
        return np.zeros_like(C_star, dtype=np.uint8)

    ws_cfg = cfg.get("pipeline", {}).get("step_e_watershed", {})
    sd_window = ws_cfg.get("sd_window", 7)
    sobel_ksize = ws_cfg.get("sobel_ksize", 3)
    w_i_land = ws_cfg.get("i_landscape_weight", 0.5)
    w_ghat = ws_cfg.get("ghat_landscape_weight", 0.5)
    marker_erosion_px = ws_cfg.get("marker_erosion_px", 1)
    eps = 1e-8

    valid = brain_mask > 0
    g_f = g_norm.astype(np.float64)

    # 1. Compute Ghat = norm(SD_7x7) * norm(Sobel)
    gx = cv2.Sobel(g_f, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(g_f, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_mag = np.sqrt(gx**2 + gy**2)

    mean_sq = ndi.uniform_filter(g_f**2, size=sd_window, mode="reflect")
    sq_mean = ndi.uniform_filter(g_f, size=sd_window, mode="reflect") ** 2
    local_sd = np.sqrt(np.maximum(0.0, mean_sq - sq_mean))

    g_min, g_max = float(np.min(grad_mag[valid])), float(np.max(grad_mag[valid]))
    sd_min, sd_max = float(np.min(local_sd[valid])), float(np.max(local_sd[valid]))

    norm_sobel = (grad_mag - g_min) / (g_max - g_min + eps) if (g_max - g_min) >= eps else np.zeros_like(grad_mag)
    norm_sd = (local_sd - sd_min) / (sd_max - sd_min + eps) if (sd_max - sd_min) >= eps else np.zeros_like(local_sd)

    ghat_raw = np.clip(norm_sobel, 0.0, 1.0) * np.clip(norm_sd, 0.0, 1.0)
    ghat_smooth = gaussian(ghat_raw, sigma=1.0)

    # 2. Combined topographic landscape L
    L = w_i_land * I_star + w_ghat * ghat_smooth

    # 3. Markers: label 1 = erode(C*, 1px), label 2 = B | (O minus C*)
    if marker_erosion_px > 0:
        obj_marker = binary_erosion(C_star > 0, disk(marker_erosion_px))
        if not np.any(obj_marker):
            obj_marker = C_star > 0
    else:
        obj_marker = C_star > 0

    other_O = (O > 0) & (~(C_star > 0))
    bg_marker = (B > 0) | other_O

    markers = np.zeros_like(C_star, dtype=np.int32)
    markers[obj_marker] = 1
    markers[bg_marker] = 2

    # Watershed constrained to brain mask
    ws_labels = watershed(L, markers=markers, mask=valid)
    basin_1 = (ws_labels == 1).astype(np.uint8)

    return basin_1


# =====================================================================
# STEP F: Morphological Post-Processing
# =====================================================================
def postprocess_mask(
    mask: np.ndarray,
    cfg: Dict,
) -> np.ndarray:
    """Step F: Morphological post-processing.

    closing(disk 2) -> fill holes -> opening(disk 1) -> keep largest component
    -> Gaussian smooth (sigma=1), threshold 0.5.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    p_cfg = cfg.get("pipeline", {}).get("step_f_postprocess", {})
    close_r = p_cfg.get("closing_radius", 2)
    open_r = p_cfg.get("opening_radius", 1)
    g_sigma = p_cfg.get("gaussian_sigma", 1.0)
    bin_thresh = p_cfg.get("binary_threshold", 0.5)

    # 1. Closing (disk 2)
    m = binary_closing(mask > 0, disk(close_r))

    # 2. Fill holes
    m = ndi.binary_fill_holes(m)

    # 3. Opening (disk 1)
    m = binary_opening(m, disk(open_r))

    # 4. Keep largest component
    labeled, num_features = label(m, return_num=True, connectivity=2)
    if num_features > 1:
        counts = np.bincount(labeled.flat)
        counts[0] = 0
        largest_label = counts.argmax()
        m = labeled == largest_label

    # 5. Gaussian smooth (sigma=1.0), threshold 0.5
    m_smooth = gaussian(m.astype(np.float64), sigma=g_sigma)
    m_final = (m_smooth >= bin_thresh).astype(np.uint8)

    return m_final


# =====================================================================
# STEP G: Clinical Readouts & Visualization Artifacts
# =====================================================================
def generate_clinical_readouts(
    img: np.ndarray,
    final_mask: np.ndarray,
    T_star: np.ndarray,
    I_star: np.ndarray,
    T_S: np.ndarray,
    brain_mask: np.ndarray,
    cfg: Dict,
) -> Dict:
    """Step G: Generates overlays, boundary uncertainty score, and region

    confidence.

    - Boundary uncertainty: mean I* in a 3px band around the contour
    - Region confidence: mean T_S inside the mask
    - Visual overlay: red contour (2px) and translucent highlight (alpha 0.35)
    """
    g_cfg = cfg.get("pipeline", {}).get("step_g_output", {})
    band_px = g_cfg.get("uncertainty_band_px", 3)
    alpha = g_cfg.get("overlay_alpha", 0.35)
    thick = g_cfg.get("contour_thickness", 2)

    has_tumor = bool(np.any(final_mask > 0))

    if has_tumor:
        # 1. Region confidence: mean T_S inside mask
        region_confidence = float(np.mean(T_S[final_mask > 0]))

        # 2. Boundary uncertainty: mean I* in a 3px band around contour
        dilated = ndi.binary_dilation(final_mask > 0, iterations=band_px)
        eroded = ndi.binary_erosion(final_mask > 0, iterations=band_px)
        boundary_band = dilated & (~eroded) & (brain_mask > 0)

        if np.any(boundary_band):
            boundary_uncertainty = float(np.mean(I_star[boundary_band]))
        else:
            boundary_uncertainty = 0.0

        # 3. Tumor size / area metrics
        tumor_px = int(np.sum(final_mask))
        brain_px = max(1, int(np.sum(brain_mask)))
        tumor_pct_brain = 100.0 * tumor_px / brain_px
    else:
        region_confidence = 0.0
        boundary_uncertainty = 0.0
        tumor_px = 0
        tumor_pct_brain = 0.0

    # 4. Render RGB Overlay
    img_uint8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
    rgb_overlay = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    if has_tumor:
        # Translucent highlight (alpha 0.35, tinted red)
        red_layer = np.zeros_like(rgb_overlay)
        red_layer[:, :, 0] = 255  # Red channel
        mask_3c = np.stack([final_mask] * 3, axis=-1)
        rgb_overlay = np.where(
            mask_3c == 1,
            cv2.addWeighted(rgb_overlay, 1.0 - alpha, red_layer, alpha, 0),
            rgb_overlay
        )

        # 2px solid red contour outline
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_overlay, cnts, -1, (255, 0, 0), thick)

    return {
        "has_tumor": has_tumor,
        "region_confidence": round(region_confidence, 4),
        "boundary_uncertainty": round(boundary_uncertainty, 4),
        "tumor_area_pixels": tumor_px,
        "tumor_pct_brain": round(tumor_pct_brain, 2),
        "overlay_rgb": rgb_overlay,
    }


# =====================================================================
# COMPLETE PIPELINE INTEGRATION
# =====================================================================
def run_neutrosophic_segmentation(
    img: np.ndarray,
    cfg: Dict,
) -> Dict:
    """Runs complete Phase 3 Neutrosophic Pipeline with Dual-Polarity on input

    slice.
    """
    # Preprocessing
    g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)

    # Dual-polarity check: run on g and on (1 - g)
    polarities = [("positive", g_denoised), ("negative", 1.0 - g_denoised)]
    best_candidate_score = -1.0
    best_result = None

    for pol_name, g_input in polarities:
        # Neutrosophic Transform
        w_size = cfg["neutrosophic"]["window_size"]
        T_init, I_init, F_init = neutrosophic_transform(g_input, window_size=w_size)

        # Step A: Indeterminacy reduction via DMF
        step_a_cfg = cfg.get("pipeline", {}).get("step_a_indeterminacy", {})
        T_star, I_star, F_star, iters = directional_alpha_mean_filtering(
            T_init, I_init, brain_mask,
            sigma=step_a_cfg.get("sigma", 0.01),
            window_size=step_a_cfg.get("dmf_window", 3),
            max_iter=step_a_cfg.get("max_iter", 10),
            convergence_tol=step_a_cfg.get("convergence_tol", 0.01),
            fallback_isotropic=step_a_cfg.get("fallback_isotropic", False),
        )

        # Step B: Adaptive Multi-Otsu S-Function
        step_b_cfg = cfg.get("pipeline", {}).get("step_b_adaptive_s", {})
        T_S, F_S, PL, PM, PR = adaptive_s_function(
            T_star, brain_mask,
            multi_otsu_classes=step_b_cfg.get("multi_otsu_classes", 4),
            fallback_classes=step_b_cfg.get("multi_otsu_fallback_classes", 3),
            min_class_ratio=step_b_cfg.get("min_class_ratio", 0.02),
            pm_candidates=step_b_cfg.get("pm_candidates", 16),
            guard_area_ratio=step_b_cfg.get("guard_brain_area_ratio", 0.35),
        )

        # Step C: Seed extraction
        step_c_cfg = cfg.get("pipeline", {}).get("step_c_seeds", {})
        O, B, E = extract_seeds(
            T_S, I_star, brain_mask,
            ts_high=step_c_cfg.get("ts_high", 0.9),
            ts_low=step_c_cfg.get("ts_low", 0.1),
            bg_erosion_px=step_c_cfg.get("bg_erosion_px", 2),
        )

        # Step D: Candidate scoring & rejection
        C_star, score, cand_records = score_and_select_candidates(
            O, T_S, I_star, g_norm, brain_mask, cfg
        )

        if score > best_candidate_score:
            best_candidate_score = score
            best_result = {
                "polarity": pol_name,
                "score": score,
                "T_star": T_star,
                "I_star": I_star,
                "F_star": F_star,
                "T_S": T_S,
                "F_S": F_S,
                "O": O,
                "B": B,
                "E": E,
                "C_star": C_star,
                "cand_records": cand_records,
                "dmf_iters": iters,
            }

    # If no candidates or score below rejection threshold: output EMPTY mask
    if best_result is None or not np.any(best_result["C_star"]):
        final_mask = np.zeros_like(img, dtype=np.uint8)
        readouts = generate_clinical_readouts(
            img, final_mask,
            best_result["T_star"] if best_result else np.zeros_like(img, dtype=float),
            best_result["I_star"] if best_result else np.zeros_like(img, dtype=float),
            best_result["T_S"] if best_result else np.zeros_like(img, dtype=float),
            brain_mask, cfg
        )
        return {
            "mask": final_mask,
            "readouts": readouts,
            "T_star": best_result["T_star"] if best_result else np.zeros_like(img, dtype=float),
            "I_star": best_result["I_star"] if best_result else np.zeros_like(img, dtype=float),
            "F_star": best_result["F_star"] if best_result else np.zeros_like(img, dtype=float),
            "T_S": best_result["T_S"] if best_result else np.zeros_like(img, dtype=float),
            "head_mask": head_mask,
            "brain_mask": brain_mask,
            "candidate_score": best_candidate_score,
            "polarity": best_result["polarity"] if best_result else "positive",
        }

    # Step E: Boundary refinement by marker-controlled watershed
    R = marker_controlled_watershed(
        best_result["C_star"],
        best_result["B"],
        best_result["O"],
        best_result["I_star"],
        g_norm,
        brain_mask,
        cfg
    )

    # Step F: Morphological post-processing
    final_mask = postprocess_mask(R, cfg)

    # Step G: Clinical readouts and uncertainty metrics
    readouts = generate_clinical_readouts(
        img, final_mask,
        best_result["T_star"],
        best_result["I_star"],
        best_result["T_S"],
        brain_mask,
        cfg
    )

    return {
        "mask": final_mask,
        "readouts": readouts,
        "T_star": best_result["T_star"],
        "I_star": best_result["I_star"],
        "F_star": best_result["F_star"],
        "T_S": best_result["T_S"],
        "head_mask": head_mask,
        "brain_mask": brain_mask,
        "candidate_score": best_candidate_score,
        "polarity": best_result["polarity"],
    }
