"""Phase 4 - Step 2: Systematic Ablation Study.

Evaluates 6 systematic ablations on the 100-image tuning subset:
(a) Remove Directional Alpha-Mean (use plain isotropic mean fallback)
(b) Remove Adaptive S-function (use fixed thresholds PL=0.2, PR=0.8)
(c) Remove Candidate Scoring (keep simply the largest connected component)
(d) Remove Bilateral Symmetry term (w_sym = 0, no mirror correlation)
(e) Remove I-landscape (use plain Sobel gradient landscape Ghat for watershed)
(f) Replace Watershed with NS + Level Set baseline (morphological active contours on T*)

Logs results to results/ablations_tuning.csv and outputs/phase4_ablations_summary.csv.
"""

import sys
from pathlib import Path
import copy
import yaml
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset
from src.preprocess import preprocess_image
from src.ns import neutrosophic_transform
from src.metrics import compute_metrics
from src.pipeline import (
    directional_alpha_mean_filtering,
    adaptive_s_function,
    extract_seeds,
    score_and_select_candidates,
    marker_controlled_watershed,
    postprocess_mask,
    apply_piecewise_s_function,
)


def run_ablations():
    with open("configs/final.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    tuning_csv_path = Path(cfg["paths"]["tuning_subset_path"])
    tuning_df = pd.read_csv(tuning_csv_path)
    print(f"Loaded tuning subset ({len(tuning_df)} images) for ablation study.")

    # Cache dataset in memory for speed
    data = []
    for _, row in tuning_df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask_p = row["mask_path"]
        if mask_p is not None and pd.notna(mask_p) and Path(str(mask_p)).exists():
            gt_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            gt = (gt_raw > 127).astype(np.uint8)
        else:
            gt = np.zeros_like(img, dtype=np.uint8)
        data.append((img, gt, row["tumor_class"], row["plane"], row["filename"]))

    ablation_names = [
        "Full Proposed Pipeline",
        "Ablation (a): No Directional Alpha-Mean",
        "Ablation (b): No Adaptive S-function",
        "Ablation (c): No Candidate Scoring",
        "Ablation (d): No Symmetry Term",
        "Ablation (e): No I-Landscape",
        "Ablation (f): NS + Level Set",
    ]

    ablation_records = []

    print("\nRunning Systematic Ablation Experiments...")

    for name in ablation_names:
        print(f"Evaluating: {name}...")
        dice_list = []
        iou_list = []
        sens_list = []
        prec_list = []
        fnr_list = []
        hd95_list = []
        false_alarms = 0
        total_negs = 0

        for img, gt, t_class, plane, fn in data:
            # 1. Preprocess
            g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)

            # 2. Neutrosophic Transform
            T_init, I_init, F_init = neutrosophic_transform(g_denoised, window_size=cfg["neutrosophic"]["window_size"])

            # (a) Directional alpha-mean vs Isotropic Fallback
            fallback_iso = (name == "Ablation (a): No Directional Alpha-Mean")
            T_star, I_star, F_star, _ = directional_alpha_mean_filtering(
                T_init, I_init, brain_mask,
                sigma=cfg["pipeline"]["step_a_indeterminacy"]["sigma"],
                window_size=cfg["pipeline"]["step_a_indeterminacy"]["dmf_window"],
                max_iter=cfg["pipeline"]["step_a_indeterminacy"]["max_iter"],
                fallback_isotropic=fallback_iso,
            )

            # (b) Adaptive S-function vs Fixed S-function
            if name == "Ablation (b): No Adaptive S-function":
                T_S = apply_piecewise_s_function(T_star, PL=0.2, PM=0.5, PR=0.8)
                F_S = 1.0 - T_S
            else:
                T_S, F_S, _, _, _ = adaptive_s_function(
                    T_star, brain_mask,
                    multi_otsu_classes=cfg["pipeline"]["step_b_adaptive_s"]["multi_otsu_classes"],
                    fallback_classes=cfg["pipeline"]["step_b_adaptive_s"]["multi_otsu_fallback_classes"],
                )

            # (c) Seeds
            O, B, E = extract_seeds(T_S, I_star, brain_mask)

            # Candidate selection
            if name == "Ablation (c): No Candidate Scoring":
                # Largest connected component of O without multi-factor scoring
                from skimage.measure import label as sk_label
                labeled, num = sk_label(O > 0, return_num=True, connectivity=2)
                if num > 0:
                    counts = np.bincount(labeled.flat)
                    counts[0] = 0
                    C_star = (labeled == counts.argmax()).astype(np.uint8)
                else:
                    C_star = np.zeros_like(O, dtype=np.uint8)
            else:
                curr_cfg = copy.deepcopy(cfg)
                if name == "Ablation (d): No Symmetry Term":
                    curr_cfg["pipeline"]["step_d_candidate_scoring"]["w_sym"] = 0.0
                    curr_cfg["pipeline"]["step_d_candidate_scoring"]["w_t"] = 0.45
                    curr_cfg["pipeline"]["step_d_candidate_scoring"]["w_i"] = 0.30
                    curr_cfg["pipeline"]["step_d_candidate_scoring"]["w_solidity"] = 0.25

                C_star, score, _ = score_and_select_candidates(O, T_S, I_star, g_norm, brain_mask, curr_cfg)

            if not np.any(C_star):
                final_mask = np.zeros_like(img, dtype=np.uint8)
            else:
                if name == "Ablation (f): NS + Level Set":
                    # Morphological active contour / level set baseline on T*
                    g_inv = inverse_gaussian_gradient(T_star, alpha=100.0, sigma=1.0)
                    ls_init = C_star.astype(bool)
                    ls_mask = morphological_geodesic_active_contour(g_inv, num_iter=15, init_level_set=ls_init)
                    final_mask = postprocess_mask((ls_mask & (brain_mask > 0)).astype(np.uint8), cfg)
                else:
                    # Watershed
                    ws_cfg = copy.deepcopy(cfg)
                    if name == "Ablation (e): No I-Landscape":
                        ws_cfg["pipeline"]["step_e_watershed"]["i_landscape_weight"] = 0.0
                        ws_cfg["pipeline"]["step_e_watershed"]["ghat_landscape_weight"] = 1.0

                    R = marker_controlled_watershed(C_star, B, O, I_star, g_norm, brain_mask, ws_cfg)
                    final_mask = postprocess_mask(R, cfg)

            mets = compute_metrics(final_mask, gt)
            if mets["is_empty_gt"]:
                total_negs += 1
                if mets["false_alarm"]:
                    false_alarms += 1
            else:
                dice_list.append(mets["dice"])
                iou_list.append(mets["iou"])
                sens_list.append(mets["sensitivity"])
                prec_list.append(mets["precision"])
                fnr_list.append(mets["fnr"])
                if not np.isnan(mets["hd95"]):
                    hd95_list.append(mets["hd95"])

        record = {
            "Configuration": name,
            "Dice_mean": np.mean(dice_list),
            "Dice_std": np.std(dice_list),
            "IoU_mean": np.mean(iou_list),
            "Sens_mean": np.mean(sens_list),
            "Prec_mean": np.mean(prec_list),
            "FNR_mean": np.mean(fnr_list),
            "HD95_mean": np.mean(hd95_list) if len(hd95_list) > 0 else np.nan,
            "FPR_pct": 100.0 * false_alarms / max(1, total_negs),
            "False_Alarms": false_alarms,
        }
        ablation_records.append(record)

    df_abl = pd.DataFrame(ablation_records)
    csv_path = results_dir / "ablations_tuning.csv"
    summary_path = outputs_dir / "phase4_ablations_summary.csv"
    df_abl.to_csv(csv_path, index=False)
    df_abl.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("SYSTEMATIC ABLATION STUDY RESULTS (TUNING SUBSET N=100)")
    print("=" * 80)
    print(df_abl[["Configuration", "Dice_mean", "IoU_mean", "Sens_mean", "Prec_mean", "HD95_mean", "FPR_pct"]].to_string(index=False))
    print("=" * 80)
    print(f"Saved ablations summary to: {summary_path}")


if __name__ == "__main__":
    run_ablations()
