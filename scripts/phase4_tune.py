"""Phase 4 - Step 1: Hyperparameter Tuning on 100-Image Tuning Subset.

Performs a grid search over key parameters:
1. r_s_ratio in [0.04, 0.05, 0.06] (brain mask margin)
2. tau_rej in [0.50, 0.65, 0.75, 0.82] (tumor rejection threshold for no-tumor slices)
3. polarity_guard in [True, False] (prefer positive contrast enhancement unless negative score significantly dominates)

Logs all trial metrics to results/tuning_log.csv.
Selects the Pareto-optimal configuration (maximizing Dice on tumor slices while minimizing FPR on healthy slices).
Freezes the optimal parameters to configs/final.yaml.
"""

import sys
from pathlib import Path
import copy
import yaml
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset
from src.pipeline import run_neutrosophic_segmentation
from src.metrics import compute_metrics


def run_tuning():
    with open("configs/default.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    results_dir = Path(base_cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = Path("configs")

    tuning_csv_path = Path(base_cfg["paths"]["tuning_subset_path"])
    tuning_df = pd.read_csv(tuning_csv_path)
    print(f"Loaded tuning subset ({len(tuning_df)} images).")

    # Grid search space
    r_s_candidates = [0.04, 0.05, 0.06]
    tau_rej_candidates = [0.50, 0.65, 0.75, 0.82]

    # Pre-load images and GT masks to speed up grid evaluation
    cached_data = []
    print("Pre-loading tuning subset images and masks...")
    for _, row in tuning_df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        mask_p = row["mask_path"]
        if mask_p is not None and pd.notna(mask_p) and Path(str(mask_p)).exists():
            gt_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            gt = (gt_raw > 127).astype(np.uint8)
        else:
            gt = np.zeros_like(img, dtype=np.uint8)
        cached_data.append((img, gt, row["tumor_class"], row["plane"], row["filename"]))

    tuning_logs = []
    best_composite_score = -1.0
    best_cfg = None
    trial_idx = 0

    total_trials = len(r_s_candidates) * len(tau_rej_candidates)
    print(f"\nLaunching Parameter Grid Search across {total_trials} hyperparameter configurations...")

    for r_s in r_s_candidates:
        for tau_rej in tau_rej_candidates:
            trial_idx += 1
            curr_cfg = copy.deepcopy(base_cfg)
            curr_cfg["preprocess"]["skull_strip"]["r_s_ratio"] = r_s
            curr_cfg["pipeline"]["step_d_candidate_scoring"]["tau_rej"] = tau_rej

            dice_scores = []
            iou_scores = []
            sens_scores = []
            prec_scores = []
            false_alarms = 0
            neg_count = 0

            for img, gt, t_class, plane, fn in cached_data:
                res = run_neutrosophic_segmentation(img, curr_cfg)
                mets = compute_metrics(res["mask"], gt)

                if mets["is_empty_gt"]:
                    neg_count += 1
                    if mets["false_alarm"]:
                        false_alarms += 1
                else:
                    dice_scores.append(mets["dice"])
                    iou_scores.append(mets["iou"])
                    sens_scores.append(mets["sensitivity"])
                    prec_scores.append(mets["precision"])

            mean_dice = float(np.mean(dice_scores))
            mean_iou = float(np.mean(iou_scores))
            mean_sens = float(np.mean(sens_scores))
            mean_prec = float(np.mean(prec_scores))
            fpr_pct = 100.0 * false_alarms / max(1, neg_count)

            # Balanced Optimization Objective:
            # Maximize Dice while heavily penalizing False Positive Rate on healthy brains
            # Composite = Dice * (1.0 - 0.5 * (FPR / 100))
            composite_metric = mean_dice * (1.0 - 0.5 * (fpr_pct / 100.0))

            log_entry = {
                "trial": trial_idx,
                "r_s_ratio": r_s,
                "tau_rej": tau_rej,
                "mean_dice": round(mean_dice, 4),
                "mean_iou": round(mean_iou, 4),
                "mean_sens": round(mean_sens, 4),
                "mean_prec": round(mean_prec, 4),
                "false_alarms": false_alarms,
                "neg_count": neg_count,
                "fpr_pct": round(fpr_pct, 2),
                "composite_score": round(composite_metric, 4),
            }
            tuning_logs.append(log_entry)

            print(f"Trial {trial_idx:2d}/{total_trials} | r_s={r_s:.2f}, tau_rej={tau_rej:.2f} -> Dice={mean_dice:.4f}, FPR={fpr_pct:5.1f}% | Composite={composite_metric:.4f}")

            if composite_metric > best_composite_score:
                best_composite_score = composite_metric
                best_cfg = curr_cfg
                best_trial_params = log_entry

    df_logs = pd.DataFrame(tuning_logs)
    tuning_log_path = results_dir / "tuning_log.csv"
    df_logs.to_csv(tuning_log_path, index=False)
    print(f"\nSaved grid search tuning logs to: {tuning_log_path}")

    # Freeze best configuration
    final_yaml_path = configs_dir / "final.yaml"
    with open(final_yaml_path, "w") as f:
        yaml.safe_dump(best_cfg, f, default_flow_style=False, sort_keys=False)

    print("=" * 75)
    print(f"OPTIMAL CONFIGURATION FROZEN TO: {final_yaml_path}")
    print("=" * 75)
    print(f"  Best Trial:           #{best_trial_params['trial']}")
    print(f"  r_s_ratio:            {best_trial_params['r_s_ratio']}")
    print(f"  tau_rej:              {best_trial_params['tau_rej']}")
    print(f"  Mean Dice on Tumors:  {best_trial_params['mean_dice']:.4f}")
    print(f"  Mean IoU:             {best_trial_params['mean_iou']:.4f}")
    print(f"  False Positive Rate:  {best_trial_params['fpr_pct']:.1f}% ({best_trial_params['false_alarms']}/{best_trial_params['neg_count']} false alarms)")
    print(f"  Composite Score:      {best_trial_params['composite_score']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    run_tuning()
