"""Phase 3 Evaluation on 100-Image Tuning Subset.

Runs the full Neutrosophic Pipeline across the 100 tuning images.
Compares performance directly against the 5 baselines.
Extracts and plots the 10 best and 10 worst cases with T/I/F maps.
Diagnoses main failure modes.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset
from src.pipeline import run_neutrosophic_segmentation
from src.metrics import compute_metrics


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_cases_panel(case_df, title, save_path, cfg):
    """Generates a diagnostic grid showing Original, T*, I*, F*, Ground Truth,

    Prediction Overlay.
    """
    n_cases = len(case_df)
    fig, axes = plt.subplots(n_cases, 6, figsize=(18, 3 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, 0)

    for i, (_, row) in enumerate(case_df.iterrows()):
        img_p = row["image_path"]
        mask_p = row["mask_path"]
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)

        if mask_p is not None and pd.notna(mask_p) and Path(str(mask_p)).exists():
            gt_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            gt = (gt_raw > 127).astype(np.uint8)
        else:
            gt = np.zeros_like(img, dtype=np.uint8)

        res = run_neutrosophic_segmentation(img, cfg)
        mask = res["mask"]
        overlay = res["readouts"]["overlay_rgb"]
        t_star = res["T_star"]
        i_star = res["I_star"]
        f_star = res["F_star"]

        # 0: Original
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_ylabel(f"#{i+1}: {row['tumor_class']}\nDice: {row['dice']:.3f}", fontsize=9, fontweight="bold")
        if i == 0:
            axes[i, 0].set_title("1. Original MRI", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 1: T*
        axes[i, 1].imshow(t_star, cmap="hot")
        if i == 0:
            axes[i, 1].set_title("2. Truth (T*)", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # 2: I*
        axes[i, 2].imshow(i_star, cmap="inferno")
        if i == 0:
            axes[i, 2].set_title("3. Indeterminacy (I*)", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

        # 3: F*
        axes[i, 3].imshow(f_star, cmap="viridis")
        if i == 0:
            axes[i, 3].set_title("4. Falsehood (F*)", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

        # 4: Ground Truth
        axes[i, 4].imshow(gt, cmap="Greens")
        if i == 0:
            axes[i, 4].set_title("5. Radiologist GT", fontsize=11, fontweight="bold")
        axes[i, 4].axis("off")

        # 5: Overlay (Red contour/highlight)
        axes[i, 5].imshow(overlay)
        if i == 0:
            axes[i, 5].set_title("6. Pipeline Overlay", fontsize=11, fontweight="bold")
        axes[i, 5].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic panel to: {save_path}")


def run_phase3_eval():
    cfg = load_config()
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 3: FULL NEUTROSOPHIC PIPELINE EVALUATION ON TUNING SUBSET")
    print("=" * 75)

    tuning_csv_path = Path(cfg["paths"]["tuning_subset_path"])
    tuning_df = pd.read_csv(tuning_csv_path)
    print(f"Loaded tuning subset ({len(tuning_df)} images).")

    pipeline_results = []

    print("\nRunning full Neutrosophic Pipeline across 100 tuning images...")
    for idx, row in tqdm(tuning_df.iterrows(), total=len(tuning_df)):
        img_p = row["image_path"]
        mask_p = row["mask_path"]
        t_class = row["tumor_class"]
        plane = row["plane"]
        fn = row["filename"]

        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if mask_p is not None and pd.notna(mask_p) and Path(str(mask_p)).exists():
            gt_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            gt = (gt_raw > 127).astype(np.uint8)
        else:
            gt = np.zeros_like(img, dtype=np.uint8)

        # Full pipeline inference
        res = run_neutrosophic_segmentation(img, cfg)
        pred_mask = res["mask"]
        readouts = res["readouts"]

        mets = compute_metrics(pred_mask, gt)

        record = {
            "filename": fn,
            "tumor_class": t_class,
            "plane": plane,
            "image_path": img_p,
            "mask_path": mask_p,
            "is_empty_gt": mets["is_empty_gt"],
            "pred_empty": mets["pred_empty"],
            "dice": mets["dice"],
            "iou": mets["iou"],
            "sensitivity": mets["sensitivity"],
            "precision": mets["precision"],
            "fnr": mets["fnr"],
            "hd95": mets["hd95"],
            "false_alarm": mets["false_alarm"],
            "candidate_score": res["candidate_score"],
            "polarity": res["polarity"],
            "boundary_uncertainty": readouts["boundary_uncertainty"],
            "region_confidence": readouts["region_confidence"],
            "tumor_pixels": readouts["tumor_area_pixels"],
        }
        pipeline_results.append(record)

    df_p3 = pd.DataFrame(pipeline_results)
    p3_csv_path = results_dir / "phase3_tuning_results.csv"
    df_p3.to_csv(p3_csv_path, index=False)
    print(f"\nSaved detailed Phase 3 results to: {p3_csv_path}")

    # Summarize Tumor Slices (GT > 0)
    tumor_df = df_p3[~df_p3["is_empty_gt"]].copy()
    no_tumor_df = df_p3[df_p3["is_empty_gt"]].copy()

    p3_tumor_summary = {
        "Dice_mean": tumor_df["dice"].mean(),
        "Dice_std": tumor_df["dice"].std(),
        "IoU_mean": tumor_df["iou"].mean(),
        "IoU_std": tumor_df["iou"].std(),
        "Sens_mean": tumor_df["sensitivity"].mean(),
        "Sens_std": tumor_df["sensitivity"].std(),
        "Prec_mean": tumor_df["precision"].mean(),
        "Prec_std": tumor_df["precision"].std(),
        "FNR_mean": tumor_df["fnr"].mean(),
        "FNR_std": tumor_df["fnr"].std(),
        "HD95_mean": np.nanmean(tumor_df["hd95"]),
        "HD95_std": np.nanstd(tumor_df["hd95"]),
        "False_Positive_Rate_pct": 100.0 * np.mean(no_tumor_df["false_alarm"]),
        "False_Alarms": int(np.sum(no_tumor_df["false_alarm"])),
        "Total_Negatives": len(no_tumor_df),
    }

    # Load baseline summary for side-by-side comparison
    base_summary_path = outputs_dir / "phase2_baselines_summary.csv"
    if base_summary_path.exists():
        base_df = pd.read_csv(base_summary_path, index_col=0)
    else:
        base_df = pd.DataFrame()

    p3_row = pd.DataFrame([p3_tumor_summary], index=["Neutrosophic Pipeline (Phase 3)"])
    comparison_table = pd.concat([base_df, p3_row])
    comp_csv_path = outputs_dir / "phase3_vs_baselines_summary.csv"
    comparison_table.to_csv(comp_csv_path)

    print("\n" + "=" * 75)
    print("COMPARATIVE EVALUATION: NEUTROSOPHIC PIPELINE vs BASELINES")
    print("=" * 75)
    print(comparison_table[["Dice_mean", "Dice_std", "IoU_mean", "Sens_mean", "Prec_mean", "HD95_mean", "False_Positive_Rate_pct"]].to_string())

    # Breakdown by tumor class
    print("\n--- Phase 3 Dice Score by Tumor Class ---")
    p3_class_dice = tumor_df.groupby("tumor_class")["dice"].agg(["mean", "std", "count"])
    print(p3_class_dice)
    p3_class_dice.to_csv(outputs_dir / "phase3_dice_by_class.csv")

    # 10 Best and 10 Worst cases
    tumor_sorted = tumor_df.sort_values(by="dice", ascending=False).reset_index(drop=True)
    best_10 = tumor_sorted.head(10)
    worst_10 = tumor_sorted.tail(10)

    best_10.to_csv(outputs_dir / "phase3_best_10_metadata.csv", index=False)
    worst_10.to_csv(outputs_dir / "phase3_worst_10_metadata.csv", index=False)

    print("\nGenerating 10 Best Cases visualization panel...")
    plot_cases_panel(
        best_10,
        "Phase 3: Top 10 Best Performing Tumor Segmentations (Highest Dice Scores)",
        outputs_dir / "phase3_10_best_cases.png",
        cfg
    )

    print("\nGenerating 10 Worst Cases visualization panel...")
    plot_cases_panel(
        worst_10,
        "Phase 3: Top 10 Challenging / Failure Cases (Lowest Dice Scores)",
        outputs_dir / "phase3_10_worst_cases.png",
        cfg
    )

    print("\nPhase 3 evaluation completed successfully!")


if __name__ == "__main__":
    run_phase3_eval()
