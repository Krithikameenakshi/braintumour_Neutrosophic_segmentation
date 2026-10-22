"""Phase 2: Baseline Benchmark Evaluation on the 100-Image Tuning Subset.

Evaluates 5 classical baselines:
1. Otsu Thresholding
2. k-Means Clustering (k=3)
3. Fuzzy C-Means (FCM)
4. Plain Guo-Cheng Neutrosophic Thresholding
5. Tufail-style NS ROI

Reports:
- Mean +/- SD for Dice, IoU, Sensitivity, Precision, FNR, HD95 on tumor slices
- False Positive Rate (FPR) and false alarm statistics on no-tumor slices
- Saves results/baselines_tuning.csv and outputs/phase2_baselines_summary.csv
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
from src.preprocess import preprocess_image
from src.metrics import compute_metrics
from src.baselines import (
    baseline_otsu,
    baseline_kmeans,
    baseline_fuzzy_c_means,
    baseline_guo_cheng_ns,
    baseline_tufail_ns,
)


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase2():
    cfg = load_config()
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2: RUNNING BASELINE BENCHMARKS ON 100-IMAGE TUNING SUBSET")
    print("=" * 75)

    tuning_csv_path = Path(cfg["paths"]["tuning_subset_path"])
    tuning_df = pd.read_csv(tuning_csv_path)
    print(f"Loaded tuning subset ({len(tuning_df)} images from {tuning_csv_path}).")

    methods = {
        "Otsu": lambda g, mb: baseline_otsu(g, mb),
        "k-Means (k=3)": lambda g, mb: baseline_kmeans(g, mb, k=cfg["baselines"]["kmeans"]["k"], seed=cfg["experiment"]["seed"]),
        "FCM (c=3)": lambda g, mb: baseline_fuzzy_c_means(
            g, mb,
            n_clusters=cfg["baselines"]["fuzzy_c_means"]["n_clusters"],
            m=cfg["baselines"]["fuzzy_c_means"]["m"],
            max_iter=cfg["baselines"]["fuzzy_c_means"]["max_iter"],
            error=cfg["baselines"]["fuzzy_c_means"]["error"],
        ),
        "Guo-Cheng NS": lambda g, mb: baseline_guo_cheng_ns(g, mb, window_size=cfg["neutrosophic"]["window_size"]),
        "Tufail NS ROI": lambda g, mb: baseline_tufail_ns(
            g, mb,
            sd_window=cfg["baselines"]["tufail_ns"]["sd_window"],
            sobel_ksize=cfg["baselines"]["tufail_ns"]["sobel_ksize"],
            tt=cfg["baselines"]["tufail_ns"]["tt"],
            tf=cfg["baselines"]["tufail_ns"]["tf"],
        ),
    }

    all_results = []

    print("\nEvaluating all 5 baselines across tuning subset...")
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

        # Preprocessing
        g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)

        for m_name, m_func in methods.items():
            pred = m_func(g_denoised, brain_mask)
            mets = compute_metrics(pred, gt)

            res_record = {
                "filename": fn,
                "tumor_class": t_class,
                "plane": plane,
                "method": m_name,
                "is_empty_gt": mets["is_empty_gt"],
                "pred_empty": mets["pred_empty"],
                "dice": mets["dice"],
                "iou": mets["iou"],
                "sensitivity": mets["sensitivity"],
                "precision": mets["precision"],
                "fnr": mets["fnr"],
                "hd95": mets["hd95"],
                "false_alarm": mets["false_alarm"],
                "fp_pixel_count": mets["fp_pixel_count"],
            }
            all_results.append(res_record)

    df_res = pd.DataFrame(all_results)
    detailed_csv = results_dir / "baselines_tuning.csv"
    df_res.to_csv(detailed_csv, index=False)
    print(f"\nSaved detailed evaluation per slice to: {detailed_csv}")

    # Summary analysis
    # 1. Tumor cases only (GT > 0)
    tumor_res = df_res[~df_res["is_empty_gt"]].copy()
    tumor_summary = tumor_res.groupby("method").agg(
        Dice_mean=("dice", "mean"),
        Dice_std=("dice", "std"),
        IoU_mean=("iou", "mean"),
        IoU_std=("iou", "std"),
        Sens_mean=("sensitivity", "mean"),
        Sens_std=("sensitivity", "std"),
        Prec_mean=("precision", "mean"),
        Prec_std=("precision", "std"),
        FNR_mean=("fnr", "mean"),
        FNR_std=("fnr", "std"),
        HD95_mean=("hd95", lambda x: np.nanmean(x)),
        HD95_std=("hd95", lambda x: np.nanstd(x)),
    )

    # 2. No-tumor cases only (GT == 0)
    empty_res = df_res[df_res["is_empty_gt"]].copy()
    no_tumor_summary = empty_res.groupby("method").agg(
        Total_Negative_Slices=("false_alarm", "count"),
        False_Alarms=("false_alarm", "sum"),
        False_Positive_Rate_pct=("false_alarm", lambda x: 100.0 * np.mean(x)),
        Avg_FP_Pixels=("fp_pixel_count", "mean"),
    )

    combined_summary = tumor_summary.join(no_tumor_summary)
    summary_csv = outputs_dir / "phase2_baselines_summary.csv"
    combined_summary.to_csv(summary_csv)
    print(f"Saved summary metrics to: {summary_csv}\n")

    print("=" * 75)
    print("BASELINE PERFORMANCE SUMMARY (Tumor Cases: N=79, No-Tumor Cases: N=21)")
    print("=" * 75)
    for m in combined_summary.index:
        row = combined_summary.loc[m]
        print(f"\nMethod: {m}")
        print(f"  Dice (DSC):         {row['Dice_mean']:.4f} +/- {row['Dice_std']:.4f}")
        print(f"  IoU (Jaccard):      {row['IoU_mean']:.4f} +/- {row['IoU_std']:.4f}")
        print(f"  Sensitivity:        {row['Sens_mean']:.4f} +/- {row['Sens_std']:.4f}")
        print(f"  Precision:          {row['Prec_mean']:.4f} +/- {row['Prec_std']:.4f}")
        print(f"  FNR:                {row['FNR_mean']:.4f} +/- {row['FNR_std']:.4f}")
        print(f"  HD95 (pixels):      {row['HD95_mean']:.2f} +/- {row['HD95_std']:.2f}")
        print(f"  False Positive Rate:{row['False_Positive_Rate_pct']:.1f}% ({int(row['False_Alarms'])}/{int(row['Total_Negative_Slices'])} false alarms on healthy brain)")

    # Breakdown by tumor type
    print("\n--- Dice Score by Tumor Class ---")
    class_dice = tumor_res.groupby(["method", "tumor_class"])["dice"].agg(["mean", "std"])
    print(class_dice)
    class_dice.to_csv(outputs_dir / "phase2_baselines_dice_by_class.csv")

    # Plot comparisons
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Boxplot of Dice Scores on Tumor Cases
    method_names = list(methods.keys())
    dice_data = [tumor_res[tumor_res["method"] == m]["dice"].dropna().values for m in method_names]
    axes[0].boxplot(dice_data, tick_labels=method_names, patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[0].set_title("Dice Similarity Coefficient (DSC) on Tuning Subset (N=79)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20)

    # 2. Bar Chart of False Positive Rate on No-Tumor Cases
    fpr_vals = [no_tumor_summary.loc[m, "False_Positive_Rate_pct"] for m in method_names]
    axes[1].bar(method_names, fpr_vals, color="#C44E52", alpha=0.75, edgecolor="black")
    axes[1].set_title("False Positive Rate on Healthy No-Tumor Slices (N=21)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("False Positive Rate (%)")
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(fpr_vals):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20)

    plt.tight_layout()
    plot_path = outputs_dir / "phase2_baselines_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved baseline comparison figure to: {plot_path}")
    print("\nPhase 2 completed successfully!")


if __name__ == "__main__":
    run_phase2()
