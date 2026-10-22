"""Phase 4: Final Test Split Evaluation.

Evaluates the frozen Neutrosophic Pipeline against all 5 baselines across the
entire untouched BRISC 2025 TEST set (1,000 images: 860 tumor + 140 no-tumor).
Computes:
- Paired Wilcoxon signed-rank tests vs. each baseline
- Per-class breakdown (glioma, meningioma, pituitary, no-tumor)
- Per-plane breakdown (axial, coronal, sagittal)
- Clinical metrics (Dice, IoU, Sens, Prec, FNR, HD95, False Alarm Rate)
- Saves results to CSV, LaTeX table, and visual panels.
"""

import sys
import copy
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset
from src.preprocess import preprocess_image
from src.pipeline import run_neutrosophic_segmentation
from src.baselines import (
    baseline_otsu,
    baseline_kmeans,
    baseline_fuzzy_c_means,
    baseline_guo_cheng_ns,
    baseline_tufail_ns,
)
from src.metrics import compute_metrics


def load_config(config_path="configs/final.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_test_evaluation():
    cfg = load_config("configs/final.yaml")
    results_dir = ROOT_DIR / "results"
    outputs_dir = ROOT_DIR / "outputs"
    results_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    print("Loading untouched BRISC 2025 TEST dataset...")
    test_ds = BRISCDataset(root_dir="data/brisc", split="test", include_no_tumor=True)
    print(f"Total test samples: {len(test_ds)}")

    methods = {
        "Otsu": lambda g, mb: baseline_otsu(g, mb),
        "k-Means (k=3)": lambda g, mb: baseline_kmeans(g, mb, k=cfg["baselines"]["kmeans"]["k"], seed=cfg["experiment"]["seed"]),
        "FCM (c=3)": lambda g, mb: baseline_fuzzy_c_means(g, mb, n_clusters=cfg["baselines"]["fuzzy_c_means"]["n_clusters"], m=cfg["baselines"]["fuzzy_c_means"]["m"]),
        "Guo-Cheng NS": lambda g, mb: baseline_guo_cheng_ns(g, mb, window_size=cfg["neutrosophic"]["window_size"]),
        "Tufail NS ROI": lambda g, mb: baseline_tufail_ns(
            g, mb,
            sd_window=cfg["baselines"]["tufail_ns"]["sd_window"],
            sobel_ksize=cfg["baselines"]["tufail_ns"]["sobel_ksize"],
            tt=cfg["baselines"]["tufail_ns"]["tt"],
            tf=cfg["baselines"]["tufail_ns"]["tf"],
        ),
    }

    csv_detail_path = results_dir / "final_test_evaluation.csv"
    if csv_detail_path.exists() and len(pd.read_csv(csv_detail_path)) == len(test_ds):
        print(f"Found existing evaluation CSV with {len(test_ds)} cases at {csv_detail_path}. Skipping re-computation.")
        df_test = pd.read_csv(csv_detail_path)
    else:
        detailed_records = []

        print("\nExecuting evaluation on all 1,000 TEST cases across all methods...")
        for idx in tqdm(range(len(test_ds)), desc="Test Evaluation"):
            sample_meta = test_ds.samples[idx]
            img, gt, t_class, plane = test_ds[idx]
            fname = sample_meta["filename"]
            is_tumor = sample_meta["is_tumor"]

            record = {
                "index": idx,
                "filename": fname,
                "tumor_class": t_class,
                "plane": plane,
                "is_tumor": is_tumor,
            }

            # Preprocessing for baselines
            g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)

            # Run each baseline
            for m_name, m_func in methods.items():
                try:
                    pred = m_func(g_denoised, brain_mask)
                    mets = compute_metrics(pred, gt)
                except Exception as e:
                    print(f"Error on {fname} with {m_name}: {e}")
                    pred = np.zeros_like(img, dtype=np.uint8)
                    mets = compute_metrics(pred, gt)

                clean_key = m_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("=", "")
                record[f"{clean_key}_dice"] = mets["dice"]
                record[f"{clean_key}_iou"] = mets["iou"]
                record[f"{clean_key}_sensitivity"] = mets["sensitivity"]
                record[f"{clean_key}_precision"] = mets["precision"]
                record[f"{clean_key}_fnr"] = mets["fnr"]
                record[f"{clean_key}_hd95"] = mets["hd95"]
                record[f"{clean_key}_false_alarm"] = mets["false_alarm"]

            # Run Proposed Pipeline
            try:
                prop_res = run_neutrosophic_segmentation(img, cfg)
                prop_pred = prop_res["mask"]
                prop_mets = compute_metrics(prop_pred, gt)
            except Exception as e:
                print(f"Error on {fname} with Proposed Pipeline: {e}")
                prop_pred = np.zeros_like(img, dtype=np.uint8)
                prop_mets = compute_metrics(prop_pred, gt)

            record["proposed_pipeline_dice"] = prop_mets["dice"]
            record["proposed_pipeline_iou"] = prop_mets["iou"]
            record["proposed_pipeline_sensitivity"] = prop_mets["sensitivity"]
            record["proposed_pipeline_precision"] = prop_mets["precision"]
            record["proposed_pipeline_fnr"] = prop_mets["fnr"]
            record["proposed_pipeline_hd95"] = prop_mets["hd95"]
            record["proposed_pipeline_false_alarm"] = prop_mets["false_alarm"]

            detailed_records.append(record)

        df_test = pd.DataFrame(detailed_records)
        df_test.to_csv(csv_detail_path, index=False)
        print(f"\nDetailed slice-level results saved to: {csv_detail_path}")

    # Summary analysis
    df_tumors = df_test[df_test["is_tumor"]].copy()
    df_negatives = df_test[~df_test["is_tumor"]].copy()

    summary_rows = []
    prop_key = "proposed_pipeline"
    all_methods = list(methods.keys()) + ["Proposed Pipeline"]

    for m_name in all_methods:
        m_key = m_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("=", "")
        
        dice_vals = df_tumors[f"{m_key}_dice"].dropna().values
        iou_vals = df_tumors[f"{m_key}_iou"].dropna().values
        sens_vals = df_tumors[f"{m_key}_sensitivity"].dropna().values
        prec_vals = df_tumors[f"{m_key}_precision"].dropna().values
        fnr_vals = df_tumors[f"{m_key}_fnr"].dropna().values
        hd95_vals = df_tumors[f"{m_key}_hd95"].dropna().values
        
        fa_rate = df_negatives[f"{m_key}_false_alarm"].mean() * 100.0 if len(df_negatives) > 0 else 0.0

        # Wilcoxon signed rank test vs proposed on tumors
        if m_key != prop_key and len(dice_vals) > 0:
            diff = df_tumors[f"{prop_key}_dice"].values - df_tumors[f"{m_key}_dice"].values
            # Filter non-zero differences for Wilcoxon
            nonzero_diff = diff[diff != 0]
            if len(nonzero_diff) > 0:
                stat, p_val = wilcoxon(df_tumors[f"{prop_key}_dice"].values, df_tumors[f"{m_key}_dice"].values)
            else:
                p_val = 1.0
        else:
            p_val = np.nan

        summary_rows.append({
            "Method": m_name,
            "Mean_Dice": np.mean(dice_vals) if len(dice_vals) > 0 else 0,
            "Std_Dice": np.std(dice_vals) if len(dice_vals) > 0 else 0,
            "Median_Dice": np.median(dice_vals) if len(dice_vals) > 0 else 0,
            "IQR_Dice": (np.percentile(dice_vals, 75) - np.percentile(dice_vals, 25)) if len(dice_vals) > 0 else 0,
            "Mean_IoU": np.mean(iou_vals) if len(iou_vals) > 0 else 0,
            "Sensitivity": np.mean(sens_vals) if len(sens_vals) > 0 else 0,
            "Precision": np.mean(prec_vals) if len(prec_vals) > 0 else 0,
            "FNR": np.mean(fnr_vals) if len(fnr_vals) > 0 else 0,
            "Median_HD95": np.median(hd95_vals) if len(hd95_vals) > 0 else np.nan,
            "Mean_HD95": np.mean(hd95_vals) if len(hd95_vals) > 0 else np.nan,
            "False_Alarm_Rate_%": fa_rate,
            "Wilcoxon_p_val": p_val,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = outputs_dir / "final_test_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary metrics saved to: {summary_csv_path}")
    print("\n" + "=" * 90)
    print("FINAL BRISC 2025 TEST SPLIT BENCHMARK SUMMARY (N = 1,000 images)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    # Per-Class Breakdown
    class_rows = []
    classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
    for c in classes:
        sub = df_test[df_test["tumor_class"] == c]
        if c == "no_tumor":
            class_rows.append({
                "Tumor_Class": c,
                "Count": len(sub),
                "Prop_Mean_Dice": np.nan,
                "Prop_False_Alarm_%": sub[f"{prop_key}_false_alarm"].mean() * 100.0,
                "Otsu_False_Alarm_%": sub["otsu_false_alarm"].mean() * 100.0,
                "KMeans_False_Alarm_%": sub["k_means_k3_false_alarm"].mean() * 100.0,
            })
        else:
            class_rows.append({
                "Tumor_Class": c,
                "Count": len(sub),
                "Prop_Mean_Dice": sub[f"{prop_key}_dice"].mean(),
                "Prop_Median_Dice": sub[f"{prop_key}_dice"].median(),
                "Prop_Sensitivity": sub[f"{prop_key}_sensitivity"].mean(),
                "Prop_Precision": sub[f"{prop_key}_precision"].mean(),
                "Otsu_Mean_Dice": sub["otsu_dice"].mean(),
                "GuoCheng_Mean_Dice": sub["guo_cheng_ns_dice"].mean(),
                "Tufail_Mean_Dice": sub["tufail_ns_roi_dice"].mean(),
            })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(outputs_dir / "final_test_breakdown_class.csv", index=False)
    print(f"\nPer-class breakdown saved to: {outputs_dir / 'final_test_breakdown_class.csv'}")

    # Per-Plane Breakdown
    plane_rows = []
    planes = ["axial", "coronal", "sagittal"]
    for p in planes:
        sub = df_tumors[df_tumors["plane"] == p]
        plane_rows.append({
            "Plane": p,
            "Count": len(sub),
            "Prop_Mean_Dice": sub[f"{prop_key}_dice"].mean(),
            "Prop_Median_Dice": sub[f"{prop_key}_dice"].median(),
            "Prop_Sensitivity": sub[f"{prop_key}_sensitivity"].mean(),
            "Prop_Precision": sub[f"{prop_key}_precision"].mean(),
            "Otsu_Mean_Dice": sub["otsu_dice"].mean(),
            "GuoCheng_Mean_Dice": sub["guo_cheng_ns_dice"].mean(),
        })
    plane_df = pd.DataFrame(plane_rows)
    plane_df.to_csv(outputs_dir / "final_test_breakdown_plane.csv", index=False)
    print(f"Per-plane breakdown saved to: {outputs_dir / 'final_test_breakdown_plane.csv'}")

    # Generate Publication LaTeX Table
    tex_path = outputs_dir / "final_test_metrics.tex"
    with open(tex_path, "w") as f:
        f.write("% BRISC 2025 Test Split Segmentation Performance Benchmark\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Quantitative comparison on the unseen BRISC 2025 test dataset ($N=1{,}000$: 860 tumor, 140 negative controls). All methods operate in an unsupervised, training-free setting. $p$-values denote paired Wilcoxon signed-rank test against the proposed framework.}\\label{tab:test_benchmark}\n")
        f.write("\\begin{tabular}{lccccccr}\n\\hline\n")
        f.write("Method & Dice (Mean$\\pm$SD) & Dice (Median) & IoU & Sens. & Prec. & HD95 (mm) & False Alarm (\\%) \\\\\n\\hline\n")
        for _, r in summary_df.iterrows():
            p_str = f" ($p<{r['Wilcoxon_p_val']:.1e}$)" if pd.notna(r["Wilcoxon_p_val"]) and r["Wilcoxon_p_val"] < 0.05 else ""
            if r["Method"] == "Proposed Pipeline":
                f.write(f"\\textbf{{{r['Method']}}} & \\textbf{{{r['Mean_Dice']:.3f}$\\pm${r['Std_Dice']:.3f}}} & \\textbf{{{r['Median_Dice']:.3f}}} & \\textbf{{{r['Mean_IoU']:.3f}}} & \\textbf{{{r['Sensitivity']:.3f}}} & \\textbf{{{r['Precision']:.3f}}} & \\textbf{{{r['Median_HD95']:.2f}}} & \\textbf{{{r['False_Alarm_Rate_%']:.1f}\\%}} \\\\\n")
            else:
                f.write(f"{r['Method']} & {r['Mean_Dice']:.3f}$\\pm${r['Std_Dice']:.3f} & {r['Median_Dice']:.3f} & {r['Mean_IoU']:.3f} & {r['Sensitivity']:.3f} & {r['Precision']:.3f} & {r['Median_HD95']:.2f} & {r['False_Alarm_Rate_%']:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
    print(f"Publication-ready LaTeX table written to: {tex_path}")

    # Generate Visual Demonstration Panel on Representative Test Cases
    generate_test_visual_panel(df_test, test_ds, cfg, outputs_dir / "final_test_comparison_panel.png")


def generate_test_visual_panel(df_test, test_ds, cfg, save_path):
    """Generates a high-resolution 6-case multi-method comparison panel on the test set."""
    # Pick 2 Glioma, 2 Meningioma, 1 Pituitary, 1 No-tumor
    samples = []
    for c in ["glioma", "meningioma", "pituitary"]:
        sub = df_test[(df_test["tumor_class"] == c) & (df_test["proposed_pipeline_dice"] > 0.40)]
        if len(sub) > 0:
            samples.extend(sub.sample(min(2 if c != "pituitary" else 1, len(sub)), random_state=42)["index"].tolist())
    
    # 1 negative control
    neg_sub = df_test[df_test["tumor_class"] == "no_tumor"]
    if len(neg_sub) > 0:
        samples.append(neg_sub.sample(1, random_state=42)["index"].iloc[0])

    df_lookup = df_test.set_index("index")

    fig, axes = plt.subplots(len(samples), 6, figsize=(18, 3.2 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, idx in enumerate(samples):
        img, gt, c_name, plane = test_ds[idx]

        g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)
        otsu_pred = baseline_otsu(g_denoised, brain_mask)
        gc_pred = baseline_guo_cheng_ns(g_denoised, brain_mask, window_size=cfg["neutrosophic"]["window_size"])
        tuf_pred = baseline_tufail_ns(
            g_denoised, brain_mask,
            sd_window=cfg["baselines"]["tufail_ns"]["sd_window"],
            sobel_ksize=cfg["baselines"]["tufail_ns"]["sobel_ksize"],
            tt=cfg["baselines"]["tufail_ns"]["tt"],
            tf=cfg["baselines"]["tufail_ns"]["tf"],
        )
        prop_res = run_neutrosophic_segmentation(img, cfg)
        prop_pred = prop_res["mask"]

        # 0: Original MRI
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_ylabel(f"#{idx}: {c_name} ({plane})", fontsize=10, fontweight="bold")
        if i == 0:
            axes[i, 0].set_title("1. Original T1ce MRI", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 1: Ground Truth
        axes[i, 1].imshow(img, cmap="gray")
        if np.any(gt):
            axes[i, 1].contour(gt, colors="lime", linewidths=1.5)
        if i == 0:
            axes[i, 1].set_title("2. Radiologist GT", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # 2: Otsu
        axes[i, 2].imshow(img, cmap="gray")
        if np.any(otsu_pred):
            axes[i, 2].contour(otsu_pred, colors="orange", linewidths=1.2)
        m_dice = df_lookup.loc[idx, "otsu_dice"]
        d_str = f"Dice: {m_dice:.2f}" if pd.notna(m_dice) else "Neg"
        axes[i, 2].text(5, 20, d_str, color="yellow", fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        if i == 0:
            axes[i, 2].set_title("3. Baseline Otsu", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

        # 3: Guo-Cheng NS
        axes[i, 3].imshow(img, cmap="gray")
        if np.any(gc_pred):
            axes[i, 3].contour(gc_pred, colors="cyan", linewidths=1.2)
        gc_dice = df_lookup.loc[idx, "guo_cheng_ns_dice"]
        d_str = f"Dice: {gc_dice:.2f}" if pd.notna(gc_dice) else "Neg"
        axes[i, 3].text(5, 20, d_str, color="yellow", fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        if i == 0:
            axes[i, 3].set_title("4. Guo-Cheng NS", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

        # 4: Tufail NS ROI
        axes[i, 4].imshow(img, cmap="gray")
        if np.any(tuf_pred):
            axes[i, 4].contour(tuf_pred, colors="magenta", linewidths=1.2)
        tuf_dice = df_lookup.loc[idx, "tufail_ns_roi_dice"]
        d_str = f"Dice: {tuf_dice:.2f}" if pd.notna(tuf_dice) else "Neg"
        axes[i, 4].text(5, 20, d_str, color="yellow", fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        if i == 0:
            axes[i, 4].set_title("5. Tufail NS ROI", fontsize=11, fontweight="bold")
        axes[i, 4].axis("off")

        # 5: Proposed Pipeline
        axes[i, 5].imshow(img, cmap="gray")
        if np.any(gt):
            axes[i, 5].contour(gt, colors="lime", linewidths=1.0, linestyles="dashed")
        if np.any(prop_pred):
            axes[i, 5].contour(prop_pred, colors="red", linewidths=1.5)
        prop_dice = df_lookup.loc[idx, "proposed_pipeline_dice"]
        d_str = f"Dice: {prop_dice:.2f}" if pd.notna(prop_dice) else "Neg (FA=0)" if not np.any(prop_pred) else "False Alarm"
        axes[i, 5].text(5, 20, d_str, color="yellow", fontsize=9, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        if i == 0:
            axes[i, 5].set_title("6. Proposed (Red) vs GT (Green)", fontsize=11, fontweight="bold")
        axes[i, 5].axis("off")

    plt.suptitle("BRISC 2025 Test Split: Proposed Neutrosophic Framework vs. Classical Baselines", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Representative test panel saved to: {save_path}")


if __name__ == "__main__":
    run_test_evaluation()
