"""Phase 1 Project Review: 50 Patient MRI Test Suite.

Runs the Neutrosophic segmentation pipeline across 50 diverse patient MRI scans.
Logs calculated T, I, F values (mean T, mean I, mean F, indeterminacy entropy),
clinical metrics (Dice, IoU, Sensitivity, Precision, HD95),
and generates visual panels showing Original, T*, I*, F*, GT, and Overlay.
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
from src.ns import neutrosophic_transform, compute_entropy
from src.pipeline import run_neutrosophic_segmentation
from src.metrics import compute_metrics


def run_50_patient_evaluation():
    cfg_path = ROOT_DIR / "configs" / "final.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = ROOT_DIR / "results"
    outputs_dir = ROOT_DIR / "outputs"
    results_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    print("Loading BRISC dataset for 50-patient review suite...")
    # Load dataset
    ds = BRISCDataset(root_dir="data/brisc", split="train", include_no_tumor=True)
    df_meta = ds.get_metadata_df()

    # Stratify 50 samples: 15 glioma, 15 meningioma, 15 pituitary, 5 no-tumor
    sampled_indices = []
    classes_target = {
        "glioma": 15,
        "meningioma": 15,
        "pituitary": 15,
        "no_tumor": 5,
    }

    for c_name, count in classes_target.items():
        sub = df_meta[df_meta["tumor_class"] == c_name]
        chosen = sub.sample(count, random_state=42).index.tolist()
        sampled_indices.extend(chosen)

    print(f"Selected {len(sampled_indices)} patient MRI slices across 4 categories.")

    records = []
    visual_cases = []

    print("Processing 50 patient scans and computing T, I, F and clinical metrics...")
    for rank, idx in enumerate(tqdm(sampled_indices, desc="Evaluating 50 Patients")):
        img, gt, tumor_class, plane = ds[idx]
        sample_meta = ds.samples[idx]
        fname = sample_meta["filename"]
        is_tumor = sample_meta["is_tumor"]

        # 1. Preprocessing
        g_norm, head_mask, brain_mask, g_den = preprocess_image(img, cfg)

        # 2. Neutrosophic decomposition on normalized denoised image
        w_size = cfg["neutrosophic"]["window_size"]
        T, I, F = neutrosophic_transform(g_den, window_size=w_size)

        # Masked statistics inside brain parenchyma
        valid = brain_mask > 0
        mean_t = float(np.mean(T[valid])) if np.sum(valid) > 0 else 0.0
        mean_i = float(np.mean(I[valid])) if np.sum(valid) > 0 else 0.0
        mean_f = float(np.mean(F[valid])) if np.sum(valid) > 0 else 0.0
        entropy_i = float(compute_entropy(I, brain_mask, bins=256))

        # 3. Full Segmentation Pipeline
        res = run_neutrosophic_segmentation(img, cfg)
        pred_mask = res["mask"]
        t_star = res["T_star"]
        i_star = res["I_star"]
        f_star = res["F_star"]
        overlay_rgb = res["readouts"]["overlay_rgb"]
        cand_score = res["candidate_score"]

        # 4. Metrics computation
        mets = compute_metrics(pred_mask, gt)

        row = {
            "Patient_ID": f"P{rank+1:02d}",
            "Filename": fname,
            "Tumor_Class": tumor_class,
            "Plane": plane,
            "Is_Tumor": is_tumor,
            "Mean_Truth_T": round(mean_t, 4),
            "Mean_Indeterminacy_I": round(mean_i, 4),
            "Mean_Falsehood_F": round(mean_f, 4),
            "Indeterminacy_Entropy": round(entropy_i, 4),
            "Candidate_Score": round(float(cand_score), 4),
            "Dice_Score": round(float(mets["dice"]), 4) if pd.notna(mets["dice"]) else np.nan,
            "IoU_Score": round(float(mets["iou"]), 4) if pd.notna(mets["iou"]) else np.nan,
            "Sensitivity": round(float(mets["sensitivity"]), 4) if pd.notna(mets["sensitivity"]) else np.nan,
            "Precision": round(float(mets["precision"]), 4) if pd.notna(mets["precision"]) else np.nan,
            "HD95_mm": round(float(mets["hd95"]), 2) if pd.notna(mets["hd95"]) else np.nan,
            "False_Alarm": mets["false_alarm"],
        }
        records.append(row)

        # Store 6 representative cases for the visual panel (2 glioma, 2 meningioma, 1 pituitary, 1 no_tumor)
        if len(visual_cases) < 6:
            visual_cases.append({
                "patient_id": f"P{rank+1:02d}",
                "class": tumor_class,
                "plane": plane,
                "dice": mets["dice"],
                "img": img,
                "gt": gt,
                "t_star": t_star,
                "i_star": i_star,
                "f_star": f_star,
                "overlay": overlay_rgb,
            })

    # Save detailed 50-patient CSV
    df_50 = pd.DataFrame(records)
    csv_path = results_dir / "project_review_50_patients.csv"
    df_50.to_csv(csv_path, index=False)
    print(f"\nDetailed 50-patient readouts saved to: {csv_path}")

    # Summary by Class
    tumor_df = df_50[df_50["Is_Tumor"]].copy()
    summary_by_class = tumor_df.groupby("Tumor_Class").agg(
        Count=("Patient_ID", "count"),
        Mean_Dice=("Dice_Score", "mean"),
        Median_Dice=("Dice_Score", "median"),
        Mean_IoU=("IoU_Score", "mean"),
        Mean_T=("Mean_Truth_T", "mean"),
        Mean_I=("Mean_Indeterminacy_I", "mean"),
        Mean_F=("Mean_Falsehood_F", "mean"),
        Indeterminacy_Entropy=("Indeterminacy_Entropy", "mean"),
    ).reset_index()

    summary_csv_path = outputs_dir / "project_review_50_summary.csv"
    summary_by_class.to_csv(summary_csv_path, index=False)
    print(f"Summary table saved to: {summary_csv_path}")

    print("\n" + "=" * 80)
    print("PHASE 1 REVIEW: 50-PATIENT BENCHMARK SUMMARY")
    print("=" * 80)
    print(summary_by_class.to_string(index=False))
    print("=" * 80)

    # Generate 6-case Visual Panel
    panel_path = outputs_dir / "phase1_50_patients_panel.png"
    generate_50_visual_panel(visual_cases, panel_path)


def generate_50_visual_panel(cases, save_path):
    """Generates a high-resolution 6-case multi-column panel showing:
    Original MRI | Truth (T*) | Indeterminacy (I*) | Falsehood (F*) | Ground Truth | Segmented Overlay
    """
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 6, figsize=(18, 3.2 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, 0)

    for i, c in enumerate(cases):
        d_str = f"Dice: {c['dice']:.3f}" if pd.notna(c['dice']) else "Neg Control"

        # 0: Original MRI
        axes[i, 0].imshow(c["img"], cmap="gray")
        axes[i, 0].set_ylabel(f"{c['patient_id']}: {c['class']}\n({c['plane']})\n{d_str}", fontsize=9, fontweight="bold")
        if i == 0:
            axes[i, 0].set_title("1. Original MRI", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 1: T*
        axes[i, 1].imshow(c["t_star"], cmap="hot")
        if i == 0:
            axes[i, 1].set_title("2. Truth (T*)", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # 2: I*
        axes[i, 2].imshow(c["i_star"], cmap="inferno")
        if i == 0:
            axes[i, 2].set_title("3. Indeterminacy (I*)", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

        # 3: F*
        axes[i, 3].imshow(c["f_star"], cmap="viridis")
        if i == 0:
            axes[i, 3].set_title("4. Falsehood (F*)", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

        # 4: Ground Truth
        axes[i, 4].imshow(c["gt"], cmap="Greens")
        if i == 0:
            axes[i, 4].set_title("5. Radiologist GT", fontsize=11, fontweight="bold")
        axes[i, 4].axis("off")

        # 5: Overlay
        axes[i, 5].imshow(c["overlay"])
        if i == 0:
            axes[i, 5].set_title("6. Clinical Overlay", fontsize=11, fontweight="bold")
        axes[i, 5].axis("off")

    plt.suptitle("Phase 1 Project Evaluation: Neutrosophic (T, I, F) Decomposition & Clinical Overlays", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visual panel to: {save_path}")


if __name__ == "__main__":
    run_50_patient_evaluation()
