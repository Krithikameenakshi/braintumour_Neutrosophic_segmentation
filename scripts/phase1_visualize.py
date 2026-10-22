"""Phase 1 Visualization Script.

Demonstrates Preprocessing and Neutrosophic Transformation across 10 diverse
tuning samples:
Displays: Original | Head Mask (H) | Brain Mask (M_b) | T | I | F
Saves high-res panel to outputs/phase1_10_samples_visualization.png
and tabular metrics to outputs/phase1_brain_mask_metrics.csv.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset
from src.preprocess import preprocess_image
from src.ns import neutrosophic_transform, compute_entropy


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase1_visualize():
    cfg = load_config()
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 1: PREPROCESSING & NEUTROSOPHIC SET (T, I, F) VISUALIZATION")
    print("=" * 75)

    # Load tuning subset to pick 10 representative images
    tuning_csv_path = Path(cfg["paths"]["tuning_subset_path"])
    if not tuning_csv_path.exists():
        raise FileNotFoundError(f"Tuning subset not found at {tuning_csv_path}. Run Phase 0 first.")

    tuning_df = pd.read_csv(tuning_csv_path)

    # Pick 10 diverse samples across tumor classes and planes
    picks = [
        # Glioma: axial, coronal, sagittal
        tuning_df[(tuning_df["tumor_class"] == "glioma") & (tuning_df["plane"] == "axial")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "glioma") & (tuning_df["plane"] == "coronal")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "glioma") & (tuning_df["plane"] == "sagittal")].iloc[0],
        # Meningioma: axial, coronal, sagittal
        tuning_df[(tuning_df["tumor_class"] == "meningioma") & (tuning_df["plane"] == "axial")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "meningioma") & (tuning_df["plane"] == "coronal")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "meningioma") & (tuning_df["plane"] == "sagittal")].iloc[0],
        # Pituitary: axial, coronal
        tuning_df[(tuning_df["tumor_class"] == "pituitary") & (tuning_df["plane"] == "axial")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "pituitary") & (tuning_df["plane"] == "coronal")].iloc[0],
        # No tumor: axial, sagittal
        tuning_df[(tuning_df["tumor_class"] == "no_tumor") & (tuning_df["plane"] == "axial")].iloc[0],
        tuning_df[(tuning_df["tumor_class"] == "no_tumor") & (tuning_df["plane"] == "sagittal")].iloc[0],
    ]

    selected_df = pd.DataFrame(picks).reset_index(drop=True)
    print(f"Selected {len(selected_df)} representative images across classes and anatomical planes.\n")

    # Prepare figure: 10 rows x 6 columns
    fig, axes = plt.subplots(10, 6, figsize=(18, 30))
    metrics_records = []

    for i, row in selected_df.iterrows():
        img_path = row["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        # Preprocessing
        g_norm, head_mask, brain_mask, g_denoised = preprocess_image(img, cfg)

        # Neutrosophic transform
        w_size = cfg["neutrosophic"]["window_size"]
        T, I, F = neutrosophic_transform(g_denoised, window_size=w_size)

        # Entropies within brain mask M_b
        n_bins = cfg["neutrosophic"]["entropy_bins"]
        en_t = compute_entropy(T, mask=brain_mask, bins=n_bins)
        en_i = compute_entropy(I, mask=brain_mask, bins=n_bins)
        en_f = compute_entropy(F, mask=brain_mask, bins=n_bins)

        # Metrics
        head_px = int(np.sum(head_mask))
        brain_px = int(np.sum(brain_mask))
        total_px = h * w
        head_pct = 100.0 * head_px / total_px
        brain_pct = 100.0 * brain_px / total_px
        retention_pct = 100.0 * brain_px / max(1, head_px)

        metrics_records.append({
            "sample_index": i + 1,
            "filename": row["filename"],
            "class": row["tumor_class"],
            "plane": row["plane"],
            "image_size": f"{w}x{h}",
            "head_mask_pct": round(head_pct, 2),
            "brain_mask_pct": round(brain_pct, 2),
            "parenchyma_retention_pct": round(retention_pct, 2),
            "En_T": round(en_t, 4),
            "En_I": round(en_i, 4),
            "En_F": round(en_f, 4),
        })

        # Plot 6 columns for this row
        # Col 0: Original
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_ylabel(f"#{i+1}: {row['tumor_class']}\n({row['plane']})", fontsize=10, fontweight="bold")
        if i == 0:
            axes[i, 0].set_title("1. Original MRI", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Col 1: Head Mask H
        axes[i, 1].imshow(head_mask, cmap="Blues")
        if i == 0:
            axes[i, 1].set_title("2. Head Mask (H)", fontsize=11, fontweight="bold")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Col 2: Brain Mask M_b (overlayed on original)
        brain_overlay = cv2.cvtColor((g_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        green_tint = np.zeros_like(brain_overlay)
        green_tint[:, :, 1] = 255
        brain_3c = np.stack([brain_mask.astype(np.uint8)] * 3, axis=-1)
        brain_overlay = np.where(brain_3c == 1, cv2.addWeighted(brain_overlay, 0.65, green_tint, 0.35, 0), brain_overlay)
        cnts, _ = cv2.findContours(brain_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(brain_overlay, cnts, -1, (0, 255, 0), 1)

        axes[i, 2].imshow(brain_overlay)
        if i == 0:
            axes[i, 2].set_title("3. Brain Mask M_b (Green)", fontsize=11, fontweight="bold")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Col 3: T map
        axes[i, 3].imshow(T, cmap="hot")
        if i == 0:
            axes[i, 3].set_title("4. True Map (T)", fontsize=11, fontweight="bold")
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

        # Col 4: I map
        axes[i, 4].imshow(I, cmap="inferno")
        if i == 0:
            axes[i, 4].set_title("5. Indeterminacy (I)", fontsize=11, fontweight="bold")
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])

        # Col 5: F map
        axes[i, 5].imshow(F, cmap="viridis")
        if i == 0:
            axes[i, 5].set_title("6. False Map (F)", fontsize=11, fontweight="bold")
        axes[i, 5].set_xticks([])
        axes[i, 5].set_yticks([])

    plt.suptitle("Phase 1: Preprocessing & Neutrosophic Transformation (10 Representative Slices)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plot_path = outputs_dir / "phase1_10_samples_visualization.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved high-resolution visualization panel to: {plot_path}")

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = outputs_dir / "phase1_brain_mask_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved quantitative metrics table to: {metrics_csv_path}\n")

    print("--- Quantitative Preprocessing & Neutrosophic Metrics ---")
    print(metrics_df.to_string(index=False))
    print("\nPhase 1 visualization completed successfully!")


if __name__ == "__main__":
    run_phase1_visualize()
