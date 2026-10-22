"""Command Line Interface (CLI) for Brain Tumor Segmentation.

Usage:
  python -m src.run --image data/brisc/segmentation_task/train/images/brisc2025_train_00001_gl_ax_t1.jpg --out outputs/demo/
"""

import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import run_neutrosophic_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Training-Free Neutrosophic Brain Tumor Segmentation CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input 2D brain MRI slice (JPG or PNG)")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground truth binary mask (PNG)")
    parser.add_argument("--out", "--output-dir", dest="out", type=str, default="outputs/demo/", help="Output directory to save results")
    parser.add_argument("--config", type=str, default="configs/final.yaml", help="Path to config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found at {img_path}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("NEUTROSOPHIC BRAIN TUMOR SEGMENTATION ENGINE (TRAINING-FREE)")
    print("=" * 70)
    print(f"Loading input slice: {img_path.name}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")

    h, w = img.shape
    print(f"Image Resolution:    {w} x {h} (8-bit grayscale)")

    # Run full neutrosophic pipeline
    print("\nRunning Neutrosophic Segmentation Pipeline...")
    res = run_neutrosophic_segmentation(img, cfg)

    mask = res["mask"]
    readouts = res["readouts"]
    overlay = readouts["overlay_rgb"]
    t_star = res["T_star"]
    i_star = res["I_star"]
    f_star = res["F_star"]

    # 1. Save binary mask
    mask_path = out_dir / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    # 2. Save overlay (translucent highlight + 2px red contour)
    overlay_path = out_dir / f"{img_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 3. Save 6-panel diagnostic visualization
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("1. Original MRI", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(t_star, cmap="hot")
    axes[1].set_title("2. Truth (T*)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(i_star, cmap="inferno")
    axes[2].set_title("3. Indeterminacy (I*)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(f_star, cmap="viridis")
    axes[3].set_title("4. Falsehood (F*)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(mask, cmap="Blues")
    axes[4].set_title("5. Binary Mask", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    axes[5].imshow(overlay)
    axes[5].set_title("6. Clinical Overlay", fontsize=11, fontweight="bold")
    axes[5].axis("off")

    status_tag = f"TUMOR DETECTED (Confidence: {readouts['region_confidence']:.2f})" if readouts["has_tumor"] else "NO TUMOR DETECTED"
    plt.suptitle(f"{img_path.name} | {status_tag}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    panel_path = out_dir / f"{img_path.stem}_panel.png"
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close()

    mets = None
    if args.mask is not None and Path(args.mask).exists():
        from src.metrics import compute_metrics
        gt_raw = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if gt_raw is not None:
            gt_bin = (gt_raw > 127).astype(np.uint8)
            mets = compute_metrics(mask, gt_bin)

    # 4. Save metadata JSON
    metrics_json_path = out_dir / f"{img_path.stem}_readouts.json"
    clean_readouts = {
        "filename": img_path.name,
        "has_tumor": readouts["has_tumor"],
        "region_confidence": readouts["region_confidence"],
        "boundary_uncertainty": readouts["boundary_uncertainty"],
        "tumor_area_pixels": readouts["tumor_area_pixels"],
        "tumor_pct_brain": readouts["tumor_pct_brain"],
        "candidate_score": round(float(res["candidate_score"]), 4),
        "polarity": res["polarity"],
    }
    if mets is not None:
        clean_readouts["metrics"] = {
            "dice": round(float(mets["dice"]), 4) if pd.notna(mets["dice"]) else None,
            "iou": round(float(mets["iou"]), 4) if pd.notna(mets["iou"]) else None,
            "sensitivity": round(float(mets["sensitivity"]), 4) if pd.notna(mets["sensitivity"]) else None,
            "precision": round(float(mets["precision"]), 4) if pd.notna(mets["precision"]) else None,
            "hd95": round(float(mets["hd95"]), 2) if pd.notna(mets["hd95"]) else None,
        }
    with open(metrics_json_path, "w") as f:
        json.dump(clean_readouts, f, indent=2)

    # Console Readout
    print("\n" + "=" * 70)
    print("CLINICAL SEGMENTATION & UNCERTAINTY REPORT")
    print("=" * 70)
    print(f"  Tumor Status:          {'DETECTED' if readouts['has_tumor'] else 'NEGATIVE (No Tumor)'}")
    print(f"  Region Confidence:     {readouts['region_confidence']:.4f} (Mean T* within mask)")
    print(f"  Boundary Uncertainty:  {readouts['boundary_uncertainty']:.4f} (Mean I* within 3px contour band)")
    print(f"  Tumor Pixel Area:      {readouts['tumor_area_pixels']} px ({readouts['tumor_pct_brain']:.2f}% of intracranial brain)")
    print(f"  Candidate Score:       {res['candidate_score']:.4f}")
    print(f"  Polarity Used:         {res['polarity']}")
    print("-" * 70)
    print(f"Saved binary mask:       {mask_path}")
    print(f"Saved clinical overlay:  {overlay_path}")
    print(f"Saved diagnostic panel:  {panel_path}")
    print(f"Saved JSON readouts:     {metrics_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
