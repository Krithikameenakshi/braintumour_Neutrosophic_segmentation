"""Phase 0: Data Inspection and Stratified Tuning Subset Generation.

Inspects BRISC 2025 dataset:
- Image sizes, bit depth, mask binarity, skull/scalp presence
- Mask area distributions and class/plane breakdowns
- Generates 30-sample visual grid (image + overlay)
- Extracts a 100-image stratified tuning subset from the TRAIN split only (fixed seed 42)
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import BRISCDataset


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase0():
    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 0: DATA INSPECTION - BRISC 2025")
    print("=" * 70)

    # 1. Load full dataset metadata
    full_ds = BRISCDataset(root_dir=data_dir, split="all", include_no_tumor=True)
    df_meta = full_ds.get_metadata_df()
    print(f"Total samples loaded across all splits: {len(df_meta)}")
    print(f"Split breakdown:\n{df_meta['split'].value_counts()}\n")

    # Save complete dataset inventory
    inventory_path = outputs_dir / "phase0_dataset_inventory.csv"
    df_meta.to_csv(inventory_path, index=False)
    print(f"Saved complete inventory to {inventory_path}")

    # 2. Detailed inspection of sizes, bit depth, masks, skull
    print("\n--- Performing Data & Mask Verification ---")
    sizes = set()
    dtypes = set()
    mask_unique_vals = set()
    non_binary_masks = []
    tumor_areas_px = []
    tumor_areas_pct = []

    # Check first 500 samples across splits for intensive checks + all masks
    print(f"Scanning samples for image properties and mask binarity...")
    sample_indices = range(len(full_ds))

    # Track skull / edge intensity in a sample of images
    all_mask_px = []
    all_mask_pct = []
    edge_intensities = []

    for i in sample_indices:
        sample_info = full_ds.samples[i]
        img_path = sample_info["image_path"]
        mask_path = sample_info["mask_path"]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        sizes.add(img.shape)
        dtypes.add(str(img.dtype))

        # Check skull/scalp: inspect intensity around outer margin (first 10 pixels from boundary)
        if i < 200:
            h, w = img.shape[:2]
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[:10, :] = True
            border_mask[-10:, :] = True
            border_mask[:, :10] = True
            border_mask[:, -10:] = True
            edge_intensities.append(float(np.mean(img[border_mask])))

        if mask_path is not None and Path(mask_path).exists():
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            u_vals = np.unique(mask_raw)
            for v in u_vals:
                mask_unique_vals.add(int(v))
            if not set(u_vals).issubset({0, 255}) and not set(u_vals).issubset({0, 1}):
                non_binary_masks.append((sample_info["filename"], u_vals.tolist()))

            # Mask area
            bin_mask = (mask_raw > 127).astype(np.uint8)
            area_px = int(np.sum(bin_mask))
            total_px = bin_mask.shape[0] * bin_mask.shape[1]
            area_pct = 100.0 * area_px / total_px
            tumor_areas_px.append(area_px)
            tumor_areas_pct.append(area_pct)
            all_mask_px.append(area_px)
            all_mask_pct.append(area_pct)
        else:
            # No tumor
            all_mask_px.append(0)
            all_mask_pct.append(0.0)

    df_meta["mask_area_px"] = all_mask_px
    df_meta["mask_area_pct"] = all_mask_pct

    print(f"Unique image dimensions: {sizes}")
    print(f"Unique image dtypes: {dtypes}")
    print(f"Unique raw mask values encountered: {sorted(list(mask_unique_vals))}")
    print(f"Non-binary masks count: {len(non_binary_masks)}")
    if len(non_binary_masks) > 0:
        print(f"Sample non-binary mask: {non_binary_masks[0]}")

    # Class and plane counts
    print("\n--- Class Counts by Split ---")
    class_split_ct = pd.crosstab(df_meta["tumor_class"], df_meta["split"])
    print(class_split_ct)

    print("\n--- Plane Counts by Split ---")
    plane_split_ct = pd.crosstab(df_meta["plane"], df_meta["split"])
    print(plane_split_ct)

    print("\n--- Class x Plane Counts (TRAIN split) ---")
    train_meta = df_meta[df_meta["split"] == "train"]
    class_plane_train = pd.crosstab(train_meta["tumor_class"], train_meta["plane"])
    print(class_plane_train)

    print("\n--- Class x Plane Counts (TEST split) ---")
    test_meta = df_meta[df_meta["split"] == "test"]
    class_plane_test = pd.crosstab(test_meta["tumor_class"], test_meta["plane"])
    print(class_plane_test)

    # Mask area statistics (for tumor slices)
    tumor_df = df_meta[df_meta["tumor_class"] != "no_tumor"]
    print("\n--- Mask Area Statistics (Tumor cases only) ---")
    print(f"Mean tumor area: {tumor_df['mask_area_px'].mean():.1f} px ({tumor_df['mask_area_pct'].mean():.2f}%)")
    print(f"Median tumor area: {tumor_df['mask_area_px'].median():.1f} px ({tumor_df['mask_area_pct'].median():.2f}%)")
    print(f"Min tumor area: {tumor_df['mask_area_px'].min():.1f} px ({tumor_df['mask_area_pct'].min():.4f}%)")
    print(f"Max tumor area: {tumor_df['mask_area_px'].max():.1f} px ({tumor_df['mask_area_pct'].max():.2f}%)")

    area_by_class = tumor_df.groupby("tumor_class")["mask_area_pct"].describe()
    print("\n--- Tumor Area (%) by Class ---")
    print(area_by_class)

    # Save summary tables to CSV
    class_split_ct.to_csv(outputs_dir / "phase0_class_by_split.csv")
    plane_split_ct.to_csv(outputs_dir / "phase0_plane_by_split.csv")
    class_plane_train.to_csv(outputs_dir / "phase0_class_plane_train.csv")
    area_by_class.to_csv(outputs_dir / "phase0_tumor_area_by_class.csv")

    # Plot mask area distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(tumor_df["mask_area_pct"], bins=50, color="#1f77b4", edgecolor="black", alpha=0.75)
    axes[0].set_title("Tumor Mask Area Distribution (% of Slice)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Area (% of total image pixels)")
    axes[0].set_ylabel("Number of Slices")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Boxplot by class
    classes_order = ["glioma", "meningioma", "pituitary"]
    data_by_c = [tumor_df[tumor_df["tumor_class"] == c]["mask_area_pct"] for c in classes_order]
    axes[1].boxplot(data_by_c, tick_labels=classes_order, patch_artist=True, boxprops=dict(facecolor="#2ca02c", alpha=0.6))
    axes[1].set_title("Tumor Area (%) by Tumor Type", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Area (% of total image pixels)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outputs_dir / "phase0_mask_area_distribution.png", dpi=200)
    plt.close()
    print(f"Saved mask area plot to {outputs_dir / 'phase0_mask_area_distribution.png'}")

    # 3. Make a grid of 30 random samples (image + mask overlay) across classes and planes
    print("\n--- Generating 30-Sample Visual Grid ---")
    np.random.seed(cfg["experiment"]["seed"])

    # Sample 30 images covering all classes and planes
    sample_picks = []
    # 4 classes x 3 planes = 12 strata. Take 3 from 6 strata, 2 from 6 strata = 30 total
    grouped = df_meta[df_meta["split"] == "train"].groupby(["tumor_class", "plane"])
    strata_list = list(grouped.groups.keys())
    # First 6 strata get 3, remaining 6 strata get 2 -> exactly 30
    for idx_st, (cls, pln) in enumerate(strata_list):
        grp = grouped.get_group((cls, pln))
        quota = 3 if idx_st < 6 else 2
        sample_subset = grp.sample(n=quota, random_state=cfg["experiment"]["seed"])
        sample_picks.append(sample_subset)

    grid_df = pd.concat(sample_picks).sample(frac=1.0, random_state=cfg["experiment"]["seed"]).reset_index(drop=True)
    assert len(grid_df) == 30, f"Expected 30 samples, got {len(grid_df)}"
    grid_df.to_csv(outputs_dir / "phase0_grid_30_metadata.csv", index=False)

    # Render 5 rows x 6 columns grid
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.flatten()

    for idx, row in grid_df.iterrows():
        ax = axes[idx]
        img_p = row["image_path"]
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        
        # Overlay
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if row["mask_path"] is not None and pd.notna(row["mask_path"]) and Path(str(row["mask_path"])).exists():
            mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask_bin = (mask > 127).astype(np.uint8)
            # Translucent red overlay
            red_layer = np.zeros_like(rgb)
            red_layer[:, :, 0] = 255  # Red
            mask_3c = np.stack([mask_bin]*3, axis=-1)
            rgb = np.where(mask_3c == 1, cv2.addWeighted(rgb, 0.65, red_layer, 0.35, 0), rgb)
            # Find contours and draw red outline
            cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, cnts, -1, (255, 0, 0), 2)
            title_tag = f"{row['tumor_class']}\n{row['plane']} (area: {row['mask_area_pct']:.1f}%)"
        else:
            title_tag = f"{row['tumor_class']}\n{row['plane']} (no tumor)"

        ax.imshow(rgb)
        ax.set_title(title_tag, fontsize=8, fontweight="bold")
        ax.axis("off")

    plt.suptitle("BRISC 2025: 30 Representative Samples with Radiologist Masks (Red Overlay/Contour)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    grid_plot_path = outputs_dir / "phase0_grid_30_samples.png"
    plt.savefig(grid_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 30-sample visual grid to {grid_plot_path}")

    # 4. Create tuning subset of 100 TRAIN images, stratified by class and plane (fixed seed)
    print("\n--- Creating Stratified 100-Image TRAIN Tuning Subset ---")
    train_df = df_meta[df_meta["split"] == "train"].copy().reset_index(drop=True)
    target_n = cfg["experiment"]["tuning_sample_size"]
    seed = cfg["experiment"]["seed"]

    # Calculate proportional allocation per stratum (tumor_class x plane)
    strata_counts = train_df.groupby(["tumor_class", "plane"]).size()
    total_train = len(train_df)
    strata_proportions = strata_counts / total_train
    strata_alloc = (strata_proportions * target_n).round().astype(int)

    # Adjust rounding differences so sum is exactly target_n
    diff = target_n - strata_alloc.sum()
    if diff != 0:
        # adjust largest stratum
        max_stratum = strata_alloc.idxmax()
        strata_alloc[max_stratum] += diff

    print(f"Stratified sample allocation across 12 (class, plane) strata (Total={target_n}):")
    for stratum, alloc_n in strata_alloc.items():
        print(f"  {stratum}: {alloc_n} samples (from {strata_counts[stratum]} total)")

    # Sample deterministically
    sampled_dfs = []
    for (cls, pln), alloc_n in strata_alloc.items():
        stratum_df = train_df[(train_df["tumor_class"] == cls) & (train_df["plane"] == pln)]
        sampled = stratum_df.sample(n=alloc_n, random_state=seed)
        sampled_dfs.append(sampled)

    tuning_df = pd.concat(sampled_dfs).sort_values("filename").reset_index(drop=True)

    # Verification: check no TEST samples are present
    assert (tuning_df["split"] == "train").all(), "FATAL ERROR: Test split leaked into tuning subset!"
    assert len(tuning_df) == target_n, f"Expected {target_n} samples, got {len(tuning_df)}"

    tuning_csv_path = Path(cfg["paths"]["tuning_subset_path"])
    tuning_df.to_csv(tuning_csv_path, index=False)
    tuning_df.to_csv(outputs_dir / "tuning_subset_100_summary.csv", index=False)
    print(f"\nSuccessfully created and saved tuning subset to {tuning_csv_path}")
    print("Tuning subset breakdown:")
    print(pd.crosstab(tuning_df["tumor_class"], tuning_df["plane"], margins=True))

    print("\nPhase 0 completed successfully!")


if __name__ == "__main__":
    run_phase0()
