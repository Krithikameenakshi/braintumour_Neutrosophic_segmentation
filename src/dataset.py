"""BRISC 2025 Dataset Loader.

Provides a robust loader that returns (image, mask, class, plane) per sample,
handling both segmentation and no-tumor classification samples.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd


# Mapping dictionary for short codes in filenames
TUMOR_CODE_MAP = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "nt": "no_tumor",
}

PLANE_CODE_MAP = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}


def parse_filename(filename: str) -> Dict[str, str]:
    """Parses standard BRISC filename convention.

    Example: brisc2025_train_00001_gl_ax_t1.jpg
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    # Expected format: [brisc2025, split, index, tumor_code, plane_code, sequence]
    if len(parts) >= 6:
        split = parts[1]
        index_str = parts[2]
        tumor_code = parts[3]
        plane_code = parts[4]
        sequence = parts[5]
        return {
            "split": split,
            "index": index_str,
            "tumor_code": tumor_code,
            "tumor_class": TUMOR_CODE_MAP.get(tumor_code, tumor_code),
            "plane_code": plane_code,
            "plane": PLANE_CODE_MAP.get(plane_code, plane_code),
            "sequence": sequence,
        }
    return {}


class BRISCDataset:
    """Dataset loader for BRISC 2025 brain MRI dataset."""

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/brisc",
        split: str = "train",
        include_no_tumor: bool = True,
    ):
        """Args:

        root_dir: Base directory containing BRISC data (manifest.csv,
        segmentation_task, classification_task).
        split: 'train', 'test', or 'all'.
        include_no_tumor: Whether to include no-tumor negative cases (masks =
        all zeros).
        """
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.include_no_tumor = include_no_tumor
        self.samples: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        manifest_path = self.root_dir / "manifest.csv"

        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            # Filter by split if specified
            if self.split != "all":
                df = df[df["split"] == self.split]

            # 1. Gather all segmentation images
            seg_images = df[(df["task"] == "segmentation") & (~df["is_mask"])].copy()

            for _, row in seg_images.iterrows():
                img_path = self.root_dir / row["relative_path"]
                # Mask filename matches image filename stem with .png extension
                mask_rel = f"segmentation_task/{row['split']}/masks/{Path(row['filename']).stem}.png"
                mask_path = self.root_dir / mask_rel

                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "filename": row["filename"],
                        "tumor_class": row["tumor_label"],
                        "plane": row["plane_label"],
                        "split": row["split"],
                        "is_tumor": True,
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                    }
                )

            # 2. Gather no_tumor images if requested
            if self.include_no_tumor:
                nt_images = df[
                    (df["task"] == "classification")
                    & (df["tumor_label"] == "no_tumor")
                ].copy()
                for _, row in nt_images.iterrows():
                    img_path = self.root_dir / row["relative_path"]
                    self.samples.append(
                        {
                            "image_path": str(img_path),
                            "mask_path": None,  # Ground truth is empty mask
                            "filename": row["filename"],
                            "tumor_class": "no_tumor",
                            "plane": row["plane_label"],
                            "split": row["split"],
                            "is_tumor": False,
                            "width": int(row["width"]),
                            "height": int(row["height"]),
                        }
                    )
        else:
            # Fallback direct disk scanning if manifest is missing
            self._scan_filesystem()

        # Sort for determinism
        self.samples.sort(key=lambda s: s["filename"])

    def _scan_filesystem(self) -> None:
        """Direct directory scanner fallback."""
        splits_to_scan = ["train", "test"] if self.split == "all" else [self.split]
        for sp in splits_to_scan:
            img_dir = self.root_dir / "segmentation_task" / sp / "images"
            mask_dir = self.root_dir / "segmentation_task" / sp / "masks"
            if img_dir.exists():
                for img_p in img_dir.glob("*.jpg"):
                    mask_p = mask_dir / f"{img_p.stem}.png"
                    meta = parse_filename(img_p.name)
                    self.samples.append(
                        {
                            "image_path": str(img_p),
                            "mask_path": str(mask_p) if mask_p.exists() else None,
                            "filename": img_p.name,
                            "tumor_class": meta.get("tumor_class", "unknown"),
                            "plane": meta.get("plane", "unknown"),
                            "split": sp,
                            "is_tumor": meta.get("tumor_class") != "no_tumor",
                        }
                    )

            if self.include_no_tumor:
                nt_dir = self.root_dir / "classification_task" / sp / "no_tumor"
                if nt_dir.exists():
                    for img_p in nt_dir.glob("*.jpg"):
                        meta = parse_filename(img_p.name)
                        self.samples.append(
                            {
                                "image_path": str(img_p),
                                "mask_path": None,
                                "filename": img_p.name,
                                "tumor_class": "no_tumor",
                                "plane": meta.get("plane", "unknown"),
                                "split": sp,
                                "is_tumor": False,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Returns:

        (image, mask, class, plane)
        - image: 2D uint8 numpy array [0, 255]
        - mask: 2D uint8 binary mask [0, 1]
        - class: str (glioma, meningioma, pituitary, no_tumor)
        - plane: str (axial, coronal, sagittal)
        """
        sample = self.samples[idx]
        img = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {sample['image_path']}")

        if sample["mask_path"] is not None and Path(sample["mask_path"]).exists():
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask at {sample['mask_path']}")
            # Ensure strictly binary {0, 1}
            mask = (mask > 127).astype(np.uint8)
        else:
            # For no-tumor cases, ground truth is strictly an empty binary mask
            mask = np.zeros_like(img, dtype=np.uint8)

        return img, mask, sample["tumor_class"], sample["plane"]

    def get_metadata_df(self) -> pd.DataFrame:
        """Returns dataframe of all loaded samples with metadata."""
        return pd.DataFrame(self.samples)
