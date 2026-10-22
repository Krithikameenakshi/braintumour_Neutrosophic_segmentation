# Unsupervised Training-Free Brain Tumor Segmentation Using Neutrosophic Sets $(T, I, F)$

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: BRISC 2025](https://img.shields.io/badge/Dataset-BRISC_2025-green.svg)](https://www.kaggle.com/)

A mathematically rigorous, completely unsupervised, and **training-free** framework for brain tumor segmentation from T1-weighted contrast-enhanced (T1ce) MRI scans, developed and benchmarked on the **BRISC 2025** dataset (6,000 images: 5,000 train, 1,000 test across glioma, meningioma, pituitary, and healthy brain controls).

---

## 📌 Key Architectural Innovations

Unlike deep learning approaches (e.g., U-Net, nnU-Net) that require thousands of labeled training volumes and GPU accelerators, this framework operates **deterministically with zero trained weights**:

1. **Robust Preprocessing & Parenchyma Isolation**:
   - Adaptive percentile normalization $[P_1, P_{99}]$.
   - Otsu head mask extraction and Euclidean distance transform erosion with parameter $r_s = 0.06 \cdot \text{width}$ to suppress scalp, subcutaneous fat, and diploic marrow hyperintensities.
   - $3 \times 3$ median filter followed by Perona-Malik anisotropic diffusion to preserve edge boundaries while homogenizing tissue regions.
2. **Multi-Scale Gradient-Aware Neutrosophic Transform**:
   - Projects intensity into three orthocomplementary neutrosophic sets:
     - **Truth ($T^*$)**: Normalized pixel intensity within brain parenchyma.
     - **Indeterminacy ($I^*$)**: Local boundary fuzziness combining normalized local entropy and Scharr gradient magnitude.
     - **Falsehood ($F^*$)**: Background/healthy tissue certainty ($1 - T^*$).
3. **Directional $\alpha$-Mean Filtering (DMF)**:
   - 4-orientation directional filter ($\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$) to preserve linear tumor margins while reducing speckle noise, with isotropic fallback for isotropic slices.
4. **Adaptive Multi-Otsu S-Function Contrast Enhancement**:
   - Dynamic 3-level Otsu threshold decomposition $[\tau_1, \tau_2]$ mapping indeterminacy to high-contrast steep sigmoidal responses.
5. **Hemispheric Asymmetry & Geometric Candidate Scoring**:
   - Computes bilateral hemispheric reflection across the interhemispheric fissure ($g_{\text{sym}}$), measuring asymmetric hyperintensity ($Asym$).
   - Multi-feature ranking incorporating region solidity, normalized intensity, and dural margin proximity.
   - Strict rejection threshold ($\tau_{\text{rej}} = 0.50$) to suppress physiological false positives on negative control slices.
6. **Marker-Controlled Watershed on Indeterminacy Landscape**:
   - Internal topological seeds from high-$T^*$ candidate cores.
   - External background markers from eroded brain margins.
   - Flooding driven by the hybrid gradient and boundary indeterminacy landscape $L = 0.5 \cdot \hat{G} + 0.5 \cdot I^*$.

---

## 📂 Repository Structure

```
fyp/
├── configs/
│   ├── default.yaml            # Base configuration
│   └── final.yaml              # Optimal frozen hyperparameters (rs=0.06, tau_rej=0.50)
├── data/
│   ├── brisc/                  # BRISC 2025 dataset (train: 5000, test: 1000)
│   └── tuning_subset_100.csv   # Stratified 100-sample training tuning subset
├── src/
│   ├── __init__.py
│   ├── dataset.py              # BRISC 2025 dataset loader & metadata parser
│   ├── preprocess.py           # Skull stripping, normalization, anisotropic diffusion
│   ├── ns.py                   # Neutrosophic transforms (T, I, F) & entropy
│   ├── baselines.py            # Otsu, k-Means, FCM, Guo-Cheng NS, Tufail NS ROI
│   ├── pipeline.py             # Steps A-G full proposed pipeline
│   ├── metrics.py              # Clinical metrics (Dice, IoU, Sens, Prec, FNR, HD95)
│   └── run.py                  # Production CLI for single-slice inference
├── scripts/
│   ├── phase0_inspect_data.py  # Dataset validation and stratified subset generation
│   ├── phase1_eval_ns.py       # Preprocessing & NS transform diagnostic panels
│   ├── phase2_eval_baselines.py# Baseline benchmarking on tuning subset
│   ├── phase3_eval_tuning.py   # Full pipeline evaluation & best/worst case panels
│   ├── phase4_tune.py          # Systematic grid search over hyperparameters
│   ├── phase4_ablations.py     # Systematic 6-condition ablation study
│   └── phase4_eval_test.py     # Final evaluation on 1,000 untouched test slices
├── tests/
│   ├── test_ns.py              # Unit tests for neutrosophic transforms
│   └── test_metrics.py         # Unit tests for clinical metric calculations
├── outputs/                    # Diagnostic visualization grids, PNGs, and metric tables
├── results/                    # CSV evaluation logs and detailed slice readouts
├── LIMITATIONS.md              # Deep analysis of clinical & algorithmic failure modes
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-neutrosophic-segmentation.git
cd brain-tumor-neutrosophic-segmentation

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `numpy`, `scipy`, `pandas`
- `opencv-python`, `scikit-image`, `scikit-fuzzy`
- `matplotlib`, `seaborn`
- `pyyaml`, `tqdm`, `pytest`

---

## 💻 CLI Usage (Single Slice Inference)

Run tumor segmentation on any brain MRI scan:

```bash
# Basic inference using default configuration
python -m src.run --image path/to/mri_slice.jpg --output-dir outputs/demo/

# Advanced inference with custom frozen config and ground-truth evaluation
python -m src.run \
    --image data/brisc/segmentation_task/test/images/brisc2025_test_00001_gl_ax_t1.jpg \
    --mask data/brisc/segmentation_task/test/masks/brisc2025_test_00001_gl_ax_t1.png \
    --config configs/final.yaml \
    --output-dir outputs/demo/
```

**Outputs generated:**
- Binary segmentation mask: `outputs/demo/<slice_stem>_mask.png`
- Visual overlay (Original, $T^*$, $I^*$, Ground Truth, Prediction): `outputs/demo/<slice_stem>_overlay.png`
- Detailed numeric readouts (Dice, IoU, area, centroid, entropy): `outputs/demo/<slice_stem>_readouts.json`

---

## 🧪 Reproducing Experiments & Benchmark Results

### 1. Run Unit Tests
```bash
pytest tests/ -v
```

### 2. Run Preprocessing & Neutrosophic Panel (Phase 1)
```bash
python scripts/phase1_eval_ns.py
```

### 3. Evaluate Classical Baselines (Phase 2)
```bash
python scripts/phase2_eval_baselines.py
```

### 4. Full Pipeline Tuning & Visual Diagnostic Panels (Phase 3)
```bash
python scripts/phase3_eval_tuning.py
```

### 5. Systematic Hyperparameter Tuning & Ablation Study (Phase 4)
```bash
# Run grid search over rs and rejection threshold
python scripts/phase4_tune.py

# Run systematic ablations (a-f)
python scripts/phase4_ablations.py
```

### 6. Benchmark on Untouched 1,000-Image TEST Split (Phase 4 Final)
```bash
python scripts/phase4_eval_test.py
```

---

## 📊 Summary of Benchmark Findings

1. **Massive Reduction in False Alarms**:
   - Classical thresholding (Otsu, $k$-Means, FCM) suffers from a **$90.5\%$ False Alarm Rate** on healthy brain controls because normal hyperintense structures (dural margins, choroid plexus) are always labeled as tumors.
   - The proposed framework reduces False Alarm Rate to **$0.0\%$** through adaptive multi-feature candidate scoring and hemispheric symmetry priors.
2. **Superior Overlap on Solid Tumors**:
   - Reaches up to **$0.965$ Dice** on solid meningiomas and well-circumscribed gliomas.
   - Achieves a $+69\%$ improvement in mean Dice over Otsu and $+132\%$ over $k$-Means and FCM across all cases.
3. **Statistical Significance**:
   - Paired Wilcoxon signed-rank tests confirm statistically significant superiority over all five baseline methods ($p < 10^{-4}$).

For full details on failure modes (pituitary sinus inversion, necrotic cores, scalp fat margins), see [LIMITATIONS.md](LIMITATIONS.md).
