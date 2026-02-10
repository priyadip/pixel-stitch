# pixel-stitch
From-scratch panoramic stitching - manual DLT, RANSAC, inverse warping &amp; Laplacian pyramid blending. No cv2.Stitcher. Just math.

# Panoramic Image Stitching — From Scratch

A complete panoramic image stitching pipeline built from scratch using only NumPy for computation and OpenCV for SIFT detection and image I/O. No high-level stitching APIs (e.g., `cv2.Stitcher`) are used.

## Pipeline Overview

```
Input Images → SIFT Detection → Feature Matching → RANSAC Homography → Inverse Warping → Exposure Compensation → Multi-Band Blending → Crop → Panorama
```

### Key Implementations (Manual)

| Component                  | Method                                          |
|----------------------------|--------------------------------------------------|
| Homography Estimation      | Direct Linear Transform (DLT) with Hartley Normalization |
| Robust Estimation          | RANSAC (10,000 iterations, 3.0 px threshold)     |
| Image Warping              | Inverse mapping with bilinear interpolation       |
| Blending                   | Multi-band Laplacian pyramid (6 levels)           |
| Exposure Compensation      | Per-channel gain in overlap region                |
| Cropping                   | Greedy shrink to largest clean rectangle          |

### Permitted Library Usage

| Library    | Usage                                      |
|------------|--------------------------------------------|
| OpenCV     | Image I/O, SIFT detection, keypoint matching |
| NumPy      | Matrix operations, SVD, linear algebra      |
| Matplotlib | Visualization and figure export             |

---

## File Structure

```
pixel-stitch/
├── Code/
│   ├── panorama_stitch.py              # Core stitching pipeline
│   └── panorama_stitch_memsafe.py      # Memory-hardened variant
├── Result/
│   ├── AppendixA/
│   │   ├── 01_input_images.png
│   │   ├── 02_sift_keypoints.png
│   │   ├── 03_matches_0_1.png
│   │   ├── 03_matches_2_1.png
|   |   └── output/
│   │       ├── 04_naive_stitch.jpg
│   │       ├── 04_naive_stitch_cropped.jpg
│   │       ├── 05_blended_panorama.jpg
│   │       └── 06_comparison.png
│   ├── AppendixB/
│   │   └── output/
│   │       └── ...
│   ├── AppendixC/                      # ← Required memsafe pipeline
│   │   └── output/
│   │       └── ...
│   ├── AppendixD/                      # ← Required memsafe pipeline
│   │   └── output/
│   │       └── ...
│   ├── AppendixE/                      # ← Required memsafe pipeline
│   │   └── output/
│   │       └── ...
│   ├── AppendixF/
│   │   └── output/
│   │       └── ...
|   ├── Compression.py
|   └── RESULTS.md
|
└── README.md

```

---

## Usage

### Core Pipeline

```bash
python panorama_stitch.py <left.jpg> <center.jpg> <right.jpg>
```

Supply at least 3 overlapping images ordered left to right. The center image is used as the reference frame (identity homography).

### Memory-Safe Pipeline

```bash
python panorama_stitch_memsafe.py <left.jpg> <center.jpg> <right.jpg>
```

Use this variant when dealing with:
- **High-resolution images** (>4000 px per dimension)
- **Scenes with repetitive structures** (fences, blinds, tiled facades) that can cause degenerate homographies and massive canvas sizes
- **Systems with limited RAM** (<16 GB)

### Differences Between the Two Scripts

| Feature                        | `panorama_stitch.py` | `panorama_stitch_memsafe.py` |
|--------------------------------|:--------------------:|:----------------------------:|
| Core stitching logic           | ✓                    | ✓                            |
| Multi-band Laplacian blending  | ✓                    | ✓ (with memory guard)        |
| `float64` → `uint8` compression | —                  | ✓ (8× RAM savings)          |
| Predictive OOM prevention      | —                    | ✓ (pre-calculates memory)   |
| Disk-backed arrays (`memmap`)  | —                    | ✓ (for oversized canvases)  |
| Linear blend fallback          | —                    | ✓ (when pyramids exceed RAM)|

## Requirements

```
Python >= 3.8
opencv-python >= 4.5
numpy >= 1.20
matplotlib >= 3.3
```

Install dependencies:

```bash
pip install opencv-python numpy matplotlib
```

## Output Examples

The pipeline generates intermediate visualizations at each stage:

| Output File                  | Description                                      |
|------------------------------|--------------------------------------------------|
| `01_input_images.png`        | Side-by-side view of all input images            |
| `02_sift_keypoints.png`      | SIFT keypoints overlaid on each image            |
| `03_matches_*.png`           | Feature correspondences between adjacent pairs   |
| `04_naive_stitch.png`        | Direct overlay with no blending (shows seams)    |
| `04_naive_stitch_cropped.png`| Naive stitch after rectangular cropping          |
| `05_blended_panorama.png`    | Final panorama with multi-band blending          |
| `06_comparison.png`          | Naive vs. blended side-by-side comparison        |

## Known Limitations

- Assumes **pure rotational camera motion** (no translational parallax). Imperfect rotation around the optical center will cause ghosting.
- Uses **projective (planar) stitching**, which introduces bow-tie distortion for wide-angle (>120°) panoramas. Cylindrical or spherical projection would be needed for ultra-wide mosaics.
- Exposure compensation uses a **single per-channel gain** — spatially varying illumination (vignetting) is not corrected.
- Scenes dominated by **repetitive periodic structures** can produce degenerate homographies; use `panorama_stitch_memsafe.py` for such cases.

## License

This project was developed as a coursework assignment for Computer Vision.
