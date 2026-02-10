# Results

## Overview

This project contains two stitching scripts and six experiments (Appendix A–F). Appendices A, B, and F were successfully stitched using the core pipeline (`panorama_stitch.py`). Appendices C, D, and E encountered out-of-memory crashes due to degenerate homographies caused by repetitive structures in the input images — this motivated the development of `panorama_stitch_memsafe.py`, which successfully handles those cases.

| Appendix | Script Used                    | Status              |
|----------|--------------------------------|---------------------|
| A        | `panorama_stitch.py`           | ✅ Success           |
| B        | `panorama_stitch.py`           | ✅ Success           |
| C        | `panorama_stitch_memsafe.py`   | ✅ Success (after fix)|
| D        | `panorama_stitch_memsafe.py`   | ✅ Success (after fix)|
| E        | `panorama_stitch_memsafe.py`   | ✅ Success (after fix)|
| F        | `panorama_stitch.py`           | ✅ Success           |

---

## Appendix A — Nighttime Campus Scene

Stitched using `panorama_stitch.py`. Three overlapping nighttime photographs with multi-band Laplacian blending and exposure compensation.

**Output:** [`Result/AppendixA/output/`](Result/AppendixA/output/)

| File | Description |
|------|-------------|
| `01_input_images.png` | Side-by-side input images |
| `02_sift_keypoints.png` | Detected SIFT keypoints |
| `03_matches_0_1.png` | Feature matches (Image 0 ↔ Image 1) |
| `03_matches_2_1.png` | Feature matches (Image 2 ↔ Image 1) |
| `04_naive_stitch.jpg` | Raw naive stitch (no blending, no crop) |
| `04_naive_stitch_cropped.jpg` | Naive stitch after cropping |
| `05_blended_panorama.jpg` | Final blended panorama |
| `06_comparison.png` | Naive vs. blended side-by-side |

---

## Appendix B

Stitched using `panorama_stitch.py`.

**Output:** [`Result/AppendixB/`](Result/AppendixB/)

---

## Appendix C — Repetitive Structures (Memory-Safe)

This dataset originally crashed `panorama_stitch.py` with an OpenCV out-of-memory error (~2.3 GB allocation failure). Repetitive patterns in the scene caused feature aliasing → degenerate homography → massive canvas size. Successfully stitched using `panorama_stitch_memsafe.py` with predictive OOM prevention and disk-backed blending.

**Output:** [`Result/AppendixC/`](Result/AppendixC/)

---

## Appendix D — Repetitive Structures (Memory-Safe)

Another dataset that triggered degenerate homographies. Successfully processed using `panorama_stitch_memsafe.py`.

**Output:** [`Result/AppendixD/`](Result/AppendixD/)

---

## Appendix E — Repetitive Structures (Memory-Safe)

Third dataset requiring the memory-hardened pipeline. `panorama_stitch_memsafe.py` fell back to linear blending where multi-band blending exceeded the 2.0 GB safety threshold.

**Output:** [`Result/AppendixE/`](Result/AppendixE/)

---

## Appendix F

Stitched using `panorama_stitch.py`.

**Output:** [`Result/AppendixF/`](Result/AppendixF/)

---

## File Structure

```
panorama-from-scratch/
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

## Why Two Scripts?

`panorama_stitch.py` works perfectly for well-behaved scenes (A, B, F). But images containing repetitive periodic structures (window blinds, tiled facades, fences) cause **feature aliasing** — SIFT matches the wrong instance of a repeated pattern, producing a degenerate homography that warps the image to an astronomically large canvas. This triggers a hard C++ OOM crash inside OpenCV that Python cannot catch.

`panorama_stitch_memsafe.py` solves this with four interventions:

- **`float64` → `uint8` compression** — 8× RAM reduction per warped image
- **Predictive crash prevention** — estimates memory before calling dangerous functions
- **Disk-backed arrays (`np.memmap`)** — uses hard drive as overflow when RAM is insufficient
- **Linear blend fallback** — automatically skips multi-band blending when it would exceed the safety threshold
