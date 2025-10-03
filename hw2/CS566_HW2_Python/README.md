# CS566 — Homework 2: Line Detection (Fall 2025)

This repository contains my implementation for HW2. The code is callable via **`runHw2.py`** and uses only the permitted libraries: **NumPy, Matplotlib, scikit-image**.

---

## How to run (as required)

From the project folder (the one containing the images and `runHw2.py`), run:

```bash
python runHw2.py all
```

You can also run individual parts:

```bash
python runHw2.py walkthrough1
python runHw2.py challenge1a
python runHw2.py challenge1b
python runHw2.py challenge1c
python runHw2.py challenge1d
```

**Inputs expected in the same folder:**  
`flower.png`, `hello.png`, `hough_1.png`, `hough_2.png`, `hough_3.png`

**Outputs produced:**  
- Walkthrough 1 → `blur_flowers.png`, `hello_edges.png`  
- Challenge 1a → `edge_hough_1.png`, `edge_hough_2.png`, `edge_hough_3.png`  
- Challenge 1b → `accumulator_hough_1.png`, `accumulator_hough_2.png`, `accumulator_hough_3.png`  
- Challenge 1c → `line_hough_1.png`, `line_hough_2.png`, `line_hough_3.png`  
- Challenge 1d → `linedetected_hough_1.png`, `linedetected_hough_2.png`, `linedetected_hough_3.png`

---

## Walkthrough 1 (1 point): Edge detection demo

Implemented in **`hw2_walkthrough1.py`**. The script demonstrates:
- Gaussian smoothing with σ ∈ {6, 12, 24} (rule-of-thumb kernel width k ≈ 2πσ via `truncate`).
- Edge detection: Sobel (thresholded by Otsu) and Canny.

**Outputs included:** `blur_flowers.png`, `hello_edges.png`

---

## Challenge 1: LineFinder (15 points)

I implement the full line-finding pipeline and test on `hough_1.png`, `hough_2.png`, `hough_3.png`. The pipeline is broken into four subparts as required.

### 1a) Edge detection (1 point)

- **What I did:** In **`runHw2.py → challenge1a()`**, I generate binary edge maps from the grayscale inputs using **`skimage.feature.canny`** with a **single global setting** for all three images:
  - **Canny:** `sigma = 2.0` (thresholds left to the function defaults)
- **Why:** σ=2.0 provides good noise suppression while retaining the tile/book/chart edges needed by the Hough step.

**Outputs:** `edge_hough_*.png`

---

### 1b) Hough Transform (5 points)

- **Equation used:** I use the **normal form**  
  \[ **x·cos(θ) + y·sin(θ) = ρ** \]  
  which is **equivalent** to the spec’s \( x\sin(θ) − y\cos(θ) + ρ = 0 \) up to a rotation/sign flip. I use this form **consistently** for voting and for drawing lines later.

- **Ranges and binning:**
  - \( θ \in [-\pi/2, \pi/2) \) with **`theta_num_bins = 180`** (≈1° resolution).  
  - \( ρ \in [-ρ_{\max}, +ρ_{\max}] \) with **`rho_num_bins = 2·ceil(hypot(H,W)) + 1`** where \( ρ_{\max} = \sqrt{H^2 + W^2} \).

- **Voting scheme (and why):**
  - For each edge pixel \((x,y)\), I compute the set \( ρ(θ) = x\cosθ + y\sinθ \) across all θ bins, **round ρ to the nearest ρ-bin**, and increment that \((ρ,θ)\) cell by **one vote**.
  - I use **nearest-bin voting** (no patch/splat of neighboring bins). In practice this gives sharp peaks for straight edges and keeps computation simple. Peak detection (next step) handles robustness via non-maximum suppression. (I discuss this choice since the spec mentions patch voting as an option if results are weak.)

- **Scaling to [0, 255]:**
  - The accumulator is saved as an 8-bit image **scaled to [0, 255]** for visualization and the subsequent steps use it as such. (Internally, I maintain raw counts during voting; scaling is applied before saving, which is equivalent for the later fraction-of-max thresholding.)

- **Disallowed functions:** I **do not** use `hough_lines`, `hough_line_peaks`, or any Hough-specific helper.

**Output:** `accumulator_hough_*.png`

---

### 1c) Peak finding & infinite lines (4 points)

Implemented in **`lineFinder.py`**.

- **Peak/threshold method:** I detect peaks in the accumulator via a **generic non-maximum suppression** (`skimage.feature.peak_local_max`) with an **absolute threshold derived as a fraction of the maximum**:
  - **`hough_threshold`** is interpreted as a **fraction of max**; I use `0.55` by default (per-image lists can be set in `runHw2.py` if needed).  
  - I also use a small **NMS neighborhood** (`min_distance=4`) to prevent adjacent-bin duplicates.

- **Orientation diversification:** To avoid selecting many peaks with the **same orientation**, I **bin peaks by θ** (6 bins over [−90°,90°)) and keep up to 5 strongest per bin, then cap total peaks (≤30). This yields a mix of vertical, horizontal, and diagonal lines (fixes a common “all horizontals” failure mode).

- **Drawing lines:** For each retained \((ρ,θ)\), I form the infinite line \( x\cosθ + y\sinθ = ρ \) and **clip** it to the image rectangle by intersecting with the four borders; I then draw the visible chord using `Axes.plot()`.

- **Disallowed functions:** I do **not** use `houghpeaks` (or any Hough-line helper). `peak_local_max` is a generic NMS utility, not a Hough helper.

**Output:** `line_hough_*.png`

---

### 1d) Line **segments** (5 points)

Implemented in **`lineSegmentFinder.py`**. This step converts the infinite lines to finite segments supported by the image edges.

**Algorithm (as requested in the spec):**
1. **Edge inliers:** For each detected line \((ρ,θ)\), collect edge pixels satisfying the **perpendicular inlier test**  
   \( |x\cosθ + y\sinθ − ρ| \le ε \), with **ε = 0.006·D**, \( D = \max(H,W) \).
2. **Gradient alignment filter:** Compute image gradients (Sobel). Keep only inliers whose **gradient direction** is within **±30°** of the line’s **normal** (rejects short curved fragments like rims).
3. **1D ordering along the line:** Project the remaining pixels onto the **tangent** coordinate  
   \( t = x(−\sinθ) + y(\cosθ) \), sort by t.
4. **Split by gaps:** Start a new segment whenever the gap between consecutive t’s exceeds **`gap = 0.012·D`**.
5. **Keep real segments only:** Require at least **`min_len_pts = 0.015·D`** inlier points **and** a **geometric span** of at least **`min_span_px = 0.030·D`** (drops tiny/unstable fragments).

- **Peak selection:** Same NMS + θ-diversification as in 1c (cap total ≤ 40).  
- **Edges for this step:** Recomputed from the grayscale original via **Canny (σ = 2.0)** inside the function to avoid dependence on file I/O.

- **Disallowed functions:** Again, no `hough_lines`/`hough_line_peaks` or similar.

**Output:** `linedetected_hough_*.png`

---

## Parameters actually used

Unless otherwise noted, I used a **single set** of parameters for all three images:

- **Challenge 1a (edges for saving):** `Canny σ = 2.0`  
- **Challenge 1b (Hough):** `theta_num_bins = 180`, `rho_num_bins = 2·ceil(hypot(H,W)) + 1`  
- **Challenge 1c (peaks/lines):** `hough_threshold = 0.55` (fraction of max), `NMS min_distance = 4`, θ-bins = 6, ≤5 per bin, cap ≤ 30 peaks  
- **Challenge 1d (segments):** `ε = 0.006·D`, `gap = 0.012·D`, `min_len_pts = 0.015·D`, `min_span_px = 0.030·D`, gradient alignment tolerance = `±30°`, cap ≤ 40 segments

> If the provided images vary slightly, lowering the middle image’s threshold to `0.45` is a safe per-image tweak (supported via lists in `runHw2.py`).

---

## Compliance / notes for the grader

- **No disallowed functions** from scikit-image’s Hough API are used. Peak detection uses a generic NMS helper only.  
- The accumulator is provided as an 8‑bit visualization image scaled to [0,255] per the spec; internally I keep raw counts and scale on save (equivalent for fraction-of-max thresholds used later).  
- All steps run cleanly with `python runHw2.py all` and generate the required PNGs.

---

## Academic honesty

Run:
```bash
python runHw2.py honesty
```
This calls:
```python
sign_academic_honesty_policy("full_name", "stu_id")
```
Please edit with your **name** and **student ID** in `runHw2.py` before submission.

---

## Design decisions (beyond the bare minimum)

- **Parameterization:** Use normal form `x cosθ + y sinθ = ρ` (equivalent to spec’s `x sinθ − y cosθ + ρ = 0`). Chosen for numerical stability and straightforward clipping/drawing.
- **Voting scheme:** **Nearest ρ-bin** voting (1 vote per (ρ,θ) for each edge pixel). Simpler/faster; peaks remained sharp enough on provided images, so I did not splat to neighbor bins.
- **Accumulator scaling:** Keep **raw counts** during voting; **normalize to [0,255] on save**. Later steps use fraction-of-max thresholds, so this is equivalent in practice. (Could return uint8 if preferred.)
- **Peak detection:** Generic **NMS** via `peak_local_max` + **θ-bin diversification** (6 bins, ≤5 peaks/bin) to avoid a single dominant orientation (e.g., “all horizontals” on `hough_2`).
- **Line drawing:** Compute infinite line from (ρ,θ) and **clip** against the image rectangle by analytical border intersections; draw the longest visible chord.
- **Segments:** Robust pruning using **perpendicular inlier test** (`ε = 0.006·D`), **gradient–normal alignment** (±30°), **gap-based splitting** along tangent (`gap = 0.012·D`), and **minimum span** (`0.030·D`) plus a **minimum inlier count** (`0.015·D`). These stabilize results and remove short curved fragments (e.g., rims).
- **Scale-adaptive constants:** Use `D = max(H, W)` so thresholds scale across resolutions without re-tuning.
- **I/O robustness:** When reloading edge images, **binarize** (`>0`) before Hough to avoid grayscale artifacts from PNG I/O.




