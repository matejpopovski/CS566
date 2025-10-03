"""
lineFinder.py — Hough lines & segments (CS566 HW2)
Dependencies: numpy, matplotlib, scikit-image
Outputs when run as a script:
  - <name>_acc.png         (Hough accumulator heatmap)
  - <name>_lines.png       (infinite lines clipped to image bounds)
  - <name>_segments.png    (finite segments snapped to edges)
Usage:
  python lineFinder.py                    # uses hough_1.png by default
  python lineFinder.py hough_2.png        # custom image
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.util import img_as_float
from skimage.feature import canny, peak_local_max
from skimage.color import gray2rgb

# ---------------------------------------
# 1) Hough Accumulator
# ---------------------------------------
def generateHoughAccumulator(edge_img, theta_num_bins=180, rho_num_bins=None):
    """
    Build the Hough accumulator for a binary edge image.

    Parameters
    ----------
    edge_img : (H,W) bool/uint8
    theta_num_bins : int
    rho_num_bins : int or None

    Returns
    -------
    A_img : (Nrho, Ntheta) uint8   # normalized 0..255 for viz/thresholding
    thetas : (Ntheta,) float        # radians in [-pi/2, pi/2)
    rhos   : (Nrho,) float          # pixels in [-rho_max, +rho_max]
    """
    H, W = edge_img.shape
    rho_max = int(np.ceil(np.hypot(H, W)))
    if rho_num_bins is None:
        rho_num_bins = 2 * rho_max + 1  # cover [-rho_max, +rho_max] inclusive

    thetas = np.linspace(-np.pi / 2, np.pi / 2, theta_num_bins, endpoint=False)
    rhos = np.linspace(-rho_max, rho_max, rho_num_bins)

    A = np.zeros((rho_num_bins, theta_num_bins), dtype=np.uint32)

    ys, xs = np.nonzero(edge_img)  # row=y, col=x
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Vote each edge pixel across all theta bins (vectorized per pixel)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t                   # (Ntheta,)
        rho_idx = np.round(rho_vals + rho_max).astype(int) # map to [0, Nrho)
        valid = (rho_idx >= 0) & (rho_idx < rho_num_bins)
        A[rho_idx[valid], np.where(valid)[0]] += 1

    # Normalize to 0..255 for convenient visualization & thresholds
    A_img = (A / (A.max() if A.max() > 0 else 1) * 255).astype(np.uint8)
    return A_img, thetas, rhos


# ---------------------------------------
# 2) Accumulator visualization with peaks
# ---------------------------------------
def save_acc_with_peaks(A, thetas, rhos, path, threshold=0.6, nms_size=9, topk=12):
    """
    Save the accumulator heatmap with detected peaks marked.
    threshold < 1 is fraction of max; >=1 is absolute.
    """
    A = A.astype(float)
    thr = threshold * A.max() if threshold < 1 else threshold
    coords = peak_local_max(
        A, min_distance=max(1, nms_size // 2),
        threshold_abs=thr, exclude_border=False
    )
    if topk is not None and len(coords) > topk:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk]]

    plt.figure(figsize=(7, 10))
    plt.imshow(A, cmap="inferno", aspect="auto")
    if len(coords):
        plt.scatter(coords[:, 1], coords[:, 0],
                    s=16, facecolors='none', edgecolors='cyan', linewidths=1)
    # coarse ticks for readability
    xt = np.linspace(0, len(thetas) - 1, 5, dtype=int)
    yt = np.linspace(0, len(rhos) - 1, 5, dtype=int)
    plt.xticks(xt, np.round(np.rad2deg(thetas[xt])).astype(int))
    plt.yticks(yt, np.round(rhos[yt]).astype(int))
    plt.xlabel("theta (deg)")
    plt.ylabel("rho (px)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------------------
# 3) Geometry helper — clip line to image
# ---------------------------------------
def _clip_line_to_image(rho, theta, H, W):
    """
    For line x*cosθ + y*sinθ = ρ, find intersections with
    the image rectangle [0,W-1]×[0,H-1].
    Return two points or None if no valid segment.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = []

    # Intersections with vertical borders x=0 and x=W-1
    if abs(sin_t) > 1e-9:
        y0 = (rho - 0 * cos_t) / sin_t
        y1 = (rho - (W - 1) * cos_t) / sin_t
        if 0 <= y0 <= H - 1: pts.append((0, y0))
        if 0 <= y1 <= H - 1: pts.append((W - 1, y1))

    # Intersections with horizontal borders y=0 and y=H-1
    if abs(cos_t) > 1e-9:
        x0 = (rho - 0 * sin_t) / cos_t
        x1 = (rho - (H - 1) * sin_t) / cos_t
        if 0 <= x0 <= W - 1: pts.append((x0, 0))
        if 0 <= x1 <= W - 1: pts.append((x1, H - 1))

    if len(pts) < 2:
        return None

    # Deduplicate near-equal points
    dedup = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-3 for q in dedup):
            dedup.append(p)
    if len(dedup) < 2:
        return None

    # Choose the farthest pair
    max_d, pair = -1, None
    for i in range(len(dedup)):
        for j in range(i + 1, len(dedup)):
            d = (dedup[i][0]-dedup[j][0])**2 + (dedup[i][1]-dedup[j][1])**2
            if d > max_d:
                max_d, pair = d, (dedup[i], dedup[j])
    return pair


# ---------------------------------------
# 4) Line finder — draw infinite (clipped) lines
# ---------------------------------------
def line_finder(orig_img, hough_img, thetas, rhos,
                hough_threshold=0.6, nms_size=9, topk=12):
    """
    Detect lines from Hough accumulator and overlay them on the original image.

    Parameters
    ----------
    orig_img : (H,W) or (H,W,3)
    hough_img : (Nrho,Ntheta) uint8 or float  # scaled OK
    thetas, rhos : arrays from generateHoughAccumulator
    hough_threshold : float  (fraction of max if <1, else absolute)
    nms_size : int  (NMS window)
    topk : keep at most K peaks

    Returns
    -------
    line_detected_img : (H,W,3) uint8
    """
    img_rgb = gray2rgb(orig_img) if orig_img.ndim == 2 else orig_img.copy()
    H, W = img_rgb.shape[:2]
    A = hough_img.astype(float)

    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold
    coords = peak_local_max(
        A, min_distance=max(1, nms_size // 2),
        threshold_abs=max(thr, 0.0), exclude_border=False
    )
    if topk is not None and len(coords) > topk:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk]]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    for (rho_idx, theta_idx) in coords:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        seg = _clip_line_to_image(rho, theta, H, W)
        if seg is None:
            continue
        (xA, yA), (xB, yB) = seg
        ax.plot([xA, xB], [yA, yB], linewidth=2)  # default color cycle

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out


# ---------------------------------------
# 5) Line segment finder — finite segments via edges
# ---------------------------------------
def line_segment_finder(orig_img, edge_img, hough_img, thetas, rhos,
                        hough_threshold=0.6, nms_size=9, topk=12,
                        eps=2.0, gap=8, min_len=15):
    """
    Find Hough peaks, then for each line (rho, theta) keep edge pixels
    near the line (|x cosθ + y sinθ - ρ| <= eps), sort them along the line,
    split by large gaps, and draw the resulting finite segments.

    eps     : max perpendicular distance (px) from edge pixel to the line
    gap     : max allowed spacing in projected coordinate before splitting
    min_len : minimum number of edge pixels to keep a segment
    """
    img_rgb = gray2rgb(orig_img) if orig_img.ndim == 2 else orig_img.copy()
    H, W = img_rgb.shape[:2]
    A = hough_img.astype(float)

    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold
    coords = peak_local_max(
        A, min_distance=max(1, nms_size // 2),
        threshold_abs=max(thr, 0.0), exclude_border=False
    )
    if topk is not None and len(coords) > topk:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk]]

    ys, xs = np.nonzero(edge_img)  # edge pixels
    fig, ax = plt.subplots()
    ax.imshow(img_rgb); ax.set_axis_off()

    for (rho_idx, theta_idx) in coords:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Perpendicular distance to the line
        d = np.abs(xs * cos_t + ys * sin_t - rho)
        mask = d <= eps
        if mask.sum() < 2:
            continue
        xe, ye = xs[mask], ys[mask]

        # Project onto a tangent coordinate along the line
        # One convenient tangent basis: t = x*(-sinθ) + y*(cosθ)
        t = xe * (-sin_t) + ye * (cos_t)
        order = np.argsort(t)
        xe, ye, t = xe[order], ye[order], t[order]

        # Split where gaps in t are large
        cut = np.where(np.diff(t) > gap)[0]
        start = 0
        for b in np.append(cut, len(t) - 1):
            i0, i1 = start, b
            if (i1 - i0 + 1) >= min_len:
                ax.plot([xe[i0], xe[i1]], [ye[i0], ye[i1]], linewidth=3)
            start = b + 1

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out


# ---------------------------------------
# 6) Script entry point
# ---------------------------------------
if __name__ == "__main__":
    # Choose input
    img_path = sys.argv[1] if len(sys.argv) > 1 else "hough_1.png"
    name = os.path.splitext(os.path.basename(img_path))[0]

    print("cwd:", os.getcwd())
    print("Reading:", img_path)
    img = io.imread(img_path)

    # Robust grayscale handling (Canny expects float in [0,1])
    if img.ndim == 2:
        gray = img_as_float(img)
    elif img.ndim == 3:
        if img.shape[2] == 4:                # RGBA → RGB
            img = color.rgba2rgb(img)
        gray = img_as_float(color.rgb2gray(img))
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Edges and accumulator
    edges = canny(gray, sigma=2.0)
    A_img, thetas, rhos = generateHoughAccumulator(edges, theta_num_bins=180)

    # Accumulator with peaks
    acc_out = f"{name}_acc.png"
    save_acc_with_peaks(A_img, thetas, rhos, acc_out,
                        threshold=0.6, nms_size=9, topk=12)
    print("Saved:", acc_out)

    # Infinite lines
    lines_out = f"{name}_lines.png"
    overlay = line_finder(img, A_img, thetas, rhos,
                          hough_threshold=0.6, nms_size=9, topk=12)
    io.imsave(lines_out, overlay)
    print("Saved:", lines_out)

    # Finite segments
    seg_out = f"{name}_segments.png"
    segments = line_segment_finder(img, edges, A_img, thetas, rhos,
                                   hough_threshold=0.6, nms_size=9, topk=12,
                                   eps=2.0, gap=8, min_len=15)
    io.imsave(seg_out, segments)
    print("Saved:", seg_out)
