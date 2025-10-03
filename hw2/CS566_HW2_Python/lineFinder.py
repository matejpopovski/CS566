import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.feature import peak_local_max

def _clip_line_to_image(rho, theta, H, W):
    """
    Clip the infinite line x*cosθ + y*sinθ = ρ to the image rectangle [0,W-1]×[0,H-1].
    Returns ((xA,yA),(xB,yB)) or None if it doesn't intersect properly.
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = []

    # Intersect with x = 0 and x = W-1  -> y = (ρ - x cosθ)/sinθ
    if abs(sin_t) > 1e-9:
        y0 = (rho - 0 * cos_t) / sin_t
        y1 = (rho - (W - 1) * cos_t) / sin_t
        if 0 <= y0 <= H - 1: pts.append((0, y0))
        if 0 <= y1 <= H - 1: pts.append((W - 1, y1))

    # Intersect with y = 0 and y = H-1  -> x = (ρ - y sinθ)/cosθ
    if abs(cos_t) > 1e-9:
        x0 = (rho - 0 * sin_t) / cos_t
        x1 = (rho - (H - 1) * sin_t) / cos_t
        if 0 <= x0 <= W - 1: pts.append((x0, 0))
        if 0 <= x1 <= W - 1: pts.append((x1, H - 1))

    if len(pts) < 2:
        return None

    # Deduplicate near-identical points
    dedup = []
    for p in pts:
        if not any(np.hypot(p[0]-q[0], p[1]-q[1]) < 1e-3 for q in dedup):
            dedup.append(p)
    if len(dedup) < 2:
        return None

    # Choose the farthest pair for stability
    max_d, pair = -1, None
    for i in range(len(dedup)):
        for j in range(i+1, len(dedup)):
            d = (dedup[i][0]-dedup[j][0])**2 + (dedup[i][1]-dedup[j][1])**2
            if d > max_d:
                max_d, pair = d, (dedup[i], dedup[j])
    return pair

def line_finder(orig_img, hough_img, hough_threshold):
    """
    Challenge 1c: detect lines from a Hough accumulator and draw them on the image.

    Parameters
    ----------
    orig_img : (H,W) or (H,W,3)
    hough_img : (Nrho,Ntheta)  -- accumulator image (any numeric scale)
    hough_threshold : float    -- if <1, fraction of max; else absolute

    Returns
    -------
    (H,W,3) uint8 image with detected (clipped) lines overlaid
    """
    # Ensure RGB for drawing
    img_rgb = gray2rgb(orig_img) if orig_img.ndim == 2 else orig_img.copy()
    H, W = img_rgb.shape[:2]

    A = hough_img.astype(float)
    Nrho, Ntheta = A.shape

    # Reconstruct parameter grids for (rho_idx, theta_idx) -> (rho, theta)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
    rho_max = float(np.hypot(H, W))
    rhos = np.linspace(-rho_max, rho_max, Nrho)

    # Threshold (fraction of max if <1, else absolute)
    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold

    # --- Peak detection with NMS ---
    coords = peak_local_max(
        A, min_distance=4, threshold_abs=max(thr, 0.0), exclude_border=False
    )

    # --- Diversify by orientation bins (avoid all-horizontals/verticals) ---
    theta_bins = 6      # number of orientation groups
    per_bin   = 5       # keep up to this many peaks per group
    if len(coords):
        strengths = A[coords[:, 0], coords[:, 1]]
        bins = (coords[:, 1] * theta_bins) // Ntheta
        keep = []
        for b in range(theta_bins):
            idx = np.where(bins == b)[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(-strengths[idx])]
            keep.extend(order[:per_bin])
        coords = coords[keep] if keep else coords

    # Optional overall cap
    topk_total = 30
    if len(coords) > topk_total:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk_total]]

    # --- Draw lines ---
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    for (rho_idx, theta_idx) in coords:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        seg = _clip_line_to_image(rho, theta, H, W)
        if seg is None:
            continue
        (xA, yA), (xB, yB) = seg
        ax.plot([xA, xB], [yA, yB], linewidth=2)  # colors auto-cycle

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out
