import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, rgba2rgb
from skimage.util import img_as_float
from skimage.feature import canny, peak_local_max

def line_segment_finder(orig_img, hough_img, hough_threshold):
    """
    Challenge 1d: turn Hough line peaks into finite segments using the edge map.

    Parameters
    ----------
    orig_img : (H,W) or (H,W,3)
    hough_img : (Nrho,Ntheta)  -- accumulator image (any numeric scale)
    hough_threshold : float    -- if <1, fraction of max; else absolute

    Returns
    -------
    (H,W,3) uint8 image with detected line SEGMENTS drawn.
    """
    # RGB for drawing + grayscale for edges
    if orig_img.ndim == 2:
        base = gray2rgb(orig_img)
        gray = img_as_float(orig_img)
    else:
        base = orig_img.copy()
        im = orig_img
        if im.shape[2] == 4:
            im = rgba2rgb(im)
        gray = img_as_float(rgb2gray(im))

    H, W = base.shape[:2]
    D = float(max(H, W))  # for size-adaptive params

    # 1) Edges (runner doesn't pass edge map)
    edges = canny(gray, sigma=2.0)

    # 2) Parameter grids from accumulator shape + image size
    A = hough_img.astype(float)
    Nrho, Ntheta = A.shape
    thetas = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
    rho_max = float(np.hypot(H, W))
    rhos = np.linspace(-rho_max, rho_max, Nrho)

    # 3) Peak detection (NMS) + orientation diversification
    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold
    coords = peak_local_max(
        A, min_distance=4, threshold_abs=max(thr, 0.0), exclude_border=False
    )

    # Diversify by theta so we don't keep many peaks with the same orientation
    theta_bins = 6   # number of orientation groups
    per_bin   = 5   # keep up to this many peaks per group
    if len(coords):
        strengths = A[coords[:, 0], coords[:, 1]]
        bins = (coords[:, 1] * theta_bins) // Ntheta  # bin by theta index
        keep = []
        for b in range(theta_bins):
            idx = np.where(bins == b)[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(-strengths[idx])]  # strongest in this bin
            keep.extend(order[:per_bin])
        coords = coords[keep] if keep else coords

    # Optional overall cap
    topk_total = 30
    if len(coords) > topk_total:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk_total]]

    # 4) Segment parameters (adaptive defaults; tweak if needed)
    eps = 0.006 * D           # on-line tolerance (px)
    gap = 0.012 * D           # split gap along the line (px)
    min_len = int(0.015 * D)  # minimum #edge points in a segment

    ys, xs = np.nonzero(edges)

    fig, ax = plt.subplots()
    ax.imshow(base)
    ax.set_axis_off()

    # 5) Build finite segments for each peak
    for (rho_idx, theta_idx) in coords:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # edge pixels near this line (perpendicular distance <= eps)
        d = np.abs(xs * cos_t + ys * sin_t - rho)
        m = d <= eps
        if m.sum() < 2:
            continue

        xe, ye = xs[m], ys[m]

        # project to a 1D coordinate along the line's tangent
        # tangent basis choice: t = x*(-sinθ) + y*(cosθ)
        t = xe * (-sin_t) + ye * (cos_t)
        order = np.argsort(t)
        xe, ye, t = xe[order], ye[order], t[order]

        # split where gaps along t are large
        cuts = np.where(np.diff(t) > gap)[0]
        start = 0
        for b in np.append(cuts, len(t) - 1):
            i0, i1 = start, b
            if (i1 - i0 + 1) >= min_len:
                ax.plot([xe[i0], xe[i1]], [ye[i0], ye[i1]], linewidth=3)
            start = b + 1

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out
