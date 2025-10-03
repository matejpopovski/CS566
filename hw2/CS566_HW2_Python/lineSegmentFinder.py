import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, rgba2rgb
from skimage.util import img_as_float
from skimage.feature import canny, peak_local_max
from skimage import filters  # for sobel_h/v

def line_segment_finder(orig_img, hough_img, hough_threshold):
    """
    Challenge 1d: turn Hough line peaks into finite segments using the edge map.
    Adds:
      - orientation diversification (avoid many peaks of same theta)
      - gradient alignment test (edge normals ~ line normal)
      - minimum geometric span (drop tiny fragments)
    """
    # ---- RGB + grayscale ----
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
    D = float(max(H, W))  # image scale for size-adaptive params

    # 1) Edges
    edges = canny(gray, sigma=2.0)

    # 2) Hough parameter grids
    A = hough_img.astype(float)
    Nrho, Ntheta = A.shape
    thetas = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
    rho_max = float(np.hypot(H, W))
    rhos = np.linspace(-rho_max, rho_max, Nrho)

    # 3) Peaks + orientation diversification
    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold
    coords = peak_local_max(
        A, min_distance=4, threshold_abs=max(thr, 0.0), exclude_border=False
    )
    theta_bins = 6
    per_bin   = 5
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
    topk_total = 40
    if len(coords) > topk_total:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:topk_total]]

    # 4) Segment parameters (tweak if needed)
    eps          = 0.006 * D           # on-line tolerance (px)
    gap          = 0.012 * D           # split gap along projected coordinate
    min_len_pts  = int(0.015 * D)      # minimum number of edge points
    min_span_px  = 0.030 * D           # NEW: minimum geometric length (px)
    ang_tol_deg  = 30.0                # NEW: gradient alignment tolerance
    ang_tol      = np.deg2rad(ang_tol_deg)

    # Precompute gradients and their orientations
    gx = filters.sobel_h(gray)
    gy = filters.sobel_v(gray)
    grad_theta = np.arctan2(gy, gx)    # [-pi, pi]

    ys, xs = np.nonzero(edges)

    fig, ax = plt.subplots()
    ax.imshow(base)
    ax.set_axis_off()

    for (rho_idx, theta_idx) in coords:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Edge pixels near this line (perpendicular distance <= eps)
        d = np.abs(xs * cos_t + ys * sin_t - rho)
        m = d <= eps
        if m.sum() < 2:
            continue
        xe, ye = xs[m], ys[m]

        # ---- Gradient alignment filter (edge normals ~ line normal) ----
        # For a true straight edge supporting (rho,theta), gradient direction
        # should align with the line *normal* (theta), up to ang_tol.
        gth = grad_theta[ye, xe]
        # wrap angle difference to [-pi, pi] then take abs
        angdiff = np.abs(np.arctan2(np.sin(gth - theta), np.cos(gth - theta)))
        align = angdiff <= ang_tol
        if align.sum() < 2:
            continue
        xe, ye = xe[align], ye[align]

        # Project onto a 1D coordinate along the line's tangent
        t = xe * (-sin_t) + ye * (cos_t)
        order = np.argsort(t)
        xe, ye, t = xe[order], ye[order], t[order]

        # Split where gaps along t are large
        cuts = np.where(np.diff(t) > gap)[0]
        start = 0
        for b in np.append(cuts, len(t) - 1):
            i0, i1 = start, b
            npts = (i1 - i0 + 1)
            if npts >= min_len_pts:
                # ---- NEW: geometric min span ----
                span = np.hypot(xe[i1] - xe[i0], ye[i1] - ye[i0])
                if span >= min_span_px:
                    ax.plot([xe[i0], xe[i1]], [ye[i0], ye[i1]], linewidth=3)
            start = b + 1

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out
