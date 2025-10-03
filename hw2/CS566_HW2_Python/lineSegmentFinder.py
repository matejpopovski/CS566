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
    # RGB for drawing
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

    # 1) Edges (we compute them here; runner doesn't pass the edge map)
    edges = canny(gray, sigma=2.0)

    # 2) Reconstruct theta/rho grids from accumulator shape + image size
    A = hough_img.astype(float)
    Nrho, Ntheta = A.shape
    thetas = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
    rho_max = float(np.hypot(H, W))
    rhos = np.linspace(-rho_max, rho_max, Nrho)

    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold
    peaks = peak_local_max(A, min_distance=4, threshold_abs=max(thr, 0.0), exclude_border=False)
    if len(peaks) > 20:
        strengths = A[peaks[:, 0], peaks[:, 1]]
        peaks = peaks[np.argsort(-strengths)[:20]]

    # 3) Segment parameters (adaptive defaults; tweak if needed)
    eps = 0.006 * D          # on-line tolerance (px)
    gap = 0.012 * D          # split gap in projected coordinate (px)
    min_len = int(0.015 * D) # minimum #edge points in a segment

    ys, xs = np.nonzero(edges)

    fig, ax = plt.subplots()
    ax.imshow(base)
    ax.set_axis_off()

    for (rho_idx, theta_idx) in peaks:
        rho, theta = rhos[rho_idx], thetas[theta_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # edge pixels near the line (perpendicular distance <= eps)
        d = np.abs(xs * cos_t + ys * sin_t - rho)
        m = d <= eps
        if m.sum() < 2:
            continue

        xe, ye = xs[m], ys[m]
        # project onto a tangent coordinate along the line
        t = xe * (-sin_t) + ye * (cos_t)
        order = np.argsort(t)
        xe, ye, t = xe[order], ye[order], t[order]

        # split where gaps are large
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
