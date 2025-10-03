import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.feature import peak_local_max

def _clip_line_to_image(rho, theta, H, W):
    """Clip infinite line x*cosθ + y*sinθ = ρ to the image rectangle."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = []

    if abs(sin_t) > 1e-9:
        y0 = (rho - 0 * cos_t) / sin_t
        y1 = (rho - (W - 1) * cos_t) / sin_t
        if 0 <= y0 <= H - 1: pts.append((0, y0))
        if 0 <= y1 <= H - 1: pts.append((W - 1, y1))
    if abs(cos_t) > 1e-9:
        x0 = (rho - 0 * sin_t) / cos_t
        x1 = (rho - (H - 1) * sin_t) / cos_t
        if 0 <= x0 <= W - 1: pts.append((x0, 0))
        if 0 <= x1 <= W - 1: pts.append((x1, H - 1))

    if len(pts) < 2:
        return None

    # dedup nearly identical endpoints
    dedup = []
    for p in pts:
        if not any(np.hypot(p[0]-q[0], p[1]-q[1]) < 1e-3 for q in dedup):
            dedup.append(p)
    if len(dedup) < 2:
        return None

    # choose farthest pair
    max_d, pair = -1, None
    for i in range(len(dedup)):
        for j in range(i + 1, len(dedup)):
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
    # ensure RGB for drawing
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    H, W = img_rgb.shape[:2]
    A = hough_img.astype(float)
    Nrho, Ntheta = A.shape

    # reconstruct parameter grids to convert (rho_idx, theta_idx) -> (rho, theta)
    thetas = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
    rho_max = float(np.hypot(H, W))
    rhos = np.linspace(-rho_max, rho_max, Nrho)

    thr = hough_threshold * A.max() if hough_threshold < 1 else hough_threshold

    # simple NMS to find peaks
    coords = peak_local_max(A, min_distance=4, threshold_abs=max(thr, 0.0), exclude_border=False)
    # keep at most top-20 strongest
    if len(coords) > 20:
        strengths = A[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(-strengths)[:20]]

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    for (rho_idx, theta_idx) in coords:
        rho   = rhos[rho_idx]
        theta = thetas[theta_idx]
        seg = _clip_line_to_image(rho, theta, H, W)
        if seg is None:
            continue
        (xA, yA), (xB, yB) = seg
        ax.plot([xA, xB], [yA, yB], linewidth=2)  # distinct colors cycle automatically

    fig.canvas.draw()
    out = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return out
