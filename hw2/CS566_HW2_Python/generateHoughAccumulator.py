import numpy as np
# tqdm was in the template but isn't needed; removed to avoid extra deps.

def generate_hough_accumulator(img, theta_num_bins, rho_num_bins):
    """
    Generate a Hough accumulator for a binary edge image (nonzeros are edges).

    Parameters
    ----------
    img : (H, W) array-like
        Edge image (bool/0-1). Nonzero pixels are treated as edges.
    theta_num_bins : int
        Number of theta bins over [-pi/2, pi/2).
    rho_num_bins : int
        Number of rho bins over [-rho_max, +rho_max], where rho_max = hypot(H,W).

    Returns
    -------
    hough_img : (rho_num_bins, theta_num_bins) float64
        Raw vote counts (NOT normalized to 0..255). The runner normalizes on save.
    """
    img = np.asarray(img)
    H, W = img.shape
    rho_max = float(np.hypot(H, W))

    thetas = np.linspace(-np.pi / 2, np.pi / 2, theta_num_bins, endpoint=False)
    rhos   = np.linspace(-rho_max, rho_max, rho_num_bins)

    A = np.zeros((rho_num_bins, theta_num_bins), dtype=np.float64)

    # edge coordinates
    ys, xs = np.nonzero(img)          # rows, cols
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # vote for all theta per edge pixel
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t                       # (Ntheta,)
        # map each rho to its nearest bin index
        rho_idx = np.round((rho_vals - rhos[0]) / (rhos[1] - rhos[0])).astype(int)
        valid = (0 <= rho_idx) & (rho_idx < rho_num_bins)
        A[rho_idx[valid], np.where(valid)[0]] += 1.0

    return A
