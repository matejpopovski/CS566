import numpy as np
from tqdm import tqdm


def generate_hough_accumulator(img, theta_num_bins, rho_num_bins):
    """
    Generate a Hough accumulator array for an edge image.

    Parameters
    ----------
    img : ndarray (H, W)
        Edge image (nonzero pixels are treated as edges).
    theta_num_bins : int
        Number of bins for theta.
    rho_num_bins : int
        Number of bins for rho.

    Returns
    -------
    hough_img : ndarray (rho_num_bins, theta_num_bins)
        Hough accumulator normalized to 0-255.
    """

    # ---------------------------
    # START ADDING YOUR CODE HERE
    # ---------------------------

    # YOU CAN MODIFY/REMOVE THE PART BELOW IF YOU WANT
    # ------------------------------------------------
    # here we assume origin = middle of image, not top-left corner
    # you can fix the top-left corner too (just remove the part below)
    # centre_x = np.floor(W/2);
    # centre_y = np.floor(H/2);
    # x = x - centre_x;
    # y = y - centre_y;
    # ------------------------------------------------

    # Initialize accumulator
    hough_img = np.zeros((rho_num_bins, theta_num_bins), dtype=np.float64)

    # img is an edge image, find edge pixels
    row_idxs, col_idxs = np.nonzero(img)

    # Calculate rho and theta for the edge pixels

    # Map to an index in the hough_img array
    # and accumulate votes.

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return hough_img
