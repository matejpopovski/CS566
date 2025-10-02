import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import gray2rgb
from tqdm import tqdm


def line_finder(orig_img, hough_img, hough_threshold):
    """
    Detect lines from Hough accumulator and overlay them on the original image.

    Parameters
    ----------
    orig_img : ndarray (H, W) or (H, W, 3)
        Original grayscale or RGB image.
    hough_img : ndarray (rho_bins, theta_bins)
        Hough accumulator.
    hough_threshold : float
        Threshold above which Hough votes are considered strong.

    Returns
    -------
    line_detected_img : ndarray
        Annotated image with detected lines.
    """

    # Ensure image is RGB for drawing
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # TODO: START ADDING YOUR CODE HERE
    # --------------------------------------

    (H, W) = orig_img.shape
    (N_rho, N_theta) = hough_img.shape

    # You'd want to change this.
    strong_hough_img = hough_img

    # for i in range(N_rho):
    #     for j in range(N_theta):
    #         if strong_hough_img[i, j] > 0:
    #         	# map to corresponding line parameters

    #         	# generate some points for the line

    #         	# and draw on the figure


    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    # Convert figure to image array
    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img
