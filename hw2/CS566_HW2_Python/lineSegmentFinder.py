import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb


def line_segment_finder(orig_img, hough_img, hough_threshold):
    """
    Detect line segments from Hough accumulator and draw them on the original image.

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
    cropped_line_img : ndarray
        Annotated image with detected line segments.
    """
    # Ensure image is RGB
    if orig_img.ndim == 2:
        img_rgb = gray2rgb(orig_img)
    else:
        img_rgb = orig_img.copy()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # --------------------------------------
    # START ADDING YOUR CODE HERE
    # --------------------------------------

    (H, W) = orig_img.shape
    (N_rho, N_theta) = hough_img.shape

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    # Convert figure to image array
    fig.canvas.draw()
    line_detected_img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    plt.close(fig)

    return line_detected_img
