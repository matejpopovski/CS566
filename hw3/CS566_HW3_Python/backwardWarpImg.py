import numpy as np
from applyHomography import apply_homography
from scipy.ndimage import map_coordinates


def backward_warp_img(src_img, resultToSrc_H, dest_canvas_width_height):
    src_height = src_img.shape[0]
    src_width = src_img.shape[1]
    src_channels = src_img.shape[2]
    dest_width = dest_canvas_width_height[0]
    dest_height = dest_canvas_width_height[1]

    result_img = np.zeros((dest_height, dest_width, src_channels))
    mask = np.zeros((dest_height, dest_width), dtype=bool)

    # this is the overall region covered by result_img
    dest_X, dest_Y = np.meshgrid(np.arange(1, dest_width + 1),
                                 np.arange(1, dest_height + 1))

    # map result_img region to src_img coordinate system using the given homography
    src_pts = apply_homography(resultToSrc_H, np.column_stack(
        [dest_X.ravel(), dest_Y.ravel()]))
    src_X = src_pts[:, 0].reshape(dest_height, dest_width)
    src_Y = src_pts[:, 1].reshape(dest_height, dest_width)

    # ---------------------------
    # START ADDING YOUR CODE HERE
    # ---------------------------

    # Build a grid of destination (canvas) pixel coordinates
    xs, ys = np.meshgrid(np.arange(dest_width), np.arange(dest_height))
    dest_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    # Map canvas -> source using the provided homography
    src_pts = apply_homography(resultToSrc_H, dest_pts)  # (N,2)
    x_s = src_pts[:, 0].reshape(dest_height, dest_width)
    y_s = src_pts[:, 1].reshape(dest_height, dest_width)

    # Valid pixels fall inside the source bounds
    mask = (
        (x_s >= 0) & (x_s <= (src_width - 1)) &
        (y_s >= 0) & (y_s <= (src_height - 1))
    )

    # Sample each channel with bilinear interpolation
    for c in range(src_channels):
        # map_coordinates expects order (row=y, col=x)
        result_img[..., c] = map_coordinates(
            src_img[..., c],
            [y_s, x_s],
            order=1, mode='constant', cval=0.0
        )

    mask = mask.astype(bool)

    # ---------------------------
    # END YOUR CODE HERE
    # ---------------------------

    return mask, result_img
