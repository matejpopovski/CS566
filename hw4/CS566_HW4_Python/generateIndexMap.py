# import numpy as np
# from scipy.signal import convolve2d
# from skimage import img_as_float


# def _box_filter(img, r):
#     if r <= 0:
#         return img
#     k = np.ones((2*r+1, 2*r+1), dtype=np.float32) / ((2*r+1)*(2*r+1))
#     return convolve2d(img, k, mode="same", boundary="symm")


# def generate_index_map(gray_stack, w_size):
#     """
#     Compute index map via Sum-Modified Laplacian (SML) focus measure.
#     gray_stack: (H, W, N) float array in [0,1]
#     w_size: half window size for spatial averaging of the SML (typ. 8~16)
#     Returns:
#         index_map: (H, W) int32 with best-focused layer index (0..N-1)
#     """
#     H, W, N = gray_stack.shape
#     gray_stack = img_as_float(gray_stack)

#     # Modified Laplacian kernels
#     Kx = np.array([[0.25, 0, 0.25],
#                    [1.0,  -3.0, 1.0],
#                    [0.25, 0, 0.25]], dtype=np.float32)
#     Ky = Kx.T

#     # Focus measure volume (H, W, N)
#     F = np.zeros((H, W, N), dtype=np.float32)
#     for i in range(N):
#         I = gray_stack[:, :, i]
#         Lx = convolve2d(I, Kx, mode='same', boundary='symm')
#         Ly = convolve2d(I, Ky, mode='same', boundary='symm')
#         SML = np.abs(Lx) + np.abs(Ly)              # focus measure
#         F[:, :, i] = _box_filter(SML, w_size)      # spatial smoothing

#     index_map = np.argmax(F, axis=2).astype(np.int32)
#     return index_map

import numpy as np
from scipy.ndimage import uniform_filter

def modified_laplacian(img):
    """
    Compute the modified Laplacian focus measure for a single grayscale image.
    """
    import cv2
    # Use small kernel to measure local focus strength
    kernel = np.array([-1, 2, -1], dtype=np.float32)

    # Compute second derivatives
    lap_x = cv2.filter2D(img.astype(np.float32), -1, kernel[np.newaxis, :])
    lap_y = cv2.filter2D(img.astype(np.float32), -1, kernel[:, np.newaxis])

    # Focus measure = |∂²x| + |∂²y|
    focus = np.abs(lap_x) + np.abs(lap_y)
    return focus


def generate_index_map(gray_stack, half_window_size=7):
    """
    Generate an index map from a grayscale focal stack.
    
    Parameters:
        gray_stack : numpy.ndarray
            shape = (H, W, N) — the focal stack (grayscale images)
        half_window_size : int
            size for local smoothing (uniform filter radius)
            
    Returns:
        index_map : numpy.ndarray
            shape = (H, W), each pixel holds the index (0–N−1)
            of the layer where it’s most in focus.
    """

    H, W, N = gray_stack.shape
    print(f"Generating index map from stack of shape {gray_stack.shape}...")

    # Step 1: Compute focus measure for each layer
    focus_measures = np.zeros((H, W, N), dtype=np.float32)
    for i in range(N):
        focus_measures[..., i] = modified_laplacian(gray_stack[..., i])

    # Step 2: Smooth each focus measure map (helps reduce noise)
    if half_window_size > 0:
        size = 2 * half_window_size + 1
        for i in range(N):
            focus_measures[..., i] = uniform_filter(focus_measures[..., i], size=size)

    # Step 3: Find the layer index with maximum focus measure at each pixel
    index_map = np.argmax(focus_measures, axis=2).astype(np.float32)

    return index_map
