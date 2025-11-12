import numpy as np
from scipy.signal import convolve2d
from skimage import img_as_float


def _box_filter(img, r):
    if r <= 0:
        return img
    k = np.ones((2*r+1, 2*r+1), dtype=np.float32) / ((2*r+1)*(2*r+1))
    return convolve2d(img, k, mode="same", boundary="symm")


def generate_index_map(gray_stack, w_size):
    """
    Compute index map via Sum-Modified Laplacian (SML) focus measure.
    gray_stack: (H, W, N) float array in [0,1]
    w_size: half window size for spatial averaging of the SML (typ. 8~16)
    Returns:
        index_map: (H, W) int32 with best-focused layer index (0..N-1)
    """
    H, W, N = gray_stack.shape
    gray_stack = img_as_float(gray_stack)

    # Modified Laplacian kernels
    Kx = np.array([[0.25, 0, 0.25],
                   [1.0,  -3.0, 1.0],
                   [0.25, 0, 0.25]], dtype=np.float32)
    Ky = Kx.T

    # Focus measure volume (H, W, N)
    F = np.zeros((H, W, N), dtype=np.float32)
    for i in range(N):
        I = gray_stack[:, :, i]
        Lx = convolve2d(I, Kx, mode='same', boundary='symm')
        Ly = convolve2d(I, Ky, mode='same', boundary='symm')
        SML = np.abs(Lx) + np.abs(Ly)              # focus measure
        F[:, :, i] = _box_filter(SML, w_size)      # spatial smoothing

    index_map = np.argmax(F, axis=2).astype(np.int32)
    return index_map
