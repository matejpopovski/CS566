import numpy as np
from scipy.signal import convolve2d
from skimage import img_as_float


def generate_index_map(gray_stack, w_size):
    """
    Compute index map via Sum-Modified Laplacian (SML) focus measure.
    gray_stack: (H, W, N) float array in [0,1]
    w_size: half window size for spatial averaging of the SML (typ. 8~16)
    Returns:
        index_map: (H, W) int32 with best-focused layer index (0..N-1)
    """
    H, W, N = gray_stack.shape
    # Compute the focus measure -- the sum-modified laplacian
    #
    # horizontal Laplacian kernel
    Kx = np.array([[0.25, 0, 0.25], [1, -3, 1], [0.25, 0, 0.25]])
    Ky = Kx.T  # vertical version

    # horizontal and vertical Laplacian responses
    Lx = np.zeros((H, W, N))
    Ly = np.zeros((H, W, N))
    for n in range(N):
        I = img_as_float(gray_stack[:, :, n])
        Lx[:, :, n] = convolve2d(I, Kx, mode="same", boundary="symm")
        Ly[:, :, n] = convolve2d(I, Ky, mode="same", boundary="symm")

    # sum-modified Laplacian
    SML = (np.abs(Lx) ** 2) + (np.abs(Ly) ** 2)
    # can also use the absolute value itself
    # this is probably more well-known
    # SML = np.abs(Lx) + np.abs(Ly)

    # Smooth the focus measure volume spatially (moving-average / box filter)
    # w_size is the HALF window size, so the kernel is (2*w_size+1)^2
    if w_size > 0:
        kside = 2 * w_size + 1
        k = np.ones((kside, kside), dtype=np.float32) / (kside * kside)
        SML_smoothed = np.empty_like(SML)
        for n in range(N):
            SML_smoothed[:, :, n] = convolve2d(
                SML[:, :, n], k, mode="same", boundary="symm"
            )
    else:
        SML_smoothed = SML

    # For each pixel, pick the layer index with maximal (smoothed) focus measure
    index_map = np.argmax(SML_smoothed, axis=2).astype(np.int32)
    return index_map
