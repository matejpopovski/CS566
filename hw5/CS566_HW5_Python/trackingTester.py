import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree


import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree


def tracking_tester(data_params, tracking_params):

    os.makedirs(data_params['out_dir'], exist_ok=True)
    plt.close('all')
    fig, ax = plt.subplots()

    trackingbox_color = [0, 255, 0]

    # starting object location
    r = np.array(tracking_params['rect'], dtype=int)
    rprev = r.copy()

    bin_n = tracking_params['bin_n']

    # histogram template and palette from first frame
    target_hist = None
    palette = None

    # for each frame
    for i, frame_id in enumerate(data_params['frame_ids']):

        # generate image filename
        fname = data_params['genFname'](frame_id)
        f_in = os.path.join(data_params['data_dir'], fname)
        f_out = os.path.join(data_params['out_dir'], fname)

        # read the image
        img = io.imread(f_in)

        # make color if image is grayscale
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # discard image's alpha channel, if present
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, 0:3]

        # search rectangle (larger than the region of interest)
        rs = expand_rect(
            rprev, tracking_params['search_radius'], img.shape)
        rprev = r.copy()

        if i == 0:
            # for the first image:
            # 1. compute the color map and indexed image using rgb2ind
            # 2. create a histogram of the region of interest

            indexed_img, palette = rgb2ind(img, n_colors=bin_n)

            x, y, w, h = r.astype(int)
            h_img, w_img = indexed_img.shape

            # clamp ROI to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)

            roi = indexed_img[y:y + h, x:x + w]

            # histogram over indices 0..bin_n-1
            target_hist, _ = np.histogram(
                roi, bins=bin_n, range=(0, bin_n)
            )
            target_hist = target_hist.astype(np.float32)
            if target_hist.sum() > 0:
                target_hist /= target_hist.sum()

        else:
            # for every image after the first
            # 1. compute the indexed image using the color map from the first image using rgb2ind
            indexed_img, _ = rgb2ind(img, n_colors=bin_n, palette=palette)

            # 2. calculate the search region of interest based on the previous image location
            sx, sy, sw, sh = rs.astype(int)
            h_img, w_img = indexed_img.shape

            sx = max(0, sx)
            sy = max(0, sy)
            sw = min(sw, w_img - sx)
            sh = min(sh, h_img - sy)

            if sw <= 0 or sh <= 0:
                r = rprev.copy()
            else:
                search_region = indexed_img[sy:sy + sh, sx:sx + sw]

                # 3. generate all sliding windows within the region of interest
                # (hint: use view_as_windows)
                win_w = int(r[2])
                win_h = int(r[3])

                if search_region.shape[0] < win_h or search_region.shape[1] < win_w:
                    r = rprev.copy()
                else:
                    windows = view_as_windows(
                        search_region, (win_h, win_w)
                    )

                    n_y, n_x = windows.shape[:2]
                    best_score = np.inf
                    best_pos = (0, 0)

                    # 4. for each window, calculate a histogram
                    for yy in range(n_y):
                        for xx in range(n_x):
                            win_idx = windows[yy, xx]
                            hist, _ = np.histogram(
                                win_idx, bins=bin_n, range=(0, bin_n)
                            )
                            hist = hist.astype(np.float32)
                            if hist.sum() > 0:
                                hist /= hist.sum()

                            score = np.sum(np.abs(hist - target_hist))

                            if score < best_score:
                                best_score = score
                                best_pos = (yy, xx)

                    # 5. find the best match, update the current object location (r)
                    best_yy, best_xx = best_pos
                    new_x = sx + best_xx
                    new_y = sy + best_yy
                    r = np.array([new_x, new_y, win_w, win_h], dtype=int)

            # save the current object location as the previous object location
            rprev = r.copy()

        # Draw rectangle on the image
        img_boxed = draw_box(img, r, trackingbox_color, thickness=3)

        ax.clear()
        ax.imshow(img_boxed)
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')

        io.imsave(f_out, save_annotated_img(fig))

        plt.pause(0.01)



def expand_rect(rin, amount, limits):
    """Expand rectangle [x, y, w, h] by 'amount' within image limits."""
    h, w = limits[:2]
    xmin = max(rin[0] - amount, 0)
    ymin = max(rin[1] - amount, 0)
    xmax = min(rin[0] + rin[2] + amount, w)
    ymax = min(rin[1] + rin[3] + amount, h)
    return np.array([xmin, ymin, xmax - xmin, ymax - ymin], dtype=int)


def draw_box(img, rect, color=(255, 0, 0), thickness=2):
    """Draw rectangular box on an RGB image (like MATLAB's drawBox)."""
    x, y, w, h = rect.astype(int)
    img = img.copy()
    img[y:y+thickness, x:x+w] = color
    img[y+h-thickness:y+h, x:x+w] = color
    img[y:y+h, x:x+thickness] = color
    img[y:y+h, x+w-thickness:x+w] = color
    return img


def save_annotated_img(fig):
    """Equivalent to MATLAB saveAnnotatedImg."""
    # Render figure to array
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., 0:3]
    return img


def rgb2ind(img, n_colors=16, palette=None, random_state=0):
    """
    MATLAB-like rgb2ind using KMeans color quantization.

    Parameters
    ----------
    img : ndarray (H, W, 3)
        RGB image, uint8 or float in [0,1].
    n_colors : int
        Number of desired colors in palette.
    palette : ndarray (K, 3), optional
        If given, assigns pixels to nearest palette color instead of re-clustering.
    random_state : int
        Random seed for KMeans reproducibility.

    Returns
    -------
    indexed_img : ndarray (H, W), dtype=int
        Each pixel replaced by its color index (0..K-1).
    palette : ndarray (K, 3)
        The RGB palette (values in [0,1]).
    """

    # Convert to float in [0,1]
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0

    h, w, c = img.shape
    pixels = img.reshape(-1, c)

    if palette is None:
        # Compute palette with KMeans
        kmeans = KMeans(n_clusters=n_colors,
                        random_state=random_state, n_init=5)
        labels = kmeans.fit_predict(pixels)
        palette = np.clip(kmeans.cluster_centers_, 0, 1)
    else:
        # Map to nearest color in existing palette
        palette = np.asarray(palette, dtype=np.float32)
        if palette.max() > 1.0:
            palette /= 255.0
        tree = cKDTree(palette)
        _, labels = tree.query(pixels)

    indexed_img = labels.reshape(h, w).astype(np.int32)
    return indexed_img, palette


def show_quantized_image(indexed_img, palette, ax=None, title="Quantized Image"):
    """
    Displays a quantized (indexed) image using the given color palette.

    Parameters
    ----------
    indexed_img : ndarray (H, W)
        2D array of integer color indices.
    palette : ndarray (K, 3)
        Palette of RGB colors, values in [0,1] or [0,255].
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axis to draw on. If None, creates a new figure.
    title : str
        Title for the plot.
    """
    # Ensure palette in [0,1]
    palette = np.asarray(palette, dtype=np.float32)
    if palette.max() > 1.0:
        palette /= 255.0

    # Reconstruct the RGB image
    quantized_img = palette[indexed_img]

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

    ax.imshow(quantized_img)
    ax.set_title(title)
    ax.axis('off')
    plt.show()
