import glob
import os
import re

import numpy as np
from skimage import color, img_as_float, io


def load_focal_stack(focal_stack_dir):
    """
    Loads a focal stack from a directory.
    Returns:
        rgb_stack: float array of shape (H, W, 3, N) in [0,1]
        gray_stack: float array of shape (H, W, N) in [0,1]
    The directory is expected to contain images with common extensions.
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(focal_stack_dir, e)))
    if not files:
        raise FileNotFoundError(
            f"No images found in '{focal_stack_dir}'. Put your focal stack images there."
        )

    # ensure consistent front-to-back order
    def _natural_key(fname):
        return [
            int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(fname))
        ]

    files = sorted(files, key=_natural_key)

    first = img_as_float(io.imread(files[0]))
    if first.ndim == 2:
        first_rgb = np.dstack([first, first, first])
        first_gray = first
    else:
        first_rgb = first[..., :3]
        first_gray = color.rgb2gray(first_rgb)

    H, W, _ = first_rgb.shape
    N = len(files)

    rgb_stack = np.zeros((H, W, 3, N), dtype=np.float32)
    gray_stack = np.zeros((H, W, N), dtype=np.float32)

    rgb_stack[..., 0] = first_rgb
    gray_stack[..., 0] = first_gray.astype(np.float32)

    for i, f in enumerate(files[1:], start=1):
        im = img_as_float(io.imread(f))
        if im.ndim == 2:
            im_rgb = np.dstack([im, im, im])
            im_gray = im
        else:
            im_rgb = im[..., :3]
            im_gray = color.rgb2gray(im_rgb)
        if im_rgb.shape[:2] != (H, W):
            raise ValueError(
                f"All images must have identical size. '{f}' has shape {im_rgb.shape[:2]}, expected {(H, W)}."
            )
        rgb_stack[..., i] = im_rgb
        gray_stack[..., i] = im_gray.astype(np.float32)

    return rgb_stack, gray_stack
