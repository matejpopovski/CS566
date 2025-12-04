import numpy as np
import cv2


def draw_box(img, rect, rgb, thickness):
    # draw_box draws a box on the input image
    #
    # arguments:
    # rect:         [x, y, width, height] //x, y, width, height all in integer
    # rgb:          [r g b] //r, g, b are numbers between 0 and 255
    # thickness:    a scalar to specify the thickness of the box
    #               *thickness has to be an odd number*
    #
    # usage:
    # img = cv2.imread('onion.png')
    # rect = [65, 45, 30, 30]
    # rgb = [255, 0, 0]
    # thickness = 3
    # img = draw_box(img, rect, rgb, thickness)

    # -----------------
    # parameters
    # -----------------
    rect = np.floor(rect).astype(int)  # indices have to be of the integer type
    x, y, w, h = rect
    img_h, img_w = img.shape[:2]

    # coordinate system
    #
    # -------x----------
    # |
    # |
    # y
    # |
    # |

    # -----------------
    # error checking
    # -----------------

    if (thickness - 1) % 2 != 0:
        raise ValueError('thickness has to be an odd number.')
    thickness = (thickness - 1) // 2

    # fill the outter box
    # boundary condition check
    # y-t:y+h-1+t, x-t:x+w-1+t
    oy_start = max(0, y - thickness)
    oy_end = min(img_h, y + h - 1 + thickness)
    ox_start = max(0, x - thickness)
    ox_end = min(img_w, x + w - 1 + thickness)
    outter_patch = img[oy_start:oy_end, ox_start:ox_end, :]

    # put the inner box back
    # y+t+1:y+h-t-2, x+t+1:x+w-t-2
    iy_start = max(0, y + thickness + 1)
    iy_end = min(img_h, y + h - thickness - 2)
    ix_start = max(0, x + thickness + 1)
    ix_end = min(img_w, x + w - thickness - 2)
    inner_patch = img[iy_start:iy_end, ix_start:ix_end, :]

    # the type of the outter box has to match the type of the image

    oh = outter_patch.shape[0]
    ow = outter_patch.shape[1]
    outter_box = np.stack((
        np.ones((oh, ow)) * rgb[0],
        np.ones((oh, ow)) * rgb[1],
        np.ones((oh, ow)) * rgb[2]
    ), axis=2)

    if img.dtype == np.float64 or img.dtype == np.float32:
        outter_box = outter_box.astype(np.float64) / 255.0
    elif img.dtype == np.uint8:
        pass
    else:
        raise TypeError(f'Unsupported image type: {img.dtype}')

    img[oy_start:oy_end, ox_start:ox_end, :] = outter_box
    img[iy_start:iy_end, ix_start:ix_end, :] = inner_patch

    return img
