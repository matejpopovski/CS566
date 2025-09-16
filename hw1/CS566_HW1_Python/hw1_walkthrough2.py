# -------------------------------------------------------------------------
# Part 2 - Create a I <3 NY image
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform


def hw1_walkthrough2():
    # Load "I <3 NY" logo
    iheartny_img = io.imread("I_Love_New_York.png")[..., :3]

    plt.figure()
    plt.imshow(iheartny_img)
    plt.title("Original I <3 NY")

    # Load NYC photo
    nyc_img = io.imread("nyc.png")

    # Resize NYC to height = 500
    scale = 500 / nyc_img.shape[0]
    small_nyc = transform.rescale(
        nyc_img, scale, channel_axis=-1, anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)

    # Resize I <3 NY to height = 400
    scale = 400 / iheartny_img.shape[0]
    resized_iheartny = transform.rescale(
        iheartny_img, scale, channel_axis=-1, anti_aliasing=False, preserve_range=True
    )

    # Convert to grayscale
    gray_iheartny_img = color.rgb2gray(resized_iheartny[..., :3])
    plt.figure(); plt.imshow(gray_iheartny_img, cmap="gray")
    plt.title("Grayscale I <3 NY")

    # Binary mask: letters < threshold → True
    threshold = 0.5
    resized_mask = gray_iheartny_img < threshold

    # Invert so letters/logo appear white
    iresized_mask = ~resized_mask
    plt.figure(); plt.imshow(iresized_mask, cmap="gray")
    plt.title("Inverted Mask")

    # Pad horizontally to match NYC width
    height_diff = small_nyc.shape[0] - iresized_mask.shape[0]
    width_diff = small_nyc.shape[1] - iresized_mask.shape[1]

    left_pad = width_diff // 2
    right_pad = width_diff - left_pad
    top_pad = height_diff // 2
    bottom_pad = height_diff - top_pad

    iresized_mask = np.pad(
        iresized_mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode="constant"
    )
    plt.figure(); plt.imshow(iresized_mask, cmap="gray")
    plt.title("Padded Mask")

    # Burn logo into NYC image
    love_small_nyc = small_nyc.copy()
    # Apply red color where mask = True
    love_small_nyc[iresized_mask] = np.array([255, 0, 0])

    plt.figure()
    plt.imshow(love_small_nyc)
    plt.title("I <3 NY on Manhattan")

    # Save
    io.imsave("output_nyc.png", love_small_nyc.astype(np.uint8))

    plt.show()

if __name__ == "__main__":
    hw1_walkthrough2()
