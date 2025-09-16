# -------------------------------------------------------------------------
# Part 1 - Create a Vincent van Gogh collage
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def hw1_walkthrough1():
    # Load the image "Vincent_van_Gogh.png" into memory
    img = io.imread("Vincent_van_Gogh.png")

    # If image is grayscale, convert to 3-channel by stacking
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # If image has an alpha channel (RGBA), drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Note the image type, shape, and range of values
    print("type:", type(img))
    print("shape:", img.shape)
    print("dtype:", img.dtype)
    print("max:", img.max())

    # Display the original image
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image")

    # Separate the image into three color channels and store each into new images
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    red_image = np.zeros_like(img)
    red_image[:, :, 0] = red_channel

    green_image = np.zeros_like(img)
    green_image[:, :, 1] = green_channel

    blue_image = np.zeros_like(img)
    blue_image[:, :, 2] = blue_channel

    # Show channel images
    plt.figure(); plt.imshow(red_image); plt.axis("off"); plt.title("Red image")
    plt.figure(); plt.imshow(green_image); plt.axis("off"); plt.title("Green image")
    plt.figure(); plt.imshow(blue_image); plt.axis("off"); plt.title("Blue image")

    # Create a 1 x 4 collage: original | red | green | blue
    collage_1x4 = np.concatenate((img, red_image, green_image, blue_image), axis=1)
    plt.figure()
    plt.imshow(collage_1x4)
    plt.axis("off")
    plt.title("1 x 4 collage")

    # Create a 2 x 2 collage:
    # original | red
    # green    | blue
    top_row = np.concatenate((img, red_image), axis=1)
    bottom_row = np.concatenate((green_image, blue_image), axis=1)
    collage_2x2 = np.concatenate((top_row, bottom_row), axis=0)

    plt.figure()
    plt.imshow(collage_2x2)
    plt.axis("off")
    plt.title("2 x 2 collage")

    # Save the collage as collage.png
    io.imsave("collage.png", collage_2x2)

    plt.show()

# If you want to run directly:
if __name__ == "__main__":
    hw1_walkthrough1()
