import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.filters import gaussian, sobel
from skimage.feature import canny

def must_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input image: {path}\n"
                                f"Current working directory: {os.getcwd()}\n"
                                f"Files here: {', '.join(os.listdir(os.getcwd()))[:300]} ...")
    return io.imread(path)

def hw2_walkthrough1():
    print("== CS566 HW2 walkthrough ==")
    print("Working directory:", os.getcwd())

    # -----------------------
    # Image processing: convolution, Gaussian smoothing
    # -----------------------
    flower_path = 'flower.png'
    img = must_read(flower_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    sigma_list = [6, 12, 24]
    for i, sigma in enumerate(sigma_list):
        k = int(np.ceil(2 * np.pi * sigma))  # kernel size heuristic
        blur_img = gaussian(img, sigma=sigma, truncate=(k/(2*sigma)), channel_axis=2)
        axes[i+1].imshow(blur_img)
        axes[i+1].set_title(f'σ = {sigma}')
        axes[i+1].axis('off')

    out1 = 'blur_flowers.png'
    plt.tight_layout()
    plt.savefig(out1, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out1}")

    # -----------------------
    # Edge detection
    # -----------------------
    hello_path = 'hello.png'
    img2 = must_read(hello_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(img2)
    axes[0].set_title('Color Image')
    axes[0].axis('off')

    gray_img = color.rgb2gray(img2)
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')

    sobel_mag = sobel(gray_img)
    thr = filters.threshold_otsu(sobel_mag)
    edge_sobel = sobel_mag > thr
    axes[2].imshow(edge_sobel, cmap='gray')
    axes[2].set_title(f'Sobel (Otsu thr={thr:.3f})')
    axes[2].axis('off')

    edge_canny = canny(gray_img, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    axes[3].imshow(edge_canny, cmap='gray')
    axes[3].set_title('Canny (σ=1.5, low=0.05, high=0.15)')
    axes[3].axis('off')

    out2 = 'hello_edges.png'
    plt.tight_layout()
    plt.savefig(out2, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out2}")

if __name__ == "__main__":
    try:
        hw2_walkthrough1()
    except Exception as e:
        print("\nERROR while running walkthrough:", repr(e))
        print("Tip: ensure 'flower.png' and 'hello.png' are in the SAME folder as this script,")
        print("or change the paths accordingly.")
