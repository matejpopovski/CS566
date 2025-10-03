import sys
import numpy as np
from pathlib import Path
from hw2_walkthrough1 import hw2_walkthrough1
from generateHoughAccumulator import generate_hough_accumulator
from lineFinder import line_finder
from lineSegmentFinder import line_segment_finder
from signAcademicPolicy import sign_academic_honesty_policy
from skimage import io, feature, img_as_ubyte, exposure
from skimage.morphology import binary_closing, disk

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run_hw2(*args):
    """
    run_hw2 is the "main" interface that lets you execute all the walkthroughs
    and challenges in homework 2.
    """
    fun_handles = {
        "honesty": honesty,
        "walkthrough1": walkthrough1,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge1d": challenge1d,
    }

    if len(args) == 0:
        print("Available functions:")
        for name in fun_handles:
            print(" -", name)
        return

    arg = args[0].lower()
    if arg == "all":
        for name, func in fun_handles.items():
            print(f"Running {name}...")
            func()
    elif arg in fun_handles:
        print(f"Running {arg}...")
        fun_handles[arg]()
    else:
        print(f"Unknown argument: {arg}")
        print("Valid options are:", list(fun_handles.keys()) + ["all"])

# -----------------------------------------------------------------------------
# Academic Honesty Policy
# -----------------------------------------------------------------------------

def honesty():
    # Replace with your own name and uni
    sign_academic_honesty_policy("full_name", "stu_id")

# -----------------------------------------------------------------------------
# Walkthrough 1
# -----------------------------------------------------------------------------

def walkthrough1():
    hw2_walkthrough1()

# -----------------------------------------------------------------------------
# Challenge 1a: Edge detection
# -----------------------------------------------------------------------------

def challenge1a():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Challenge 1a: edge detection -> edge_*.png')

    # Gentle, per-image params. hough_2 is the tricky one (was missing a square).
    params = {
        # a little stricter (cleaner)
        "hough_1": dict(sigma=1.2, low=0.05, high=0.14, clip=0.010),
        # a little softer + more local contrast to recover weak border
        "hough_2": dict(sigma=1.0, low=0.030, high=0.100, clip=0.012),
        # similar to 1
        "hough_3": dict(sigma=1.2, low=0.05, high=0.14, clip=0.010),
    }

    for name in img_list:
        g = io.imread(f"{name}.png", as_gray=True)

        # 1) Local contrast boost (CLAHE)
        p = params[name]
        g_eq = exposure.equalize_adapthist(g, clip_limit=p["clip"])

        # 2) Canny (no object removal; we want to keep weak but real borders)
        edges = feature.canny(
            g_eq,
            sigma=p["sigma"],
            low_threshold=p["low"],
            high_threshold=p["high"],
        )

        # 3) Close tiny gaps (one-pixel structural element)
        edges = binary_closing(edges, footprint=disk(1))

        io.imsave(f"edge_{name}.png", img_as_ubyte(edges))
# -----------------------------------------------------------------------------
# Challenge 1b: Hough accumulator
# -----------------------------------------------------------------------------

def challenge1b():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Challenge 1b: accumulator -> accumulator_*.png')

    theta_num_bins = 180
    for img_name in img_list:
        edge = io.imread(f"edge_{img_name}.png", as_gray=True)
        H, W = edge.shape
        rho_num_bins = int(2 * np.ceil(np.hypot(H, W)) + 1)

        Hacc = generate_hough_accumulator(edge, theta_num_bins, rho_num_bins)
        io.imsave(f"accumulator_{img_name}.png",
                  img_as_ubyte(Hacc / (Hacc.max() if Hacc.max() > 0 else 1.0)))

# -----------------------------------------------------------------------------
# Challenge 1c: Line finding
# -----------------------------------------------------------------------------

def challenge1c():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Challenge 1c: lines -> line_*.png')

    # same threshold for all three works well; tweak if needed (fraction of max)
    hough_threshold = [0.55, 0.55, 0.55]

    for i, img_name in enumerate(img_list):
        orig_img  = io.imread(f"{img_name}.png")
        hough_img = io.imread(f"accumulator_{img_name}.png", as_gray=True)
        line_img  = line_finder(orig_img, hough_img, hough_threshold[i])
        io.imsave(f"line_{img_name}.png", img_as_ubyte(line_img))

# -----------------------------------------------------------------------------
# Challenge 1d: Line segment finding
# -----------------------------------------------------------------------------

def challenge1d():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Challenge 1d: segments -> linedetected_*.png')

    hough_threshold = [0.55, 0.55, 0.55]  # same idea; adjust per image if needed

    for i, img_name in enumerate(img_list):
        orig_img  = io.imread(f"{img_name}.png")
        hough_img = io.imread(f"accumulator_{img_name}.png", as_gray=True)
        seg_img   = line_segment_finder(orig_img, hough_img, hough_threshold[i])
        io.imsave(f"linedetected_{img_name}.png", img_as_ubyte(seg_img))

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_hw2(*sys.argv[1:])
