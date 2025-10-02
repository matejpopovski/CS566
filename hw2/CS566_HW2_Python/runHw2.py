import sys
import numpy as np
from skimage import io, feature, img_as_ubyte
from pathlib import Path
from hw2_walkthrough1 import hw2_walkthrough1
from generateHoughAccumulator import generate_hough_accumulator
from lineFinder import line_finder
from lineSegmentFinder import line_segment_finder
from signAcademicPolicy import sign_academic_honesty_policy


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def run_hw2(*args):
    """
    run_hw2 is the "main" interface that lets you execute all the walkthroughs
    and challenges in homework 2.

    Usage:
        run_hw2()                  -> list all registered functions
        run_hw2('function_name')   -> execute a specific test
        run_hw2('all')             -> execute all registered functions
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
    print('Todo: challenge1a')
    for img_name in img_list:
        img = io.imread(f"{img_name}.png", as_gray=True)
    
        # edge_img = ?? # TODO

        # io.imsave(f"edge_{img_name}.png", img_as_ubyte(edge_img))

# -----------------------------------------------------------------------------
# Challenge 1b: Hough accumulator
# -----------------------------------------------------------------------------


def challenge1b():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Todo: challenge1b')
    # rho_num_bins = ??  # TODO
    # theta_num_bins = ??  # TODO

    # for img_name in img_list:
    #     img = io.imread(f"edge_{img_name}.png", as_gray=True)
    #     hough_accumulator = generate_hough_accumulator(img,
    #                                                  theta_num_bins,
    #                                                  rho_num_bins)
    #     io.imsave(f"accumulator_{img_name}.png",
    #               img_as_ubyte(hough_accumulator / np.max(hough_accumulator)))

# -----------------------------------------------------------------------------
# Challenge 1c: Line finding
# -----------------------------------------------------------------------------


def challenge1c():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Todo: challenge1c')
    # hough_threshold = [??, ??, ??]  # TODO

    # for i, img_name in enumerate(img_list):
    #     orig_img = io.imread(f"{img_name}.png")
    #     hough_img = io.imread(f"accumulator_{img_name}.png", as_gray=True)
    #     line_img = line_finder(orig_img, hough_img, hough_threshold[i])
    #     line_img = np.flip(line_img, axis=0)
    #     io.imsave(f"line_{img_name}.png", img_as_ubyte(line_img))

# -----------------------------------------------------------------------------
# Challenge 1d: Line segment finding
# -----------------------------------------------------------------------------


def challenge1d():
    img_list = ["hough_1", "hough_2", "hough_3"]
    print('Todo: challenge1d')
    # hough_threshold = [??, ??, ??]  # TODO

    # for i, img_name in enumerate(img_list):
    #     orig_img = io.imread(f"{img_name}.png")
    #     hough_img = io.imread(f"accumulator_{img_name}.png", as_gray=True)
    #     line_img = line_segment_finder(orig_img, hough_img, hough_threshold[i])
    #     line_img = np.flip(line_img, axis=0)
    #     io.imsave(f"linedetected_{img_name}.png", img_as_ubyte(line_img))

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_hw2(*sys.argv[1:])
