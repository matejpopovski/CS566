# runHw4.py
# runHw4 is the "main" interface that lets you execute all the challenges
# in homework 4. It lists a set of functions corresponding to the problems
# that need to be solved.
#
# Note that this file also serves as the specifications for the functions
# you are asked to implement. In some cases, your submissions will be
# autograded. Thus, it is critical that you adhere to all the specified
# function signatures.
#
# Before your submission, make sure you can run runHw4('all')
# without any error.
#
# Usage:
# runHw4()                     : list all the registered functions
# runHw4('function_name')      : execute a specific test
# runHw4('all')                : execute all the registered functions

import sys

import matplotlib.pyplot as plt
from generateIndexMap import generate_index_map
from loadFocalStack import load_focal_stack
from refocusApp import refocus_app
from signAcademicPolicy import sign_academic_honesty_policy
from skimage import exposure, io


def runHw4(*args):
    fun_handles = {
        "honesty": honesty,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
    }
    runTests(args, fun_handles)


# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------
def honesty():
    # Type your full name and uni (both in string) to state your agreement
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy("Matej Popovski", "popovski")


# --------------------------------------------------------------------------
# Tests for Challenge 1: Refocusing Application
# --------------------------------------------------------------------------


def challenge1a():
    # Load the focal stack into memory
    focal_stack_dir = "stack"
    rgb_stack, gray_stack = load_focal_stack(focal_stack_dir)
    # rgb_stack is an mxnx3k matrix, where m and n are the height and width of
    # the image, respectively, and 3k is the number of images in a focal stack
    # multiplied by 3 (each image contains RGB channels).
    #
    # rgb_stack will only be used for the refocusing app viewer (it is not used
    # here).
    #
    # gray_stack is an mxnxk matrix.

    # Specify the (half) window size used for focus measure computation
    half_window_size = 12
    # Generate an index map, here we will only use the gray-scale images
    index_map = generate_index_map(gray_stack, half_window_size)
    io.imsave("index_map.png", index_map.astype("uint8"))
    print("index_map.png has been updated and saved.")

    # Normalize for visualization
    index_map_norm = exposure.rescale_intensity(index_map, out_range=(0, 255)).astype(
        "uint8"
    )
    plt.figure()
    plt.imshow(index_map_norm, cmap="gray")
    plt.title("Computed Index Map (Challenge 1a)")
    plt.axis("off")
    plt.show()


def challenge1b():
    focal_stack_dir = "stack"
    rgb_stack, gray_stack = load_focal_stack(focal_stack_dir)

    # Load the index map
    index_map = io.imread("index_map.png")

    # Run the interactive refocusing app
    refocus_app(rgb_stack, index_map)


# --------------------------------------------------------------------------
# Stub definitions for external functions (to be implemented elsewhere)
# --------------------------------------------------------------------------
def runTests(args, fun_handles):
    if not args:
        print("Registered functions:")
        for f in fun_handles:
            print(" -", f)
        return
    arg = args[0]
    if arg == "all":
        for name, func in fun_handles.items():
            print(f"Running {name}()...")
            func()
    elif arg in fun_handles:
        print(f"Running {arg}()...")
        fun_handles[arg]()
    else:
        print("Unknown function name:", arg)


def signAcademicHonestyPolicy(name, uni):
    print(f"Signed Academic Honesty Policy: {name} ({uni})")


# Allow running from command line
if __name__ == "__main__":
    runHw4(*sys.argv[1:])
