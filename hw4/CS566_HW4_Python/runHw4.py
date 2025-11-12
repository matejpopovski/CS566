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
from skimage import io
from loadFocalStack import load_focal_stack
from generateIndexMap import generate_index_map
from refocusApp import refocus_app
from signAcademicPolicy import sign_academic_honesty_policy


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
    sign_academic_honesty_policy("full_name", "stu_id")


# --------------------------------------------------------------------------
# Tests for Challenge 1: Refocusing Application
# --------------------------------------------------------------------------

def challenge1a():
    # Load the focal stack into memory
    focal_stack_dir = "stack"
    rgb_stack, gray_stack = load_focal_stack(focal_stack_dir)  # TODO: Modify loadFocalStack.py
    # rgb_stack is an mxnx3k matrix, where m and n are the height and width of
    # the image, respectively, and 3k is the number of images in a focal stack
    # multiplied by 3 (each image contains RGB channels).
    #
    # rgb_stack will only be used for the refocusing app viewer (it is not used
    # here).
    #
    # gray_stack is an mxnxk matrix.

    # Specify the (half) window size used for focus measure computation
    half_window_size = 0  # TODO: correct value here.
    # half_window_size = ??
    half_window_size = 16

    # Generate an index map, here we will only use the gray-scale images
    index_map = generate_index_map(gray_stack, half_window_size) # TODO: Modify generateIndexMap.py
    io.imsave("index_map.png", index_map.astype("uint8"))


def challenge1b():
    focal_stack_dir = "stack"
    rgb_stack, gray_stack = load_focal_stack(focal_stack_dir)

    index_map = io.imread("index_map.png")
    refocus_app(rgb_stack, index_map) # TODO: Modify refocusApp.py


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
