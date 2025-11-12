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
    sign_academic_honesty_policy("Matej Popovski", "popovski")


# --------------------------------------------------------------------------
# Tests for Challenge 1: Refocusing Application
# --------------------------------------------------------------------------

def challenge1a():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import exposure
    from loadFocalStack import load_focal_stack
    from generateIndexMap import generate_index_map

    # === 1. Load the focal stack ===
    rgb_stack, gray_stack = load_focal_stack("stack")

    # === 2. Generate the index map using grayscale stack ===
    half_window_size = 12
    index_map = generate_index_map(gray_stack, half_window_size)

    # === 3. Normalize for visualization ===
    index_map_norm = exposure.rescale_intensity(index_map, out_range=(0, 255)).astype("uint8")

    # === 4. Save (overwrite existing file) ===
    plt.imsave("index_map.png", index_map_norm, cmap="gray")
    print("index_map.png has been updated and saved.")

    # === 5. Show the result ===
    plt.figure()
    plt.imshow(index_map_norm, cmap="gray")
    plt.title("Computed Index Map (Challenge 1a)")
    plt.axis("off")
    plt.show()




def challenge1b():
    import numpy as np
    from skimage import io
    from loadFocalStack import load_focal_stack
    from refocusApp import refocus_app

    # === 1. Load the focal stack ===
    rgb_stack, gray_stack = load_focal_stack("stack")

    # === 2. Load the index map ===
    index_map = io.imread("index_map.png")

    # Handle different possible shapes
    if index_map.ndim == 3:
        # If the PNG is saved as RGB, take the first channel
        index_map = index_map[:, :, 0]
    index_map = index_map.astype(np.float32)

    # === 3. Run the interactive refocusing app ===
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
