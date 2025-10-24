import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from signAcademicPolicy import sign_academic_honesty_policy
from computeHomography import compute_homography
from applyHomography import apply_homography
from showCorrespondence import show_correspondence
from backwardWarpImg import backward_warp_img
from genSIFTMatches import gen_sift_matches
from blendImagePair import blend_image_pair
from stitchImg import stitch_img
from runRANSAC import run_ransac
from getPointsFromUser import get_points_from_user

# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def run_hw3(*args):
    """
    run_hw3 is the 'main' interface that lists and runs the registered functions.

    Usage:
    run_hw3()                   : list all registered functions
    run_hw3('function_name')    : execute a specific test
    run_hw3('all')              : execute all registered functions

    Note that this file also serves as the specifications for the functions 
    you are asked to implement. In some cases, your submissions will be autograded. 
    Thus, it is critical that you adhere to all the specified function signatures.

    Before your submssion, make sure you can run runHw3('all') 
    without any error.
    """

    fun_handles = {
        "honesty": honesty,
        "debug1": debug1,
        "challenge1a": challenge1a,
        "challenge1b": challenge1b,
        "challenge1c": challenge1c,
        "challenge1d": challenge1d,
        "challenge1e": challenge1e,
    }

    if not args:
        print("Registered functions:")
        for k in fun_handles:
            print(" -", k)
        return

    if args[0] == "all":
        for name, f in fun_handles.items():
            print(f"Running {name}...")
            f()
        return

    if args[0] in fun_handles:
        print(f"Running {args[0]}...")
        fun_handles[args[0]]()
    else:
        print("Unknown function name:", args[0])


# --------------------------------------------------------------------------
# Academic Honesty Policy
# --------------------------------------------------------------------------
def honesty():
    # Replace with your name and uni
    sign_academic_honesty_policy("full_name", "stu_id")


# --------------------------------------------------------------------------
# Debugging and Challenge Functions
# --------------------------------------------------------------------------
def debug1():
    # Load images
    orig_img = cv2.imread("portrait.png")
    warped_img = cv2.imread("portrait_transformed.png")

    src_pts = get_points_from_user(orig_img, 4, "Click any 4 points")
    dst_pts = get_points_from_user(
        warped_img, 4, "Click corresponding 4 points")

    # Compute homography
    H_3x3 = compute_homography(src_pts, dst_pts)
    # src_pts_nx2 and dest_pts_nx2 are the coordinates of corresponding points
    # of the two images, respectively. src_pts_nx2 and dest_pts_nx2
    # are nx2 matrices, where the first column contains
    # the x coodinates and the second column contains the y coordinates.
    #
    # H, a 3x3 matrix, is the estimated homography that
    # transforms src_pts_nx2 to dest_pts_nx2.

    # Choose another set of points on orig_img for testing.
    # test_pts_nx2 should be an nx2 matrix, where n is the number of points, the
    # first column contains the x coordinates and the second column contains
    # the y coordinates.
    test_pts = get_points_from_user(
        orig_img, 5, "Click 5 test points to visualize the homography")

    # Apply homography
    dest_pts = apply_homography(H_3x3, test_pts)
    # test_pts and dest_pts are the coordinates of corresponding points
    # of the two images, and H_3x3 is the homography.

    result_img = show_correspondence(orig_img, warped_img, test_pts, dest_pts)
    cv2.imwrite("homography_result.png", result_img)


def challenge1a():
    bg_img = cv2.imread("Osaka.png").astype(np.float32) / 255.0
    portrait_img = cv2.imread("portrait_small.png").astype(np.float32) / 255.0

    # -------------------
    # Estimate homography
    # Choose 4 points (image corners work well) on the portrait image, and
    # select their corresponding locations in the bg_img.
    # You might find the getPointsFromUser function used in debug1 useful.
    # -------------------
    portrait_pts = np.zeros((4, 2))  # replace this
    bg_pts = np.zeros((4, 2))  # replace this

    H = np.eye(3)

    # TODO: fill points and compute homography
    # portrait_pts = np.array([[xp1, yp1], [xp2, yp2], [xp3, yp3], [xp4, yp4]])
    # bg_pts = np.array([[xb1, yb1], [xb2, yb2], [xb3, yb3], [xb4, yb4]])
    # H = compute_homography(portrait_pts, bg_pts)
    # 1) Click 4 corners on the PORTRAIT (clockwise), then their targets on the BILLBOARD (clockwise)
    portrait_pts = get_points_from_user(
        portrait_img, 4, "Click 4 corners on the PORTRAIT (clockwise)"
    ).astype(np.float32)

    bg_pts = get_points_from_user(
        bg_img, 4, "Click corresponding 4 points on the BILLBOARD (clockwise)"
    ).astype(np.float32)

    # 2) Compute homography that maps portrait -> billboard
    H = compute_homography(portrait_pts, bg_pts)

    dest_w, dest_h = bg_img.shape[1], bg_img.shape[0]

    # warp the portrait image
    (mask, dest_img) = backward_warp_img(
        portrait_img, np.linalg.inv(H), (dest_w, dest_h))

    # mask should be of the type logical
    # it represents where the portrait is present in the canvas
    # we invert it because we need it to represent where the background image is present
    mask_inv = 1 - mask

    result = bg_img * mask_inv[:, :, None] + dest_img
    cv2.imwrite("Van_Gogh_in_Osaka.png", (result * 255).astype(np.uint8))


def challenge1b():
    # Test RANSAC -- outlier rejection

    imgs = cv2.imread("mountain_left.png")
    imgd = cv2.imread("mountain_center.png")

    xs, xd = gen_sift_matches(imgs, imgd)
    # xs and xd are nx2 matrices (x,y) of matched keypoints

    # BEFORE RANSAC (all raw matches)
    before_img = show_correspondence(imgs, imgd, xs, xd)
    cv2.imwrite("before_ransac.png", before_img)

    # Use RANSAC to reject outliers
    ransac_n = 100        # recommended
    ransac_eps = 3.0      # pixels, recommended
    H_3x3 = np.eye(3)     # placeholder

    (inliers_id, H_3x3) = run_ransac(xs, xd, ransac_n, ransac_eps)

    # ---- quality metrics for RANSAC result ----
    proj_all = apply_homography(H_3x3, xs)         # reproject all src points
    err_all  = np.linalg.norm(proj_all - xd, axis=1)

    if len(inliers_id) > 0:
        err_inl = err_all[inliers_id]

        def _stats(e):
            return (e.mean(), np.median(e), np.percentile(e, 95))

        m_inl, med_inl, p95_inl = _stats(err_inl)
        m_all, med_all, p95_all = _stats(err_all)
        span_x, span_y = np.ptp(xs[inliers_id], axis=0)  # spatial coverage

        print(f"[1b] Inliers: {len(inliers_id)}/{len(xs)} "
              f"({100*len(inliers_id)/len(xs):.1f}%)")
        print(f"[1b] Inlier error: mean={m_inl:.2f}px, median={med_inl:.2f}px, p95={p95_inl:.2f}px")
        print(f"[1b] All error:    mean={m_all:.2f}px, median={med_all:.2f}px, p95={p95_all:.2f}px")
        print(f"[1b] Inlier span:  Δx={span_x:.1f}, Δy={span_y:.1f}")

        # AFTER RANSAC (inliers only)
        after_img = show_correspondence(imgs, imgd, xs[inliers_id], xd[inliers_id])
        cv2.imwrite("after_ransac.png", after_img)
    else:
        print('no correspondence points found.')




def challenge1c():
    # Test image blending
    fish = cv2.imread("escher_fish.png")
    horse = cv2.imread("escher_horsemen.png")

    # Assume masks precomputed or alpha channel
    fish_mask = (fish[..., 2] > 0).astype(
        np.uint8) if fish.shape[2] == 3 else np.ones(fish.shape[:2], np.uint8)
    horse_mask = (horse[..., 2] > 0).astype(
        np.uint8) if horse.shape[2] == 3 else np.ones(horse.shape[:2], np.uint8)

    blended_result = blend_image_pair(  # TODO: Modify BlendImagPair.py code
        fish, fish_mask, horse, horse_mask, mode="blend")
    cv2.imwrite("blended_result.png", blended_result)

    overlay_result = blend_image_pair(  # TODO: Modify BlendImagPair.py code
        fish, fish_mask, horse, horse_mask, mode="overlay")
    cv2.imwrite("overlay_result.png", overlay_result)


def challenge1d():
    # Test image stitching
    imgc = cv2.imread("mountain_center.png").astype(np.float32) / 255.0
    imgl = cv2.imread("mountain_left.png").astype(np.float32) / 255.0
    imgr = cv2.imread("mountain_right.png").astype(np.float32) / 255.0

    # TODO: Modify stitchImg.py code
    stitched_img = stitch_img(imgl, imgc, imgr)
    cv2.imwrite("mountain_panorama.png", (stitched_img * 255).astype(np.uint8))


def challenge1e():
    # User can adapt with their own images
    print("Implement your own panorama with stitch_img()")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_hw3(*sys.argv[1:])



