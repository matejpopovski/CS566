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
    # Blending demo: compare "overlay" vs weighted "blend"
    import cv2
    import numpy as np

    # Use files that exist in your repo
    img1 = cv2.imread("mountain_left.png")
    img2 = cv2.imread("mountain_center.png")

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read mountain_left.png or mountain_center.png")

    # Make sure same size (resize img2 to img1 if needed)
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    if (H1, W1) != (H2, W2):
        img2 = cv2.resize(img2, (W1, H1), interpolation=cv2.INTER_LINEAR)

    # Build simple overlapping masks:
    # - img1 dominates the LEFT side
    # - img2 dominates the RIGHT side
    H, W = img1.shape[:2]
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask2 = np.zeros((H, W), dtype=np.uint8)

    # 60% / 60% with 20% overlap in the middle
    cut1 = int(0.6 * W)          # img1 active up to 60%
    cut2 = int(0.4 * W)          # img2 active from 40%
    mask1[:, :cut1] = 1
    mask2[:, cut2:] = 1

    # OVERLAY (hard seam)
    out_overlay = blend_image_pair(img1, mask1, img2, mask2, mode="overlay")
    cv2.imwrite("blend_overlay.png", out_overlay)

    # BLEND (distance-transform weighted, smooth seam)
    out_blend = blend_image_pair(img1, mask1, img2, mask2, mode="blend")
    cv2.imwrite("blend_weighted.png", out_blend)

    print("Saved: blend_overlay.png, blend_weighted.png")



def challenge1d():
    import cv2
    imgl = cv2.imread("mountain_left.png")
    imgc = cv2.imread("mountain_center.png")
    imgr = cv2.imread("mountain_right.png")
    if imgl is None or imgc is None or imgr is None:
        raise FileNotFoundError("Missing mountain_left/center/right.png")
    
    pano = stitch_img(imgl, imgc, imgr)
    cv2.imwrite("stitched.png", pano)
    print("Saved: stitched.png")


def challenge1e():
    """
    Three-photo mosaic (img1, img2, img3) without touching stitchImg.py.
    - Loads the three images (jpg/png/jpeg accepted).
    - Normalizes heights (and lightly harmonizes exposure).
    - Scores ALL 6 permutations using SIFT+RANSAC on adjacent pairs.
    - Picks the best order (so the middle becomes the reference in stitch_img).
    - Stitches and saves results.
    """
    import os, glob
    import cv2
    import numpy as np
    from stitchImg import stitch_img
    from genSIFTMatches import gen_sift_matches
    from runRANSAC import run_ransac
    from applyHomography import apply_homography

    # ---------- helpers ----------
    def load_any(basename):
        # Try exact, then common extensions, case-insensitive.
        cand = [basename]
        root, ext = os.path.splitext(basename)
        if ext == "":
            cand += [f"{root}.{e}" for e in ["jpg","jpeg","png","JPG","JPEG","PNG"]]
        for p in cand:
            im = cv2.imread(p)
            if im is not None:
                return im, p
        for p in glob.glob(f"{root}.*"):
            im = cv2.imread(p)
            if im is not None:
                return im, p
        return None, None

    def resize_to_h(im, H):
        if im.shape[0] == H: return im
        W = int(round(im.shape[1] * (H / im.shape[0])))
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

    def harmonize_L(a, b):
        """Match a's L-channel stats to b's to help SIFT on sky/grass."""
        al = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
        bl = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        ma, sa = al[...,0].mean(), al[...,0].std() + 1e-6
        mb, sb = bl[...,0].mean(), bl[...,0].std() + 1e-6
        al[...,0] = (al[...,0] - ma) * (sb/sa) + mb
        al = np.clip(al, 0, 255).astype(np.uint8)
        return cv2.cvtColor(al, cv2.COLOR_LAB2BGR)

    def score_pair(A, B):
        """
        Score how well A->B stitches using SIFT+RANSAC.
        Returns (inlier_ratio, p95_error, combined_score).
        """
        xs, xd = gen_sift_matches(A, B)
        if xs.shape[0] < 8:
            return 0.0, np.inf, -1e9
        inliers_id, H = run_ransac(xs, xd, ransac_n=300, eps=2.5)
        if H is None or inliers_id.size < 4:
            return 0.0, np.inf, -1e9
        proj = apply_homography(H, xs)
        err  = np.linalg.norm(proj - xd, axis=1)
        err_inl = err[inliers_id]
        ratio = inliers_id.size / max(1, xs.shape[0])
        p95   = np.percentile(err_inl, 95) if err_inl.size else np.inf
        # Combined score: prioritize ratio, then penalize error
        comb  = ratio - 0.005 * p95
        return ratio, p95, comb

    # ---------- load images ----------
    img1, p1 = load_any("img1")
    img2, p2 = load_any("img2")
    img3, p3 = load_any("img3")
    if img1 is None or img2 is None or img3 is None:
        raise FileNotFoundError("Could not find img1/img2/img3 with common extensions (.jpg/.png/.jpeg).")

    print(f"[1e] Loaded: {p1}, {p2}, {p3}")

    # Normalize size (same height; cap to ~900px for robust SIFT)
    Ht = min(img1.shape[0], img2.shape[0], img3.shape[0], 900)
    img1 = resize_to_h(img1, Ht)
    img2 = resize_to_h(img2, Ht)
    img3 = resize_to_h(img3, Ht)

    # Light exposure harmonization (match to img2 as anchor)
    img1 = harmonize_L(img1, img2)
    img3 = harmonize_L(img3, img2)

    # ---------- evaluate all orders ----------
    orders = [
        ("(1,2,3)", (img1, img2, img3)),
        ("(1,3,2)", (img1, img3, img2)),
        ("(2,1,3)", (img2, img1, img3)),
        ("(2,3,1)", (img2, img3, img1)),
        ("(3,1,2)", (img3, img1, img2)),
        ("(3,2,1)", (img3, img2, img1)),
    ]

    def score_order(triple):
        A, B, C = triple
        r1, e1, s1 = score_pair(A, B)
        r2, e2, s2 = score_pair(B, C)
        # prefer good inlier ratios on both pairs; penalize big errors
        total = s1 + s2
        return total, (r1, e1), (r2, e2)

    best = None
    for name, triple in orders:
        total, (r1,e1), (r2,e2) = score_order(triple)
        print(f"[1e] order {name}: pair1 ratio={r1:.2f}, p95={e1:.2f}px | "
              f"pair2 ratio={r2:.2f}, p95={e2:.2f}px | score={total:.3f}")
        if (best is None) or (total > best[0]):
            best = (total, name, triple)

    total, name, ordered = best
    print(f"[1e] Chosen L-C-R order: {name}")

    # Save inputs actually used (for submission)
    for i, im in enumerate(ordered, 1):
        cv2.imwrite(f"my_photo_{i}.png", im)

    # ---------- stitch (stitchImg uses middle as reference) ----------
    pano = stitch_img(*ordered)
    cv2.imwrite("my_stitched.png", pano)
    print("Saved: my_photo_1.png, my_photo_2.png, my_photo_3.png, my_stitched.png")







# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_hw3(*sys.argv[1:])



