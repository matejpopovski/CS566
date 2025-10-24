import numpy as np
from genSIFTMatches import gen_sift_matches
from runRANSAC import run_ransac
from backwardWarpImg import backward_warp_img
from blendImagePair import blend_image_pair
from bboxCrop import bbox_crop
import matplotlib.pyplot as plt

import numpy as np
import cv2

def stitch_img(*args):
    # GENERAL NOTE: Feel free to change all of this file, not just the
    #               "ADD YOUR CODE HERE" sections. We're just trying to help
    #               get you started.

    # The code below makes sure there is a very large canvas for us to put
    # the stitched image in. It's height is twice the sum of the heights of
    # the input images, and its width is twice the sum of their widths.
    #
    # This makes the image really large, so you might want to crop the
    # blank borders at the end, using the helper function bbox_crop.
    H_stitched = sum([img.shape[0] for img in args])
    W_stitched = sum([img.shape[1] for img in args])

    # Images should be all grayscale or all colour
    assert max([img.shape[2] for img in args]) == min(
        [img.shape[2] for img in args])
    C_stitched = args[0].shape[2]

    stitched_img = np.zeros(
        (H_stitched, W_stitched, C_stitched), dtype=args[0].dtype)

    # NOTE: The scaffolding code given below assumes that the reference
    # image is the "middle" image in the image sequence passed in through
    # 'varargin'. So if you call this function like:
    #       stitchImg(img_l, img_c, img_r)
    # for images taken left-to-right in the sequence [img_l, img_c, img_r],
    # this code will assume img_c is the reference and it covers the middle
    # of the canvas.
    #
    # If you'd like to do something else, you will have to change the
    # scaffolding code in addition to the new code that you add.
    num_imgs = len(args)
    middle_idx = round((num_imgs + 1) / 2)
    # NOTE: you can put a different value here if you want!
    ref_idx = middle_idx - 1  # adjust for 0-based indexing

    # paste the reference image into the output canvas.
    ref_img = args[ref_idx]
    H_ref, W_ref, _ = ref_img.shape
    ref_start_x = 0 + (W_stitched - W_ref) // 2
    ref_start_y = 0 + (H_stitched - H_ref) // 2

    stitched_img[ref_start_y: ref_start_y + H_ref,
                 ref_start_x: ref_start_x + W_ref,
                 :] = ref_img

    stitch_mask = np.zeros((H_stitched, W_stitched), dtype=bool)
    stitch_mask[ref_start_y: ref_start_y + H_ref,
                ref_start_x: ref_start_x + W_ref] = True

    for n in range(num_imgs):
        if n == ref_idx:
            continue
        img_n = args[n]

        kp_stitched, kp_n = gen_sift_matches(stitched_img, img_n)

        # ---------------------------------------
        # ADD YOUR CODE HERE
        # ---------------------------------------
        # Run RANSAC to find homography (source: img_n -> dest: stitched_img)
        # kp_stitched are points in the current canvas; kp_n are points in img_n
        # Run RANSAC to find homography (source: img_n -> dest: stitched_img)
        # kp_stitched are points in the current canvas; kp_n are points in img_n
        # Run RANSAC to find homography (source: img_n -> dest: stitched_img)
        # --- match current image to the REFERENCE image (not the canvas) ---
        xs, xd = gen_sift_matches(img_n, ref_img)   # xs in img_n, xd in ref_img

        # RANSAC homography: img_n -> ref_img
        inliers_id, H_src_to_ref = run_ransac(xs, xd, ransac_n=100, eps=3.0)
        if (H_src_to_ref is None) or (inliers_id.size < 4):
            continue

        # Normalize homography scale for stability
        if H_src_to_ref[2, 2] != 0:
            H_src_to_ref = H_src_to_ref / H_src_to_ref[2, 2]

        # Compose with the translation that placed the reference on the canvas
        T_ref_to_canvas = np.array([[1.0, 0.0, ref_start_x],
                                    [0.0, 1.0, ref_start_y],
                                    [0.0, 0.0, 1.0]], dtype=np.float64)
        H_src_to_canvas = T_ref_to_canvas @ H_src_to_ref

        # Backward warp needs (canvas -> src)
        H_canvas_to_src = np.linalg.inv(H_src_to_canvas)

        # Warp current image into the canvas frame
        mask_n, warped_n = backward_warp_img(img_n, H_canvas_to_src,
                                            (W_stitched, H_stitched))  # (width, height)
        mask_n = mask_n.astype(np.uint8)

        # Ensure dtypes match for blending
        if warped_n.dtype != stitched_img.dtype:
            warped_n = warped_n.astype(stitched_img.dtype)

        # Blend onto panorama (weighted)
        stitched_img = blend_image_pair(stitched_img, stitch_mask.astype(np.uint8),
                                        warped_n, mask_n, mode="blend")

        # Update accumulated mask
        stitch_mask = stitch_mask | (mask_n > 0)

                # ---------------------------------------
        # END ADD YOUR CODE HERE
        # ---------------------------------------

    # OPTIONAL: remove excess padding from the output
    stitched_img = bbox_crop(stitched_img)

    return stitched_img
