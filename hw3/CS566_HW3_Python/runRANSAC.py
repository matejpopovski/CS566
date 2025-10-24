import numpy as np
from computeHomography import compute_homography
from applyHomography import apply_homography


def run_ransac(Xs, Xd, ransac_n, eps):
    num_pts = Xs.shape[0]
    pts_id = np.arange(num_pts)
    inliers_id = np.array([])
    H = np.eye(3)  # H placeholder

    for iter in range(ransac_n):
        # ---------------------------
        # START ADDING YOUR CODE HERE
        # ---------------------------
        # Sample 4 unique correspondences
        idx = np.random.choice(pts_id, size=4, replace=False)
        try:
            H_try = compute_homography(Xs[idx], Xd[idx])
        except Exception:
            continue  # degenerate sample

        # Project all source points and compute reprojection error
        proj = apply_homography(H_try, Xs)  # (N,2)
        err = np.linalg.norm(proj - Xd, axis=1)

        inliers = err < eps
        n_inl = int(inliers.sum())
        if n_inl > inliers_id.size:
            inliers_id = np.where(inliers)[0]
            # Refit on inliers if we have enough
            if n_inl >= 4:
                try:
                    H = compute_homography(Xs[inliers], Xd[inliers])
                except Exception:
                    H = H_try
        # ---------------------------
        # END ADDING YOUR CODE HERE
        # ---------------------------
        pass  # placeholder so for loop isn't empty.

    return inliers_id, H
